from __future__ import annotations

import html
import hashlib
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .exceptions import (
    FontNotFoundError,
    InsufficientDiskSpaceError,
    ValidationBlockedError,
)
from .models import TranslationTask, ValidationReport
from .placeholders import PlaceholderService, TOKEN_PATTERN
from .utils import atomic_write_json, safe_stem, utc_now


ProgressCallback = Callable[[int, int], None]
_FONT_REGISTRATION_LOCK = threading.Lock()


@dataclass
class RenderOutputs:
    chinese_pdf: Path | None = None
    bilingual_pdf: Path | None = None
    quality_json: Path | None = None
    quality_html: Path | None = None
    translation_pages: int = 0
    layout_pdf: Path | None = None
    source_layout_pdf: Path | None = None
    source_layout_cache_hit: bool = False
    layout_quality_json: Path | None = None
    layout_quality_html: Path | None = None


class ChineseFontResolver:
    font_name_prefix = "PDFTranslatorChinese"

    def __init__(self, explicit_path: str | Path | None = None):
        self.explicit_path = Path(explicit_path).expanduser() if explicit_path else None

    def register(self) -> tuple[str, Path]:
        candidates = [
            self.explicit_path,
            Path(os.getenv("PDF_TRANSLATOR_FONT", "")).expanduser()
            if os.getenv("PDF_TRANSLATOR_FONT")
            else None,
            Path(__file__).resolve().parent / "assets" / "NotoSansCJKsc-Regular.otf",
            Path("/System/Library/Fonts/STHeiti Light.ttc"),
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                resolved = candidate.resolve()
                font_stat = resolved.stat()
                font_digest = hashlib.sha256(
                    (
                        f"{resolved}\0{font_stat.st_size}\0"
                        f"{font_stat.st_mtime_ns}"
                    ).encode("utf-8")
                ).hexdigest()[:12]
                font_name = f"{self.font_name_prefix}_{font_digest}"
                try:
                    with _FONT_REGISTRATION_LOCK:
                        if font_name not in pdfmetrics.getRegisteredFontNames():
                            pdfmetrics.registerFont(
                                TTFont(
                                    font_name,
                                    str(resolved),
                                    subfontIndex=0,
                                )
                            )
                    return font_name, resolved
                except Exception:
                    continue
        raise FontNotFoundError(
            "找不到可嵌入的中文字体。请设置 PDF_TRANSLATOR_FONT 为 "
            "Noto Sans CJK、思源黑体、微软雅黑等字体文件路径。"
        )


class PDFRenderer:
    def __init__(self, font_path: str | Path | None = None):
        self.font_resolver = ChineseFontResolver(font_path)
        self.placeholders = PlaceholderService()

    def generate(
        self,
        task: TranslationTask,
        validation: ValidationReport,
        task_dir: Path,
        chinese: bool = True,
        bilingual: bool = True,
        show_segment_ids: bool = False,
        progress: ProgressCallback | None = None,
    ) -> RenderOutputs:
        if not validation.can_generate:
            raise ValidationBlockedError(
                f"存在 {validation.blocking_errors} 个阻断性错误，不能生成 PDF。"
            )
        source_path = Path(task.source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"原始 PDF 已移动或删除：{source_path}")
        if not chinese and not bilingual:
            raise ValueError("至少选择一种 PDF 输出。")

        output_dir = task_dir / "outputs"
        temp_dir = task_dir / "tmp" / "render"
        output_dir.mkdir(parents=True, exist_ok=True)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        self._check_disk_space(source_path, output_dir, bilingual)

        font_name, font_path = self.font_resolver.register()
        source_reader = PdfReader(str(source_path), strict=False)
        if source_reader.is_encrypted:
            raise ValidationBlockedError("原始 PDF 已加密，无法生成双语文件。")
        expected_source_pages = task.source_page_count or task.page_count
        if len(source_reader.pages) != expected_source_pages:
            raise ValidationBlockedError("原始 PDF 页数已经变化，请重新创建任务。")
        page_numbers = task.selected_page_numbers
        if len(page_numbers) != task.page_count or any(
            page_number > len(source_reader.pages) for page_number in page_numbers
        ):
            raise ValidationBlockedError("任务页码范围无效，请重新创建任务。")

        page_segments: dict[int, list] = {
            page_number: [] for page_number in page_numbers
        }
        placed_segment_ids: list[str] = []
        for segment in task.translatable_segments:
            page_segments.setdefault(segment.page_number, []).append(segment)

        page_pdf_paths: dict[int, Path] = {}
        page_translation_counts: dict[int, int] = {}
        total_translation_pages = 0

        for selected_index, page_number in enumerate(page_numbers, start=1):
            source_page = source_reader.pages[page_number - 1]
            width = float(source_page.mediabox.width)
            height = float(source_page.mediabox.height)
            page_size = self._safe_page_size(width, height)
            page_path = temp_dir / f"translation_page_{page_number:05d}.pdf"
            ids = self._render_source_page_translation(
                page_path,
                source_filename=task.source_filename,
                source_page_number=page_number,
                segments=page_segments.get(page_number, []),
                page_size=page_size,
                font_name=font_name,
                show_segment_ids=show_segment_ids,
                ignore_number_warnings=task.settings.ignore_number_warnings,
            )
            placed_segment_ids.extend(ids)
            page_reader = PdfReader(str(page_path), strict=False)
            page_count = len(page_reader.pages)
            page_pdf_paths[page_number] = page_path
            page_translation_counts[page_number] = page_count
            total_translation_pages += page_count
            if progress:
                progress(selected_index, task.page_count)

        stem = safe_stem(task.source_filename)
        chinese_path = output_dir / f"{stem}_中文阅读版.pdf" if chinese else None
        bilingual_path = output_dir / f"{stem}_双语核对版.pdf" if bilingual else None

        if chinese_path:
            self._write_chinese_pdf(
                task,
                chinese_path,
                temp_dir,
                page_pdf_paths,
                font_name,
            )
        if bilingual_path:
            self._write_bilingual_pdf(
                task,
                source_reader,
                bilingual_path,
                page_pdf_paths,
            )

        expected_ids = {
            segment.segment_id for segment in task.translatable_segments
        }
        missing_ids = sorted(expected_ids - set(placed_segment_ids))
        unrecovered = self._find_unrecovered_tokens(task)
        output_checks = {}
        if chinese_path:
            output_checks["chinese_pdf"] = self._verify_pdf(chinese_path, 1)
        if bilingual_path:
            output_checks["bilingual_pdf"] = self._verify_pdf(
                bilingual_path,
                task.page_count + total_translation_pages,
            )

        quality_payload = {
            "task_id": task.task_id,
            "generated_at": utc_now(),
            "source_filename": task.source_filename,
            "source_pages": task.page_count,
            "source_page_numbers": page_numbers,
            "segments": len(task.translatable_segments),
            "translated_segments": validation.translated_segments,
            "source_fallback_segments": len(task.source_fallback_segments),
            "source_fallback_segment_ids": [
                segment.segment_id for segment in task.source_fallback_segments
            ],
            "translation_pages": total_translation_pages,
            "missing_segments": len(missing_ids),
            "missing_segment_ids": missing_ids,
            "placeholder_errors": len(unrecovered),
            "unrecovered_placeholders": unrecovered,
            "overlap_errors": 0,
            "overflow_errors": 0,
            "layout_strategy": "single-column-flow",
            "layout_note": (
                "正文通过流式分页排版，未使用固定文本框；因此不会因固定文本框产生重叠。"
            ),
            "font": str(font_path),
            "validation": validation.to_dict(),
            "page_translation_counts": {
                str(key): value for key, value in page_translation_counts.items()
            },
            "outputs": output_checks,
            "status": (
                "passed"
                if not missing_ids and not unrecovered
                else "failed"
            ),
        }
        quality_json = task_dir / "reports" / "layout_quality_report.json"
        quality_json.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(quality_json, quality_payload)
        quality_html = task_dir / "reports" / "translation_report.html"
        self._write_html_report(quality_html, quality_payload)

        if missing_ids or unrecovered:
            raise ValidationBlockedError(
                "生成后质量检查失败，请查看 layout_quality_report.json。"
            )

        shutil.rmtree(temp_dir)
        return RenderOutputs(
            chinese_pdf=chinese_path,
            bilingual_pdf=bilingual_path,
            quality_json=quality_json,
            quality_html=quality_html,
            translation_pages=total_translation_pages,
        )

    def _render_source_page_translation(
        self,
        output_path: Path,
        source_filename: str,
        source_page_number: int,
        segments: list,
        page_size: tuple[float, float],
        font_name: str,
        show_segment_ids: bool,
        ignore_number_warnings: bool,
    ) -> list[str]:
        width, height = page_size
        margin_x = max(13 * mm, width * 0.055)
        top_margin = max(18 * mm, height * 0.07)
        bottom_margin = max(16 * mm, height * 0.06)
        document = SimpleDocTemplate(
            str(output_path),
            pagesize=page_size,
            rightMargin=margin_x,
            leftMargin=margin_x,
            topMargin=top_margin,
            bottomMargin=bottom_margin,
            allowSplitting=True,
            title=f"{source_filename} - 原第 {source_page_number} 页译文",
            author="Local PDF Translator",
        )
        styles = self._styles(font_name)
        story: list = [
            Paragraph(f"原第 {source_page_number} 页译文", styles["page_heading"]),
            # Some TTC fonts report unusually tight ascent/descent metrics to
            # ReportLab. A generous explicit spacer prevents a large first
            # translated title from visually touching the page heading.
            Spacer(1, 8 * mm),
        ]
        placed_ids: list[str] = []

        if not segments:
            story.append(
                Paragraph(
                    "本页没有检测到可翻译的文本。图片、表格和公式请查看双语核对版中的原文页。",
                    styles["note"],
                )
            )

        for segment in sorted(segments, key=lambda item: item.reading_order):
            restored = self.placeholders.restore(
                segment.translated_text, segment.placeholders
            )
            if not self.placeholders.output_restore_ok(
                restored,
                segment.placeholders,
                ignore_number_warnings,
            ):
                raise ValidationBlockedError(
                    f"{segment.segment_id} 的占位符无法恢复。"
                )
            text = self._paragraph_markup(restored.restored_text)
            if show_segment_ids:
                story.append(
                    Paragraph(segment.segment_id, styles["segment_id"])
                )
            style_key = {
                "title": "title",
                "heading": "heading",
                "caption": "caption",
                "footnote": "footnote",
                "header": "marginal",
                "footer": "marginal",
            }.get(segment.block_type, "body")
            story.append(Paragraph(text, styles[style_key]))
            # Heading/title styles already include their own spacing and use
            # keepWithNext. Inserting a Spacer immediately after them would
            # satisfy that constraint without keeping the following content
            # together, leaving an orphan heading at the bottom of a page.
            if segment.block_type not in {"title", "heading"}:
                story.append(Spacer(1, 2.5 * mm))
            placed_ids.append(segment.segment_id)

        def draw_page(canvas, doc):
            canvas.saveState()
            canvas.setFont(font_name, 7.5)
            canvas.setFillColor(colors.HexColor("#64748B"))
            canvas.drawString(
                margin_x,
                height - 10 * mm,
                f"{source_filename} / 原第 {source_page_number} 页",
            )
            canvas.drawRightString(
                width - margin_x,
                9 * mm,
                f"原第 {source_page_number} 页译文 · {doc.page}",
            )
            canvas.restoreState()

        document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
        return placed_ids

    @staticmethod
    def _styles(font_name: str) -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        return {
            "page_heading": ParagraphStyle(
                "PageHeading",
                parent=base["Heading1"],
                fontName=font_name,
                fontSize=17,
                leading=24,
                textColor=colors.HexColor("#0F172A"),
                alignment=TA_LEFT,
                spaceAfter=6,
                wordWrap="CJK",
            ),
            "title": ParagraphStyle(
                "TranslatedTitle",
                parent=base["Heading1"],
                fontName=font_name,
                fontSize=15,
                leading=22,
                textColor=colors.HexColor("#0F172A"),
                spaceBefore=5,
                spaceAfter=5,
                wordWrap="CJK",
                keepWithNext=True,
            ),
            "heading": ParagraphStyle(
                "TranslatedHeading",
                parent=base["Heading2"],
                fontName=font_name,
                fontSize=13,
                leading=19,
                textColor=colors.HexColor("#1E3A8A"),
                spaceBefore=6,
                spaceAfter=4,
                wordWrap="CJK",
                keepWithNext=True,
            ),
            "body": ParagraphStyle(
                "TranslatedBody",
                parent=base["BodyText"],
                fontName=font_name,
                fontSize=10.5,
                leading=16,
                textColor=colors.HexColor("#111827"),
                alignment=TA_LEFT,
                wordWrap="CJK",
                splitLongWords=True,
                allowWidows=0,
                allowOrphans=0,
            ),
            "caption": ParagraphStyle(
                "TranslatedCaption",
                parent=base["BodyText"],
                fontName=font_name,
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor("#334155"),
                wordWrap="CJK",
            ),
            "footnote": ParagraphStyle(
                "TranslatedFootnote",
                parent=base["BodyText"],
                fontName=font_name,
                fontSize=8.5,
                leading=12,
                textColor=colors.HexColor("#475569"),
                wordWrap="CJK",
            ),
            "marginal": ParagraphStyle(
                "TranslatedMarginal",
                parent=base["BodyText"],
                fontName=font_name,
                fontSize=8.5,
                leading=12,
                textColor=colors.HexColor("#64748B"),
                wordWrap="CJK",
            ),
            "segment_id": ParagraphStyle(
                "SegmentId",
                parent=base["BodyText"],
                fontName="Helvetica",
                fontSize=6.5,
                leading=8,
                textColor=colors.HexColor("#94A3B8"),
                keepWithNext=True,
            ),
            "note": ParagraphStyle(
                "Note",
                parent=base["BodyText"],
                fontName=font_name,
                fontSize=10,
                leading=15,
                textColor=colors.HexColor("#64748B"),
                backColor=colors.HexColor("#F1F5F9"),
                borderPadding=8,
                wordWrap="CJK",
            ),
        }

    def _write_chinese_pdf(
        self,
        task: TranslationTask,
        output_path: Path,
        temp_dir: Path,
        page_pdf_paths: dict[int, Path],
        font_name: str,
    ) -> None:
        cover = temp_dir / "cover.pdf"
        self._render_cover(task, cover, font_name)
        writer = PdfWriter()
        cover_reader = PdfReader(str(cover), strict=False)
        writer.add_page(cover_reader.pages[0])
        readers = [cover_reader]
        for page_number in task.selected_page_numbers:
            reader = PdfReader(str(page_pdf_paths[page_number]), strict=False)
            readers.append(reader)
            for page in reader.pages:
                writer.add_page(page)
        writer.add_metadata(
            {
                "/Title": f"{task.source_filename} - 中文阅读版",
                "/Author": "Local PDF Translator",
                "/Subject": f"Task {task.task_id}",
            }
        )
        self._atomic_write_pdf(writer, output_path)

    def _write_bilingual_pdf(
        self,
        task: TranslationTask,
        source_reader: PdfReader,
        output_path: Path,
        page_pdf_paths: dict[int, Path],
    ) -> None:
        writer = PdfWriter()
        translation_readers: list[PdfReader] = []
        for page_number in task.selected_page_numbers:
            writer.add_page(source_reader.pages[page_number - 1])
            reader = PdfReader(str(page_pdf_paths[page_number]), strict=False)
            translation_readers.append(reader)
            for page in reader.pages:
                writer.add_page(page)
        writer.add_metadata(
            {
                "/Title": f"{task.source_filename} - 双语核对版",
                "/Author": "Local PDF Translator",
                "/Subject": f"Task {task.task_id}",
            }
        )
        self._atomic_write_pdf(writer, output_path)

    @staticmethod
    def _render_cover(
        task: TranslationTask, output_path: Path, font_name: str
    ) -> None:
        document = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=24 * mm,
            leftMargin=24 * mm,
            topMargin=35 * mm,
            bottomMargin=25 * mm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=24,
            leading=34,
            textColor=colors.HexColor("#0F172A"),
            alignment=TA_CENTER,
            wordWrap="CJK",
        )
        body_style = ParagraphStyle(
            "CoverBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=11,
            leading=18,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            wordWrap="CJK",
        )
        story = [
            Spacer(1, 25 * mm),
            Paragraph(html.escape(Path(task.source_filename).stem), title_style),
            Spacer(1, 12 * mm),
            Paragraph("中文阅读版", title_style),
            Spacer(1, 18 * mm),
            Paragraph(
                f"原文件：{html.escape(task.source_filename)}<br/>"
                f"处理范围：{html.escape(task.page_range_label)}<br/>"
                f"翻译段落：{len(task.translatable_segments)}<br/>"
                f"任务编号：{html.escape(task.task_id)}",
                body_style,
            ),
            Spacer(1, 20 * mm),
            Paragraph(
                "本文件采用单栏流式排版。图片、复杂表格和公式请结合双语核对版的原文页查看。",
                body_style,
            ),
        ]
        document.build(story)

    @staticmethod
    def _atomic_write_pdf(writer: PdfWriter, output_path: Path) -> None:
        temporary = output_path.with_suffix(".pdf.tmp")
        with temporary.open("wb") as stream:
            writer.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output_path)

    @staticmethod
    def _safe_page_size(width: float, height: float) -> tuple[float, float]:
        if width < 250 or height < 250 or width > 3000 or height > 3000:
            return A4
        return width, height

    @staticmethod
    def _paragraph_markup(text: str) -> str:
        escaped = html.escape(text.strip())
        return escaped.replace("\n", "<br/>")

    @staticmethod
    def _check_disk_space(
        source_path: Path, output_dir: Path, bilingual: bool
    ) -> None:
        source_size = source_path.stat().st_size
        required = 250 * 1024 * 1024
        if bilingual:
            required += int(source_size * 1.6)
        free = shutil.disk_usage(output_dir).free
        if free < required:
            raise InsufficientDiskSpaceError(
                f"可用磁盘空间不足。预计至少需要 {required / 1024**3:.2f} GB，"
                f"当前只有 {free / 1024**3:.2f} GB。"
            )

    def _find_unrecovered_tokens(self, task: TranslationTask) -> list[str]:
        problems: list[str] = []
        for segment in task.translatable_segments:
            result = self.placeholders.restore(
                segment.translated_text, segment.placeholders
            )
            if (
                not self.placeholders.output_restore_ok(
                    result,
                    segment.placeholders,
                    task.settings.ignore_number_warnings,
                )
                or TOKEN_PATTERN.search(result.restored_text)
            ):
                problems.append(segment.segment_id)
        return problems

    @staticmethod
    def _verify_pdf(path: Path, minimum_pages: int) -> dict:
        reader = PdfReader(str(path), strict=False)
        page_count = len(reader.pages)
        if page_count < minimum_pages:
            raise ValidationBlockedError(
                f"{path.name} 页数异常：{page_count}，预期至少 {minimum_pages}。"
            )
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "page_count": page_count,
            "valid": True,
        }

    @staticmethod
    def _write_html_report(path: Path, payload: dict) -> None:
        validation = payload["validation"]
        issues = validation.get("issues", [])
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(issue.get('severity', ''))}</td>"
            f"<td>{html.escape(issue.get('code', ''))}</td>"
            f"<td>{html.escape(issue.get('segment_id') or '')}</td>"
            f"<td>{html.escape(issue.get('message', ''))}</td>"
            "</tr>"
            for issue in issues
        )
        document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>翻译检查报告</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#0f172a}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px}}
.value{{font-size:28px;font-weight:700}} table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;border-bottom:1px solid #e2e8f0;padding:10px;vertical-align:top}}
th{{background:#1d4ed8;color:white}} .passed{{color:#15803d;font-weight:700}}
</style>
</head>
<body>
<h1>翻译检查报告</h1>
<p>{html.escape(payload['source_filename'])}</p>
<p class="passed">质量状态：{html.escape(payload['status'])}</p>
<div class="cards">
<div class="card"><div>原文页数</div><div class="value">{payload['source_pages']}</div></div>
<div class="card"><div>翻译段落</div><div class="value">{payload['translated_segments']}</div></div>
<div class="card"><div>阻断错误</div><div class="value">{validation['blocking_errors']}</div></div>
<div class="card"><div>警告</div><div class="value">{validation['warnings']}</div></div>
</div>
<h2>排版</h2>
<p>策略：单栏流式排版。文字空间不足时自动增加页面，不缩小到不可读字号。</p>
<p>生成译文页：{payload['translation_pages']}；缺失段落：{payload['missing_segments']}；
未恢复占位符：{payload['placeholder_errors']}。</p>
<h2>检查问题</h2>
<table><thead><tr><th>级别</th><th>代码</th><th>段落</th><th>说明</th></tr></thead>
<tbody>{rows or '<tr><td colspan="4">没有问题</td></tr>'}</tbody></table>
</body></html>"""
        path.write_text(document, encoding="utf-8")
