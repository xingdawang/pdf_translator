from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from functools import wraps
from pathlib import Path
from typing import Callable

from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError

from .config import AppConfig, DEFAULT_DPI
from .exceptions import PDFAnalysisError, ValidationBlockedError
from .models import (
    TaskSettings,
    TranslationTask,
    ValidationIssue,
    ValidationReport,
)
from .layout_renderer import LayoutPreservingRenderer
from .pdf_parser import PDFInspection, PDFParser
from .pdf_renderer import PDFRenderer, RenderOutputs
from .repository import TaskRepository
from .utils import (
    atomic_write_json,
    normalize_local_path,
    safe_stem,
    sha256_file,
    utc_now,
)
from .validation import TranslationValidator
from .xlsx_io import ImportResult, XLSXExporter, XLSXImporter


ProgressCallback = Callable[[int, int], None]


def _task_locked(method):
    """Serialize mutations for one task across web and CLI entry points."""

    @wraps(method)
    def wrapped(self, task_id: str, *args, **kwargs):
        with self.repository.task_guard(task_id):
            return method(self, task_id, *args, **kwargs)

    return wrapped


class TranslationWorkflow:
    def __init__(self, config: AppConfig):
        self.config = config
        self.repository = TaskRepository(config)
        self.validator = TranslationValidator()

    def inspect_source(
        self,
        source_path: str | Path,
        page_start: int = 1,
        page_end: int | None = None,
    ) -> PDFInspection:
        source = Path(normalize_local_path(source_path)).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到 PDF：{source}")
        if source.suffix.lower() != ".pdf":
            raise PDFAnalysisError("请选择扩展名为 .pdf 的文件。")
        parser = PDFParser(
            self.config.minimum_text_characters_per_page,
            ocr_cache_dir=self.config.data_dir / "cache" / "vision-ocr",
        )
        return parser.inspect(
            source,
            page_start=page_start,
            page_end=page_end,
        )

    def create_task(
        self,
        source_path: str | Path,
        settings: TaskSettings | None = None,
        copy_source: bool = False,
        progress: ProgressCallback | None = None,
    ) -> TranslationTask:
        source = Path(normalize_local_path(source_path)).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到 PDF：{source}")
        if source.suffix.lower() != ".pdf":
            raise PDFAnalysisError("请选择扩展名为 .pdf 的文件。")

        task_id = f"{utc_now()[:10].replace('-', '')}-{uuid.uuid4().hex[:10]}"
        task_dir = self.repository.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        actual_source = source
        if copy_source:
            source_dir = task_dir / "source"
            source_dir.mkdir(parents=True, exist_ok=True)
            destination = source_dir / f"{safe_stem(source.name)}.pdf"
            shutil.copy2(source, destination)
            actual_source = destination

        try:
            source_stat = actual_source.stat()
            file_hash = sha256_file(actual_source)
            selected_settings = settings or TaskSettings()
            parser = PDFParser(
                self.config.minimum_text_characters_per_page,
                ocr_cache_dir=self.config.data_dir / "cache" / "vision-ocr",
            )
            parsed = parser.parse(
                actual_source,
                task_id=task_id,
                settings=selected_settings,
                progress=progress,
            )
            warnings = list(parsed.warnings)
            text_ratio = parsed.text_page_count / max(parsed.page_count, 1)
            if text_ratio < self.config.minimum_text_page_ratio:
                warnings.append(
                    f"只有 {text_ratio:.0%} 的页面包含足够文本。该文件可能以图片或扫描页为主。"
                )

            now = utc_now()
            task = TranslationTask(
                task_id=task_id,
                source_path=str(actual_source),
                source_filename=source.name,
                source_file_hash=file_hash,
                source_size_bytes=source_stat.st_size,
                created_at=now,
                updated_at=now,
                status="analyzed",
                page_count=parsed.page_count,
                text_page_count=parsed.text_page_count,
                scanned_page_count=parsed.scanned_page_count,
                image_count=parsed.image_count,
                parser_version=parser.parser_version,
                settings=selected_settings,
                segments=parsed.segments,
                warnings=warnings,
                source_page_count=parsed.source_page_count,
                source_mtime_ns=source_stat.st_mtime_ns,
                document_ir=parsed.document_ir,
            )
            self.repository.save(task)
            return task
        except Exception:
            shutil.rmtree(task_dir, ignore_errors=True)
            raise

    @_task_locked
    def export_packages(
        self,
        task_id: str,
        output_dir: str | Path | None = None,
    ) -> tuple[TranslationTask, list[Path]]:
        task = self.repository.load(task_id)
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else self.repository.task_dir(task_id) / "exports"
        )
        exporter = XLSXExporter(
            max_rows=self.config.max_package_rows,
            max_characters=self.config.max_package_characters,
            max_file_bytes=self.config.max_package_bytes,
        )
        packages = exporter.export(task, destination)
        task.export_packages = packages
        task.status = "package_exported"
        task.updated_at = utc_now()
        self.repository.save(task)
        return task, [destination / package.filename for package in packages]

    @_task_locked
    def import_packages(
        self,
        task_id: str,
        paths: list[str | Path],
    ) -> tuple[TranslationTask, list[ImportResult], ValidationReport]:
        task = self.repository.load(task_id)
        importer = XLSXImporter()
        import_dir = self.repository.task_dir(task_id) / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        results: list[ImportResult] = []
        extra_issues: list[ValidationIssue] = []

        for supplied_path in paths:
            source = Path(supplied_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"找不到翻译结果：{source}")
            if source.suffix.lower() != ".xlsx":
                raise ValueError(f"只支持 .xlsx 翻译结果：{source.name}")
            destination = import_dir / source.name
            if source != destination.resolve():
                shutil.copy2(source, destination)
            result = importer.import_file(task, destination)
            results.append(result)
            extra_issues.extend(result.issues)

        task.review_confirmation = None
        task.outputs = {}
        task.last_generation = None
        repaired_issues = self._apply_source_fallbacks(task, extra_issues)
        report = self.validator.validate(task, repaired_issues)
        task.last_validation = report.to_dict()
        task.status = "ready" if report.can_generate else "needs_review"
        task.updated_at = utc_now()
        self.repository.save(task)
        return task, results, report

    @_task_locked
    def validate(self, task_id: str) -> tuple[TranslationTask, ValidationReport]:
        task = self.repository.load(task_id)
        persisted_technical_issues = [
            ValidationIssue.from_dict(issue)
            for issue in (task.last_validation or {}).get("issues", [])
            if issue.get("severity") == "blocking"
            and issue.get("code") in {"DUPLICATE_ROW_KEY"}
        ]
        repair_issues = self._apply_source_fallbacks(
            task,
            persisted_technical_issues,
        )
        report = self.validator.validate(task, repair_issues)
        task.last_validation = report.to_dict()
        task.status = "ready" if report.can_generate else "needs_review"
        task.updated_at = utc_now()
        self.repository.save(task)
        return task, report

    @_task_locked
    def update_translations(
        self, task_id: str, updates: dict[str, str]
    ) -> tuple[TranslationTask, ValidationReport]:
        task = self.repository.load(task_id)
        by_id = {segment.segment_id: segment for segment in task.translatable_segments}
        unknown = sorted(set(updates) - set(by_id))
        if unknown:
            raise KeyError("未知 Segment ID：" + ", ".join(unknown[:10]))
        for segment_id, translation in updates.items():
            segment = by_id[segment_id]
            segment.translated_text = str(translation).strip()
            segment.status = "translated" if segment.translated_text else "empty"
            segment.fallback_reason = None
        task.review_confirmation = None
        task.outputs = {}
        task.last_generation = None
        report = self.validator.validate(task)
        task.last_validation = report.to_dict()
        task.status = "ready" if report.can_generate else "needs_review"
        task.updated_at = utc_now()
        self.repository.save(task)
        return task, report

    def _apply_source_fallbacks(
        self,
        task: TranslationTask,
        issues: list[ValidationIssue],
    ) -> list[ValidationIssue]:
        missing_ids = {
            issue.segment_id
            for issue in issues
            if issue.code == "MISSING_ROW_KEY" and issue.segment_id
        }
        retained_issues = [
            issue
            for issue in issues
            if issue.code
            not in {"MISSING_ROW_KEY", "MISSING_ROW_KEY_SUMMARY"}
        ]
        fallback_issues: list[ValidationIssue] = []

        for segment in task.translatable_segments:
            reason = segment.fallback_reason
            if segment.status != "source_fallback":
                translation = segment.translated_text.strip()
                if not translation:
                    reason = (
                        "翻译件中缺少对应译文"
                        if segment.segment_id in missing_ids
                        else "Google 返回的译文为空或未导入"
                    )
                else:
                    restored = self.validator.placeholders.restore(
                        translation,
                        segment.placeholders,
                    )
                    if not self.validator.placeholders.output_restore_ok(
                        restored,
                        segment.placeholders,
                        task.settings.ignore_number_warnings,
                    ):
                        reason = (
                            "关键内容无法安全恢复，已保留原英文"
                        )

            if reason:
                segment.translated_text = segment.source_text
                segment.status = "source_fallback"
                segment.fallback_reason = reason
                fallback_issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="SOURCE_FALLBACK",
                        message=reason,
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )

        return retained_issues + fallback_issues

    @_task_locked
    def confirm_review(
        self, task_id: str
    ) -> tuple[TranslationTask, ValidationReport]:
        task, report = self.validate(task_id)
        if not report.can_generate:
            task.review_confirmation = None
            self.repository.save(task)
            return task, report
        task.review_confirmation = {
            "confirmed_at": utc_now(),
            "translated_segments": report.translated_segments,
            "warnings_accepted": report.warnings,
            "source_file_hash": task.source_file_hash,
            "validation_created_at": report.created_at,
        }
        task.status = "ready"
        task.updated_at = utc_now()
        self.repository.save(task)
        return task, report

    def _generate_source_layout_pdf(
        self,
        task: TranslationTask,
        layout_pdf: Path,
        layout_quality_json: Path,
    ) -> tuple[Path, bool, str]:
        """Interleave each selected source page with its layout-preserved translation."""
        output_dir = self.repository.task_dir(task.task_id) / "outputs"
        report_dir = self.repository.task_dir(task.task_id) / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir
            / f"{safe_stem(task.source_filename)}_原文+原版面中文版.pdf"
        )
        manifest_path = report_dir / "source_layout_manifest.json"

        try:
            layout_report = json.loads(
                layout_quality_json.read_text(encoding="utf-8")
            )
            layout_fingerprint = str(
                layout_report["generation_fingerprint"]
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValidationBlockedError(
                "无法确认原版面中文版的生成指纹，请重新生成。"
            ) from exc

        fingerprint_payload = {
            "version": "source-layout-interleave-v2-deduplicated",
            "source_file_hash": task.source_file_hash,
            "source_page_numbers": task.selected_page_numbers,
            "layout_fingerprint": layout_fingerprint,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        if output_path.is_file() and manifest_path.is_file():
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if (
                    manifest.get("generation_fingerprint") == fingerprint
                    and manifest.get("output_size_bytes")
                    == output_path.stat().st_size
                    and self._verify_source_layout_pdf(task, output_path)
                ):
                    return output_path, True, fingerprint
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        source_path = Path(task.source_path)
        source_reader = PdfReader(str(source_path), strict=False)
        translated_reader = PdfReader(str(layout_pdf), strict=False)
        if source_reader.is_encrypted or translated_reader.is_encrypted:
            raise ValidationBlockedError("PDF 已加密，无法生成原文与中文版交错文件。")
        if len(translated_reader.pages) != task.page_count:
            raise ValidationBlockedError(
                "原版面中文版页数与任务范围不一致，请重新生成。"
            )
        page_numbers = task.selected_page_numbers
        if len(page_numbers) != task.page_count or any(
            page_number > len(source_reader.pages)
            for page_number in page_numbers
        ):
            raise ValidationBlockedError("任务页码范围无效，请重新创建任务。")

        writer = PdfWriter()
        for translated_index, page_number in enumerate(page_numbers):
            writer.add_page(source_reader.pages[page_number - 1])
            writer.add_page(translated_reader.pages[translated_index])
        writer.add_metadata(
            {
                "/Title": f"{task.source_filename} - 原文与原版面中文版",
                "/Author": "Local PDF Translator",
                "/Subject": (
                    "原文页与对应原版面中文版逐页交错；"
                    f"Task {task.task_id}"
                ),
            }
        )
        writer.compress_identical_objects()
        temporary_output = output_path.with_suffix(".pdf.tmp")
        with temporary_output.open("wb") as stream:
            writer.write(stream)
            stream.flush()
        temporary_output.replace(output_path)

        if not self._verify_source_layout_pdf(task, output_path):
            raise ValidationBlockedError(
                "原文与原版面中文版的页序或页面尺寸校验失败。"
            )
        atomic_write_json(
            manifest_path,
            {
                **fingerprint_payload,
                "generation_fingerprint": fingerprint,
                "generated_at": utc_now(),
                "output_path": str(output_path),
                "output_size_bytes": output_path.stat().st_size,
                "output_pages": task.page_count * 2,
                "page_order": "source-1, translated-1, source-2, translated-2, ...",
            },
        )
        return output_path, False, fingerprint

    @staticmethod
    def _verify_source_layout_pdf(
        task: TranslationTask,
        output_path: Path,
    ) -> bool:
        try:
            source_reader = PdfReader(str(task.source_path), strict=False)
            output_reader = PdfReader(str(output_path), strict=False)
            if len(output_reader.pages) != task.page_count * 2:
                return False
            for selected_index, page_number in enumerate(
                task.selected_page_numbers
            ):
                source_page = source_reader.pages[page_number - 1]
                source_width = float(source_page.mediabox.width)
                source_height = float(source_page.mediabox.height)
                source_rotation = int(source_page.get("/Rotate", 0) or 0) % 360
                for output_index in (
                    selected_index * 2,
                    selected_index * 2 + 1,
                ):
                    output_page = output_reader.pages[output_index]
                    if (
                        abs(float(output_page.mediabox.width) - source_width)
                        > 0.01
                        or abs(
                            float(output_page.mediabox.height) - source_height
                        )
                        > 0.01
                        or int(output_page.get("/Rotate", 0) or 0) % 360
                        != source_rotation
                    ):
                        return False
            return True
        except (OSError, IndexError, TypeError, ValueError, PyPdfError):
            return False

    @_task_locked
    def generate(
        self,
        task_id: str,
        chinese: bool = True,
        bilingual: bool = True,
        layout: bool = False,
        source_layout: bool = False,
        show_segment_ids: bool = False,
        font_path: str | Path | None = None,
        layout_dpi: int = DEFAULT_DPI,
        progress: ProgressCallback | None = None,
    ) -> tuple[TranslationTask, RenderOutputs]:
        source_task = self.repository.load(task_id)
        self._assert_source_unchanged(source_task)
        task, report = self.validate(task_id)
        if not any((chinese, bilingual, layout, source_layout)):
            raise ValueError("至少选择一种 PDF 输出。")
        outputs = RenderOutputs()
        layout_outputs = None
        source_layout_fingerprint = None
        if chinese or bilingual:
            renderer = PDFRenderer(font_path)
            outputs = renderer.generate(
                task=task,
                validation=report,
                task_dir=self.repository.task_dir(task_id),
                chinese=chinese,
                bilingual=bilingual,
                show_segment_ids=show_segment_ids,
                progress=progress,
            )
        if layout or source_layout:
            layout_task_dir = self.repository.task_dir(task_id)
            if source_layout and not layout:
                layout_task_dir = (
                    layout_task_dir / "cache" / "source-layout"
                )
            layout_outputs = LayoutPreservingRenderer(
                font_path=font_path,
                render_dpi=layout_dpi,
            ).generate(
                task=task,
                validation=report,
                task_dir=layout_task_dir,
                progress=progress,
            )
            outputs.layout_pdf = layout_outputs.layout_pdf
            outputs.layout_quality_json = layout_outputs.quality_json
            outputs.layout_quality_html = layout_outputs.quality_html
        if source_layout and layout_outputs is not None:
            (
                outputs.source_layout_pdf,
                outputs.source_layout_cache_hit,
                source_layout_fingerprint,
            ) = self._generate_source_layout_pdf(
                task,
                layout_outputs.layout_pdf,
                layout_outputs.quality_json,
            )

        task.outputs = {}
        if outputs.chinese_pdf:
            task.outputs["chinese_pdf"] = str(outputs.chinese_pdf)
        if outputs.bilingual_pdf:
            task.outputs["bilingual_pdf"] = str(outputs.bilingual_pdf)
        if layout and outputs.layout_pdf:
            task.outputs["layout_pdf"] = str(outputs.layout_pdf)
        if source_layout and outputs.source_layout_pdf:
            task.outputs["source_layout_pdf"] = str(
                outputs.source_layout_pdf
            )
        task.last_generation = {
            "completed_at": utc_now(),
            "output_mode": (
                "source_layout"
                if source_layout
                else "layout"
                if layout
                else "legacy_reflow"
            ),
            "layout_requested": layout or source_layout,
            "source_layout_requested": source_layout,
            "layout_dpi": layout_dpi if layout or source_layout else None,
            "layout_cache_hit": (
                layout_outputs.cache_hit
                if layout_outputs is not None
                else False
            ),
            "layout_render_workers": (
                layout_outputs.render_workers
                if layout_outputs is not None
                else 0
            ),
            "source_layout_cache_hit": outputs.source_layout_cache_hit,
            "source_layout_fingerprint": source_layout_fingerprint,
            "chinese_requested": chinese,
            "bilingual_requested": bilingual,
        }
        task.status = "generated"
        task.updated_at = utc_now()
        self.repository.save(task)
        return task, outputs

    @_task_locked
    def cleanup_task(self, task_id: str) -> int:
        return self.repository.cleanup_task_files(task_id)

    @_task_locked
    def delete_task(self, task_id: str) -> Path:
        return self.repository.delete_task(task_id)

    def _assert_source_unchanged(self, task: TranslationTask) -> None:
        source = Path(task.source_path)
        if not source.is_file():
            raise ValidationBlockedError(
                f"原始 PDF 已移动或删除，请重新创建任务：{source}"
            )
        stat = source.stat()
        if (
            task.source_mtime_ns is not None
            and stat.st_size == task.source_size_bytes
            and stat.st_mtime_ns == task.source_mtime_ns
        ):
            return

        current_hash = sha256_file(source)
        if current_hash != task.source_file_hash:
            raise ValidationBlockedError(
                "原始 PDF 自任务创建后已经发生变化。为避免把旧译文放到错误位置，"
                "请重新分析该 PDF。"
            )
        task.source_size_bytes = stat.st_size
        task.source_mtime_ns = stat.st_mtime_ns
        task.updated_at = utc_now()
        self.repository.save(task)
