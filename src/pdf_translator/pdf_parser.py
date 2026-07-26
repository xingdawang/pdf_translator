from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .exceptions import NoTextLayerError, PDFAnalysisError
from .models import Segment, TaskSettings
from .ocr import MacVisionOCR
from .placeholders import PlaceholderService
from .utils import normalize_whitespace, sha256_text


ProgressCallback = Callable[[int, int], None]


@dataclass
class ParsedPDF:
    page_count: int
    source_page_count: int
    text_page_count: int
    scanned_page_count: int
    image_count: int
    segments: list[Segment]
    warnings: list[str]


@dataclass
class PDFInspection:
    filename: str
    size_bytes: int
    page_count: int
    page_start: int
    page_end: int
    sampled_pages: list[int]
    text_layer_pages: list[int]
    image_pages: list[int]
    recommended_mode: str
    status: str
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "sampled_pages": self.sampled_pages,
            "text_layer_pages": self.text_layer_pages,
            "image_pages": self.image_pages,
            "recommended_mode": self.recommended_mode,
            "status": self.status,
            "recommendation": self.recommendation,
        }


@dataclass
class _RawBlock:
    page_number: int
    order: int
    text: str
    bbox: list[float]
    page_width: float
    page_height: float
    max_font_size: float
    median_font_size: float
    font_name: str
    block_type: str = "paragraph"
    erase_bboxes: list[list[float]] | None = None


class PDFParser:
    parser_version = "pymupdf-blocks-v3-table-cells-vision-ocr"

    def __init__(
        self,
        minimum_text_characters_per_page: int = 20,
        ocr_cache_dir: Path | None = None,
    ):
        self.minimum_text_characters_per_page = minimum_text_characters_per_page
        self.ocr_cache_dir = ocr_cache_dir

    def inspect(
        self,
        path: Path,
        page_start: int = 1,
        page_end: int | None = None,
        maximum_samples: int = 5,
    ) -> PDFInspection:
        try:
            import fitz
        except ImportError as exc:
            raise PDFAnalysisError(
                "缺少 PyMuPDF。请先运行 pip install -e . 安装项目依赖。"
            ) from exc

        try:
            document = fitz.open(path)
        except Exception as exc:
            raise PDFAnalysisError(f"PDF 无法打开，文件可能已损坏：{exc}") from exc

        try:
            if document.needs_pass:
                raise PDFAnalysisError("PDF 已加密，当前版本不支持需要密码的文件。")
            if document.page_count <= 0:
                raise PDFAnalysisError("PDF 没有可读取的页面。")
            resolved_start, resolved_end = self._resolve_page_range(
                document.page_count, page_start, page_end
            )
            selected_indexes = list(range(resolved_start - 1, resolved_end))
            sample_indexes = self._sample_page_indexes(
                selected_indexes, maximum_samples
            )
            text_layer_pages: list[int] = []
            image_pages: list[int] = []
            for page_index in sample_indexes:
                page = document.load_page(page_index)
                text = normalize_whitespace(
                    page.get_text(
                        "text",
                        flags=fitz.TEXTFLAGS_TEXT,
                        sort=True,
                    )
                )
                if len(text) >= self.minimum_text_characters_per_page:
                    text_layer_pages.append(page_index + 1)
                try:
                    if page.get_images(full=True):
                        image_pages.append(page_index + 1)
                except Exception:
                    pass

            sampled_pages = [index + 1 for index in sample_indexes]
            if len(text_layer_pages) == len(sampled_pages):
                status = "text"
                recommended_mode = "off"
                recommendation = (
                    "抽样页面均包含可用文字层，可以使用“仅使用 PDF 文字层”。"
                )
            elif text_layer_pages:
                status = "mixed"
                recommended_mode = "vision"
                recommendation = (
                    "抽样页面同时包含文字页和图片页，建议使用“自动识别”。"
                )
            else:
                status = "image"
                recommended_mode = "vision"
                recommendation = (
                    "抽样页面未检测到可用文字层，建议使用“自动识别”。"
                )
            return PDFInspection(
                filename=path.name,
                size_bytes=path.stat().st_size,
                page_count=document.page_count,
                page_start=resolved_start,
                page_end=resolved_end,
                sampled_pages=sampled_pages,
                text_layer_pages=text_layer_pages,
                image_pages=image_pages,
                recommended_mode=recommended_mode,
                status=status,
                recommendation=recommendation,
            )
        finally:
            document.close()

    def parse(
        self,
        path: Path,
        task_id: str,
        settings: TaskSettings,
        progress: ProgressCallback | None = None,
    ) -> ParsedPDF:
        try:
            import fitz
        except ImportError as exc:
            raise PDFAnalysisError(
                "缺少 PyMuPDF。请先运行 pip install -e . 安装项目依赖。"
            ) from exc

        try:
            document = fitz.open(path)
        except Exception as exc:
            raise PDFAnalysisError(f"PDF 无法打开，文件可能已损坏：{exc}") from exc

        try:
            if document.needs_pass:
                raise PDFAnalysisError("PDF 已加密，当前版本不支持需要密码的文件。")
            if document.page_count <= 0:
                raise PDFAnalysisError("PDF 没有可读取的页面。")

            raw_blocks: list[_RawBlock] = []
            text_page_count = 0
            scanned_page_count = 0
            image_count = 0
            ocr_page_count = 0
            ocr_engine = (
                MacVisionOCR(
                    self.ocr_cache_dir or path.parent / ".pdf-translator-ocr",
                    dpi=settings.ocr_dpi,
                )
                if settings.ocr_mode == "vision"
                else None
            )

            page_start, page_end = self._resolve_page_range(
                document.page_count,
                settings.page_start,
                settings.page_end,
            )
            selected_page_indexes = list(range(page_start - 1, page_end))

            for selected_index, page_index in enumerate(
                selected_page_indexes, start=1
            ):
                page = document.load_page(page_index)
                # TEXTFLAGS_TEXT deliberately excludes image bytes. The default
                # dict output can embed decoded images, which is unsafe for
                # image-heavy PDFs hundreds of megabytes in size.
                page_dict = page.get_text(
                    "dict", flags=fitz.TEXTFLAGS_TEXT, sort=True
                )
                page_blocks: list[_RawBlock] = []
                page_text_parts: list[str] = []
                page_image_count = 0

                text_blocks = [
                    block for block in page_dict.get("blocks", []) if block.get("type") == 0
                ]
                try:
                    page_image_count = len(page.get_images(full=True))
                except Exception:
                    page_image_count = 0
                image_count += page_image_count

                block_order = 0
                for block in text_blocks:
                    for fragment in self.text_block_fragments(block):
                        text = normalize_whitespace(str(fragment["text"]))
                        spans = list(fragment["spans"])
                        if not text:
                            continue
                        block_order += 1
                        sizes = [
                            float(span.get("size", 0.0))
                            for span in spans
                            if span.get("size")
                        ]
                        fonts = [
                            str(span.get("font", ""))
                            for span in spans
                            if span.get("font")
                        ]
                        page_text_parts.append(text)
                        page_blocks.append(
                            _RawBlock(
                                page_number=page_index + 1,
                                order=block_order,
                                text=text,
                                bbox=list(fragment["bbox"]),
                                page_width=float(page.rect.width),
                                page_height=float(page.rect.height),
                                max_font_size=max(sizes, default=0.0),
                                median_font_size=(
                                    statistics.median(sizes) if sizes else 0.0
                                ),
                                font_name=(
                                    Counter(fonts).most_common(1)[0][0]
                                    if fonts
                                    else ""
                                ),
                                erase_bboxes=[
                                    list(box)
                                    for box in fragment["line_bboxes"]
                                ],
                            )
                        )

                page_character_count = len("".join(page_text_parts).strip())
                used_ocr = False
                if (
                    page_character_count < self.minimum_text_characters_per_page
                    and page_image_count > 0
                    and ocr_engine is not None
                ):
                    ocr_blocks = ocr_engine.recognize_page(page, page_index + 1)
                    page_blocks = [
                        _RawBlock(
                            page_number=page_index + 1,
                            order=order,
                            text=block.text,
                            bbox=block.bbox,
                            page_width=float(page.rect.width),
                            page_height=float(page.rect.height),
                            max_font_size=block.font_size,
                            median_font_size=block.font_size,
                            font_name="macOS Vision OCR",
                            block_type=block.block_type,
                            erase_bboxes=block.line_bboxes,
                        )
                        for order, block in enumerate(ocr_blocks, start=1)
                    ]
                    page_character_count = sum(
                        len(block.text) for block in page_blocks
                    )
                    used_ocr = bool(page_blocks)
                    ocr_page_count += int(used_ocr)

                if page_character_count >= self.minimum_text_characters_per_page:
                    text_page_count += 1
                elif page_image_count > 0:
                    scanned_page_count += 1

                if not used_ocr:
                    self._classify_page_blocks(page_blocks)
                raw_blocks.extend(page_blocks)
                if progress:
                    progress(selected_index, len(selected_page_indexes))

            self._mark_repeated_marginal_blocks(
                raw_blocks, len(selected_page_indexes)
            )
            segments = self._to_segments(raw_blocks, task_id, settings)
            warnings: list[str] = []
            if scanned_page_count:
                warnings.append(
                    f"检测到 {scanned_page_count} 个疑似扫描或无文本页面；这些页面不会产生译文。"
                )
            if ocr_page_count:
                warnings.append(
                    f"已使用 macOS Vision 对 {ocr_page_count} 个扫描页面执行 OCR；"
                    "复杂多栏页面的阅读顺序需要人工抽查。"
                )
            if not segments:
                if settings.ocr_mode == "off":
                    raise NoTextLayerError(
                        "所选页面没有可用的 PDF 文字层。"
                        "请改用“自动识别”后重新分析。"
                    )
                raise PDFAnalysisError("PDF 中没有检测到可翻译文字，请抽查 OCR 语言和页面质量。")

            return ParsedPDF(
                page_count=len(selected_page_indexes),
                source_page_count=document.page_count,
                text_page_count=text_page_count,
                scanned_page_count=scanned_page_count,
                image_count=image_count,
                segments=segments,
                warnings=warnings,
            )
        finally:
            document.close()

    @staticmethod
    def _resolve_page_range(
        page_count: int,
        page_start: int,
        page_end: int | None,
    ) -> tuple[int, int]:
        resolved_start = max(1, int(page_start))
        resolved_end = page_count if page_end is None else int(page_end)
        if resolved_start > page_count:
            raise PDFAnalysisError(
                f"起始页 {resolved_start} 超出 PDF 总页数 {page_count}。"
            )
        if resolved_end < resolved_start:
            raise PDFAnalysisError("结束页不能小于起始页。")
        if resolved_end > page_count:
            raise PDFAnalysisError(
                f"结束页 {resolved_end} 超出 PDF 总页数 {page_count}。"
            )
        return resolved_start, resolved_end

    @staticmethod
    def _sample_page_indexes(
        selected_indexes: list[int],
        maximum_samples: int,
    ) -> list[int]:
        if len(selected_indexes) <= maximum_samples:
            return selected_indexes
        sample_count = max(2, maximum_samples)
        last_position = len(selected_indexes) - 1
        return sorted(
            {
                selected_indexes[
                    round(position * last_position / (sample_count - 1))
                ]
                for position in range(sample_count)
            }
        )

    @staticmethod
    def _block_text_and_spans(block: dict) -> tuple[str, list[dict]]:
        lines: list[str] = []
        spans: list[dict] = []
        for line in block.get("lines", []):
            line_spans = line.get("spans", [])
            spans.extend(line_spans)
            line_text = "".join(str(span.get("text", "")) for span in line_spans).strip()
            if line_text:
                lines.append(line_text)

        merged = ""
        for line in lines:
            if not merged:
                merged = line
            elif merged.endswith("-") and line[:1].islower():
                merged = merged[:-1] + line
            else:
                merged += " " + line
        return merged, spans

    @classmethod
    def text_block_fragments(cls, block: dict) -> list[dict[str, object]]:
        """Split one PyMuPDF block into visual table cells when appropriate.

        PyMuPDF often returns every cell in a table row as a separate ``line``
        inside one wide text block. Treating that union as a single paragraph
        erases column rules and pushes all translated text to the left. Lines
        that share the same vertical band but occupy separate horizontal bands
        are therefore emitted as independent fragments. Ordinary wrapped
        paragraphs keep their original block geometry.
        """

        visual_lines: list[dict[str, object]] = []
        for line in block.get("lines", []):
            spans = [
                span
                for span in line.get("spans", [])
                if str(span.get("text", "")).strip()
            ]
            text = "".join(str(span.get("text", "")) for span in spans).strip()
            if not text:
                continue
            supplied_bbox = line.get("bbox")
            if not supplied_bbox and spans:
                supplied_bbox = (
                    min(float(span["bbox"][0]) for span in spans),
                    min(float(span["bbox"][1]) for span in spans),
                    max(float(span["bbox"][2]) for span in spans),
                    max(float(span["bbox"][3]) for span in spans),
                )
            if not supplied_bbox:
                continue
            bbox = [round(float(value), 2) for value in supplied_bbox]
            visual_lines.append({"text": text, "spans": spans, "bbox": bbox})

        full_text, full_spans = cls._block_text_and_spans(block)
        block_bbox = [
            round(float(value), 2)
            for value in block.get("bbox", (0, 0, 0, 0))
        ]
        fallback = [
            {
                "text": full_text,
                "spans": full_spans,
                "bbox": block_bbox,
                "line_bboxes": [block_bbox],
            }
        ]
        if len(visual_lines) < 2:
            return fallback

        groups: list[list[dict[str, object]]] = []
        for line in sorted(visual_lines, key=lambda item: float(item["bbox"][0])):
            x0, _, x1, _ = [float(value) for value in line["bbox"]]
            matching: list[int] = []
            for index, group in enumerate(groups):
                gx0 = min(float(item["bbox"][0]) for item in group)
                gx1 = max(float(item["bbox"][2]) for item in group)
                overlap = min(x1, gx1) - max(x0, gx0)
                gap = max(x0, gx0) - min(x1, gx1)
                if overlap > 0.5 or gap <= 1.5:
                    matching.append(index)
            if not matching:
                groups.append([line])
                continue
            destination = matching[0]
            groups[destination].append(line)
            for index in reversed(matching[1:]):
                groups[destination].extend(groups.pop(index))

        if len(groups) < 2:
            return fallback
        shared_top = max(
            min(float(item["bbox"][1]) for item in group) for group in groups
        )
        shared_bottom = min(
            max(float(item["bbox"][3]) for item in group) for group in groups
        )
        median_height = statistics.median(
            float(item["bbox"][3]) - float(item["bbox"][1])
            for item in visual_lines
        )
        if shared_bottom - shared_top < max(1.0, median_height * 0.2):
            return fallback

        fragments: list[dict[str, object]] = []
        for group in sorted(
            groups,
            key=lambda items: min(float(item["bbox"][0]) for item in items),
        ):
            ordered = sorted(
                group,
                key=lambda item: (
                    float(item["bbox"][1]),
                    float(item["bbox"][0]),
                ),
            )
            group_text = ""
            spans: list[dict] = []
            for item in ordered:
                line_text = str(item["text"])
                if not group_text:
                    group_text = line_text
                elif group_text.endswith("-") and line_text[:1].islower():
                    group_text = group_text[:-1] + line_text
                else:
                    group_text += " " + line_text
                spans.extend(item["spans"])
            line_bboxes = [list(item["bbox"]) for item in ordered]
            fragments.append(
                {
                    "text": group_text,
                    "spans": spans,
                    "bbox": [
                        round(min(box[0] for box in line_bboxes), 2),
                        round(min(box[1] for box in line_bboxes), 2),
                        round(max(box[2] for box in line_bboxes), 2),
                        round(max(box[3] for box in line_bboxes), 2),
                    ],
                    "line_bboxes": line_bboxes,
                }
            )
        return fragments

    @staticmethod
    def _classify_page_blocks(blocks: list[_RawBlock]) -> None:
        sizes = [
            block.median_font_size for block in blocks if block.median_font_size > 0
        ]
        page_median = statistics.median(sizes) if sizes else 10.0
        for block in blocks:
            text = block.text.strip()
            y0, y1 = block.bbox[1], block.bbox[3]
            top_ratio = y0 / max(block.page_height, 1)
            bottom_ratio = y1 / max(block.page_height, 1)

            if (
                re.fullmatch(
                    r"(?:page\s*)?\d+(?:\s*/\s*\d+)?",
                    text,
                    re.IGNORECASE,
                )
                and (top_ratio < 0.08 or bottom_ratio > 0.92)
            ):
                block.block_type = "page_number"
            elif re.match(r"^(?:figure|fig\.|table)\s+\d+", text, re.IGNORECASE):
                block.block_type = "caption"
            elif (
                block.page_number == 1
                and block.order <= 3
                and block.max_font_size >= max(16.0, page_median * 1.55)
            ):
                block.block_type = "title"
            elif (
                len(text) <= 180
                and block.max_font_size >= max(12.0, page_median * 1.25)
            ):
                block.block_type = "heading"
            elif top_ratio < 0.035:
                block.block_type = "header"
            elif bottom_ratio > 0.965:
                block.block_type = "footer"
            elif bottom_ratio > 0.82 and block.median_font_size < page_median * 0.9:
                block.block_type = "footnote"
            else:
                block.block_type = "paragraph"

    @staticmethod
    def _mark_repeated_marginal_blocks(
        blocks: list[_RawBlock], page_count: int
    ) -> None:
        candidates = [
            normalize_whitespace(block.text).casefold()
            for block in blocks
            if block.block_type in {"header", "footer"} and len(block.text) <= 200
        ]
        counts = Counter(candidates)
        threshold = max(2, math.ceil(page_count * 0.35))
        repeated = {text for text, count in counts.items() if count >= threshold}
        for block in blocks:
            normalized = normalize_whitespace(block.text).casefold()
            if normalized in repeated and block.block_type not in {"header", "footer"}:
                block.block_type = (
                    "header"
                    if block.bbox[1] / max(block.page_height, 1) < 0.5
                    else "footer"
                )

    @staticmethod
    def _to_segments(
        blocks: list[_RawBlock], task_id: str, settings: TaskSettings
    ) -> list[Segment]:
        placeholder_service = PlaceholderService()
        segments: list[Segment] = []
        page_orders: Counter[int] = Counter()
        translation_index = 0

        for block in blocks:
            page_orders[block.page_number] += 1
            order = page_orders[block.page_number]
            segment_id = f"P{block.page_number:04d}-S{order:06d}"
            should_translate = True
            if block.block_type == "header" and not settings.translate_headers:
                should_translate = False
            elif block.block_type == "footer" and not settings.translate_footers:
                should_translate = False
            elif block.block_type == "page_number" and settings.ignore_page_numbers:
                should_translate = False
            elif not re.search(r"[A-Za-z]", block.text):
                # Numeric labels, formulas made only from punctuation, and
                # page references should remain untouched. Sending them
                # through document translation only creates false overflow
                # warnings without adding Chinese content.
                should_translate = False
            elif re.fullmatch(
                r"[A-Z]", normalize_whitespace(block.text)
            ):
                # Standalone alphabetic index dividers (A, B, C...) are
                # navigation markers, not prose. Some translators turn them
                # into unrelated Chinese words, so preserve the source glyph.
                should_translate = False

            if should_translate:
                translation_index += 1
                row_key = f"{translation_index:08d}"
                protected = placeholder_service.protect(
                    block.text,
                    task_id=task_id,
                    segment_id=segment_id,
                    protected_terms=settings.protected_terms,
                )
            else:
                row_key = ""
                protected = placeholder_service.protect(
                    block.text,
                    task_id=task_id,
                    segment_id=segment_id,
                    protected_terms=[],
                )

            segments.append(
                Segment(
                    segment_id=segment_id,
                    row_key=row_key,
                    page_number=block.page_number,
                    reading_order=order,
                    block_type=block.block_type,
                    source_text=block.text,
                    protected_text=protected.protected_text,
                    source_text_hash=sha256_text(block.text),
                    bbox=block.bbox,
                    erase_bboxes=block.erase_bboxes or [block.bbox],
                    font_name=block.font_name,
                    font_size=round(block.median_font_size, 2),
                    should_translate=should_translate,
                    placeholders=protected.placeholders,
                    status="pending" if should_translate else "skipped",
                )
            )
        return segments
