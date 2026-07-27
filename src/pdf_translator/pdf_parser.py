from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .exceptions import NoTextLayerError, PDFAnalysisError
from .models import (
    DocumentIR,
    PageIR,
    ParagraphAnchor,
    Segment,
    TaskSettings,
)
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
    document_ir: DocumentIR


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
    column_id: int = 0
    is_table_cell: bool = False
    rotation_degrees: float = 0.0
    source_kind: str = "pdf_text"


class PDFParser:
    parser_version = "pymupdf-blocks-v4-oriented-ir-vision-ocr"

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
                recommendation = "抽样页面同时包含文字页和图片页，建议使用“自动识别”。"
            else:
                status = "image"
                recommended_mode = "vision"
                recommendation = "抽样页面未检测到可用文字层，建议使用“自动识别”。"
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
            ir_pages: list[PageIR] = []
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

            for selected_index, page_index in enumerate(selected_page_indexes, start=1):
                page = document.load_page(page_index)
                ir_pages.append(
                    PageIR(
                        page_number=page_index + 1,
                        width=round(float(page.rect.width), 3),
                        height=round(float(page.rect.height), 3),
                        rotation=int(page.rotation or 0) % 360,
                        mediabox=[
                            round(float(value), 3)
                            for value in (
                                page.mediabox.x0,
                                page.mediabox.y0,
                                page.mediabox.x1,
                                page.mediabox.y1,
                            )
                        ],
                        cropbox=[
                            round(float(value), 3)
                            for value in (
                                page.cropbox.x0,
                                page.cropbox.y0,
                                page.cropbox.x1,
                                page.cropbox.y1,
                            )
                        ],
                    )
                )
                # TEXTFLAGS_TEXT deliberately excludes image bytes. The default
                # dict output can embed decoded images, which is unsafe for
                # image-heavy PDFs hundreds of megabytes in size.
                page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, sort=True)
                page_blocks: list[_RawBlock] = []
                page_text_parts: list[str] = []
                page_image_count = 0

                text_blocks = [
                    block
                    for block in page_dict.get("blocks", [])
                    if block.get("type") == 0
                ]
                try:
                    page_image_count = len(page.get_images(full=True))
                except Exception:
                    page_image_count = 0
                image_count += page_image_count

                block_order = 0
                for block in text_blocks:
                    fragments = self.text_block_fragments(block)
                    is_table_row = len(fragments) > 1 and all(
                        abs(float(fragment.get("rotation_degrees", 0.0))) < 5.0
                        for fragment in fragments
                    )
                    for fragment in fragments:
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
                                    Counter(fonts).most_common(1)[0][0] if fonts else ""
                                ),
                                erase_bboxes=[
                                    list(box) for box in fragment["line_bboxes"]
                                ],
                                is_table_cell=is_table_row,
                                rotation_degrees=float(
                                    fragment.get("rotation_degrees", 0.0)
                                ),
                                source_kind="pdf_text",
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
                            rotation_degrees=0.0,
                            source_kind="ocr",
                        )
                        for order, block in enumerate(ocr_blocks, start=1)
                    ]
                    page_character_count = sum(len(block.text) for block in page_blocks)
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

            self._mark_repeated_marginal_blocks(raw_blocks, len(selected_page_indexes))
            paragraph_groups = self._assemble_paragraph_groups(raw_blocks)
            segments = self._to_segments(
                paragraph_groups,
                task_id,
                settings,
                stable_row_keys=True,
            )
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
                        "所选页面没有可用的 PDF 文字层。请改用“自动识别”后重新分析。"
                    )
                raise PDFAnalysisError(
                    "PDF 中没有检测到可翻译文字，请抽查 OCR 语言和页面质量。"
                )

            return ParsedPDF(
                page_count=len(selected_page_indexes),
                source_page_count=document.page_count,
                text_page_count=text_page_count,
                scanned_page_count=scanned_page_count,
                image_count=image_count,
                segments=segments,
                warnings=warnings,
                document_ir=DocumentIR(pages=ir_pages),
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
                selected_indexes[round(position * last_position / (sample_count - 1))]
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
            line_text = "".join(
                str(span.get("text", "")) for span in line_spans
            ).strip()
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

    @staticmethod
    def _normalize_rotation(value: float) -> float:
        normalized = (float(value) + 180.0) % 360.0 - 180.0
        return 0.0 if abs(normalized) < 0.05 else normalized

    @classmethod
    def _line_rotation(cls, line: dict) -> float:
        direction = line.get("dir", (1.0, 0.0))
        try:
            dx, dy = float(direction[0]), float(direction[1])
        except (IndexError, TypeError, ValueError):
            return 0.0
        if abs(dx) + abs(dy) < 0.001:
            return 0.0
        return cls._normalize_rotation(math.degrees(math.atan2(dy, dx)))

    @classmethod
    def _dominant_rotation(cls, lines: list[dict[str, object]]) -> float:
        if not lines:
            return 0.0
        buckets: Counter[int] = Counter()
        for line in lines:
            rotation = cls._normalize_rotation(float(line.get("rotation_degrees", 0.0)))
            bucket = int(round(rotation / 5.0) * 5)
            buckets[bucket] += max(
                1, len(normalize_whitespace(str(line.get("text", ""))))
            )
        return float(buckets.most_common(1)[0][0])

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
        for source_order, line in enumerate(block.get("lines", [])):
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
            visual_lines.append(
                {
                    "text": text,
                    "spans": spans,
                    "bbox": bbox,
                    "rotation_degrees": cls._line_rotation(line),
                    "source_order": source_order,
                }
            )

        full_text, full_spans = cls._block_text_and_spans(block)
        block_bbox = [
            round(float(value), 2) for value in block.get("bbox", (0, 0, 0, 0))
        ]
        dominant_rotation = cls._dominant_rotation(visual_lines)
        fallback = [
            {
                "text": full_text,
                "spans": full_spans,
                "bbox": block_bbox,
                "line_bboxes": (
                    [list(line["bbox"]) for line in visual_lines] or [block_bbox]
                ),
                "rotation_degrees": dominant_rotation,
            }
        ]
        if len(visual_lines) < 2:
            return fallback

        # PyMuPDF can place several unrelated vertical labels into one text
        # block. A union bbox then spans artwork and causes both destructive
        # background repair and oversized replacement text. Keep every
        # oriented line independent; paragraph assembly can still join nearby
        # lines later in logical reading coordinates.
        rotation_deltas = [
            abs(
                cls._normalize_rotation(
                    float(line["rotation_degrees"]) - dominant_rotation
                )
            )
            for line in visual_lines
        ]
        if abs(dominant_rotation) >= 5.0 or max(rotation_deltas) >= 5.0:
            return [
                {
                    "text": str(line["text"]),
                    "spans": list(line["spans"]),
                    "bbox": list(line["bbox"]),
                    "line_bboxes": [list(line["bbox"])],
                    "rotation_degrees": float(line["rotation_degrees"]),
                }
                for line in sorted(
                    visual_lines,
                    key=lambda item: int(item["source_order"]),
                )
            ]

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
            float(item["bbox"][3]) - float(item["bbox"][1]) for item in visual_lines
        )
        if shared_bottom - shared_top < max(1.0, median_height * 0.2):
            ordered_groups = sorted(
                groups,
                key=lambda group: min(float(item["bbox"][0]) for item in group),
            )
            minimum_horizontal_gap = min(
                (
                    min(float(item["bbox"][0]) for item in right)
                    - max(float(item["bbox"][2]) for item in left)
                    for left, right in zip(
                        ordered_groups,
                        ordered_groups[1:],
                    )
                ),
                default=0.0,
            )
            # A marker on one side of an infographic and a multi-line title
            # on the other can be emitted as one PyMuPDF block despite a large
            # empty horizontal gap. Keep those visual clusters independent.
            if minimum_horizontal_gap < max(12.0, median_height * 2.0):
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
                    "rotation_degrees": cls._dominant_rotation(ordered),
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
            logical = PDFParser._logical_bbox(block)
            _, logical_height = PDFParser._logical_page_size(block)
            top_ratio = logical[1] / max(logical_height, 1)
            bottom_ratio = logical[3] / max(logical_height, 1)

            if re.fullmatch(
                r"(?:page\s*)?\d+(?:\s*/\s*\d+)?",
                text,
                re.IGNORECASE,
            ) and (top_ratio < 0.08 or bottom_ratio > 0.92):
                block.block_type = "page_number"
            elif re.match(r"^(?:figure|fig\.|table)\s+\d+", text, re.IGNORECASE):
                block.block_type = "caption"
            elif (
                block.page_number == 1
                and block.order <= 3
                and block.max_font_size >= max(16.0, page_median * 1.55)
            ):
                block.block_type = "title"
            elif len(text) <= 180 and block.max_font_size >= max(
                12.0, page_median * 1.25
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
                logical = PDFParser._logical_bbox(block)
                _, logical_height = PDFParser._logical_page_size(block)
                block.block_type = (
                    "header" if logical[1] / max(logical_height, 1) < 0.5 else "footer"
                )

    @staticmethod
    def _rotations_match(
        first: _RawBlock,
        second: _RawBlock,
        tolerance: float = 5.0,
    ) -> bool:
        delta = PDFParser._normalize_rotation(
            first.rotation_degrees - second.rotation_degrees
        )
        return abs(delta) <= tolerance

    @staticmethod
    def _logical_geometry(
        bbox: list[float],
        page_width: float,
        page_height: float,
        rotation_degrees: float,
    ) -> tuple[list[float], tuple[float, float]]:
        """Return an orientation-normalized bbox and page size.

        ``u`` follows the text baseline and ``v`` follows line stacking. This
        lets the same grouping rules handle horizontal and rotated pages.
        """

        angle = math.radians(rotation_degrees)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        def project(x: float, y: float) -> tuple[float, float]:
            return (
                x * cosine + y * sine,
                -x * sine + y * cosine,
            )

        page_points = [
            project(0.0, 0.0),
            project(page_width, 0.0),
            project(0.0, page_height),
            project(page_width, page_height),
        ]
        page_u0 = min(point[0] for point in page_points)
        page_v0 = min(point[1] for point in page_points)
        page_u1 = max(point[0] for point in page_points)
        page_v1 = max(point[1] for point in page_points)
        x0, y0, x1, y1 = [float(value) for value in bbox]
        box_points = [
            project(x0, y0),
            project(x1, y0),
            project(x0, y1),
            project(x1, y1),
        ]
        logical_bbox = [
            min(point[0] for point in box_points) - page_u0,
            min(point[1] for point in box_points) - page_v0,
            max(point[0] for point in box_points) - page_u0,
            max(point[1] for point in box_points) - page_v0,
        ]
        return logical_bbox, (page_u1 - page_u0, page_v1 - page_v0)

    @classmethod
    def _logical_bbox(cls, block: _RawBlock) -> list[float]:
        bbox, _ = cls._logical_geometry(
            block.bbox,
            block.page_width,
            block.page_height,
            block.rotation_degrees,
        )
        return bbox

    @classmethod
    def _logical_page_size(cls, block: _RawBlock) -> tuple[float, float]:
        _, page_size = cls._logical_geometry(
            block.bbox,
            block.page_width,
            block.page_height,
            block.rotation_degrees,
        )
        return page_size

    @classmethod
    def _assemble_paragraph_groups(
        cls,
        blocks: list[_RawBlock],
    ) -> list[list[_RawBlock]]:
        """Build semantic paragraphs while retaining every visual anchor.

        The rules are intentionally conservative: nearby blocks are joined only
        when their geometry, style and punctuation all suggest continuity.
        Table cells, headings and marginal content remain independent.
        """

        if not blocks:
            return []
        cls._assign_columns(blocks)
        groups_by_page: dict[int, list[list[_RawBlock]]] = {}
        for page_number in sorted({block.page_number for block in blocks}):
            page_blocks = [
                block for block in blocks if block.page_number == page_number
            ]
            page_groups: list[list[_RawBlock]] = []
            mergeable = [
                block
                for block in page_blocks
                if block.block_type in {"paragraph", "footnote"}
                and not block.is_table_cell
            ]
            consumed: set[int] = set()
            for column_id in sorted({block.column_id for block in mergeable}):
                column_blocks = sorted(
                    (block for block in mergeable if block.column_id == column_id),
                    key=lambda item: (
                        cls._logical_bbox(item)[1],
                        cls._logical_bbox(item)[0],
                        item.order,
                    ),
                )
                current: list[_RawBlock] = []
                for block in column_blocks:
                    if current and cls._can_merge_blocks(current[-1], block):
                        current.append(block)
                    else:
                        if current:
                            page_groups.append(current)
                        current = [block]
                    consumed.add(id(block))
                if current:
                    page_groups.append(current)

            page_groups.extend(
                [block] for block in page_blocks if id(block) not in consumed
            )
            page_groups.sort(
                key=lambda group: (
                    min(item.order for item in group),
                    cls._logical_bbox(group[0])[1],
                    cls._logical_bbox(group[0])[0],
                )
            )
            page_groups = cls._merge_cross_column_groups(page_groups)
            groups_by_page[page_number] = page_groups

        page_numbers = sorted(groups_by_page)
        for page_number, next_page_number in zip(page_numbers, page_numbers[1:]):
            if next_page_number != page_number + 1:
                continue
            current_candidates = [
                group
                for grouped_page in groups_by_page.values()
                for group in grouped_page
                if group[-1].page_number == page_number
                and group[-1].block_type == "paragraph"
            ]
            following_candidates = [
                group
                for group in groups_by_page[next_page_number]
                if group[0].block_type == "paragraph"
            ]
            if not current_candidates or not following_candidates:
                continue
            current = max(
                current_candidates,
                key=lambda group: (
                    cls._logical_bbox(group[-1])[3],
                    group[-1].order,
                ),
            )
            following = min(
                following_candidates,
                key=lambda group: (
                    cls._logical_bbox(group[0])[1],
                    group[0].order,
                ),
            )
            if cls._can_continue_across_page(current[-1], following[0]):
                current.extend(following)
                groups_by_page[next_page_number].remove(following)

        ordered: list[list[_RawBlock]] = []
        for page_number in page_numbers:
            ordered.extend(groups_by_page[page_number])
        return ordered

    @staticmethod
    def _assign_columns(blocks: list[_RawBlock]) -> None:
        for page_number in {block.page_number for block in blocks}:
            page_blocks = [
                block for block in blocks if block.page_number == page_number
            ]
            if not page_blocks:
                continue
            candidates = [
                block
                for block in page_blocks
                if block.block_type in {"paragraph", "footnote"}
                and not block.is_table_cell
                and (
                    PDFParser._logical_bbox(block)[2]
                    - PDFParser._logical_bbox(block)[0]
                    < PDFParser._logical_page_size(block)[0] * 0.72
                )
            ]
            clusters: list[list[_RawBlock]] = []
            for block in sorted(
                candidates,
                key=lambda item: PDFParser._logical_bbox(item)[0],
            ):
                logical = PDFParser._logical_bbox(block)
                page_width, _ = PDFParser._logical_page_size(block)
                width = max(logical[2] - logical[0], 1.0)
                best_cluster: list[_RawBlock] | None = None
                best_score = 0.0
                for cluster in clusters:
                    matching_rotation = [
                        item
                        for item in cluster
                        if PDFParser._rotations_match(block, item)
                    ]
                    if not matching_rotation:
                        continue
                    left = statistics.median(
                        PDFParser._logical_bbox(item)[0] for item in matching_rotation
                    )
                    right = statistics.median(
                        PDFParser._logical_bbox(item)[2] for item in matching_rotation
                    )
                    overlap = max(
                        0.0,
                        min(logical[2], right) - max(logical[0], left),
                    )
                    overlap_ratio = overlap / max(1.0, min(width, right - left))
                    alignment = abs(logical[0] - left)
                    score = max(
                        overlap_ratio,
                        1.0 - alignment / max(page_width * 0.08, 1.0),
                    )
                    if score > best_score:
                        best_score = score
                        best_cluster = cluster
                if best_cluster is not None and best_score >= 0.48:
                    best_cluster.append(block)
                else:
                    clusters.append([block])

            clusters.sort(
                key=lambda cluster: statistics.median(
                    PDFParser._logical_bbox(item)[0] for item in cluster
                )
            )
            for column_id, cluster in enumerate(clusters, start=1):
                for block in cluster:
                    block.column_id = column_id
            for block in page_blocks:
                if block.is_table_cell:
                    block.column_id = -1

    @staticmethod
    def _can_merge_blocks(first: _RawBlock, second: _RawBlock) -> bool:
        if (
            first.page_number != second.page_number
            or first.column_id != second.column_id
            or first.block_type != second.block_type
            or first.is_table_cell
            or second.is_table_cell
            or not PDFParser._rotations_match(first, second)
        ):
            return False
        first_bbox = PDFParser._logical_bbox(first)
        second_bbox = PDFParser._logical_bbox(second)
        logical_width, _ = PDFParser._logical_page_size(first)
        base_size = max(first.median_font_size, second.median_font_size, 1.0)
        if abs(first.median_font_size - second.median_font_size) > max(
            1.5, base_size * 0.22
        ):
            return False
        horizontal_alignment = abs(first_bbox[0] - second_bbox[0])
        if horizontal_alignment > max(14.0, logical_width * 0.04):
            return False
        vertical_gap = second_bbox[1] - first_bbox[3]
        if vertical_gap < -base_size * 0.4 or vertical_gap > base_size * 1.35:
            return False

        first_text = normalize_whitespace(first.text)
        second_text = normalize_whitespace(second.text)
        if not first_text or not second_text:
            return False
        if re.match(r"^(?:[-•▪◦‣]|\d+[.)])\s*", second_text):
            return False
        if first_text.endswith("-") and second_text[:1].islower():
            return True
        if second_text[:1].islower() and vertical_gap <= base_size:
            return True
        return (
            not re.search(r"[.!?。！？:：;；][\"'”’)]?$", first_text)
            and vertical_gap <= base_size * 0.65
        )

    @classmethod
    def _merge_cross_column_groups(
        cls,
        groups: list[list[_RawBlock]],
    ) -> list[list[_RawBlock]]:
        columns = sorted(
            {
                group[0].column_id
                for group in groups
                if group[0].column_id > 0 and group[0].block_type == "paragraph"
            }
        )
        for left_column, right_column in zip(columns, columns[1:]):
            left_groups = [
                group for group in groups if group[0].column_id == left_column
            ]
            right_groups = [
                group for group in groups if group[0].column_id == right_column
            ]
            if not left_groups or not right_groups:
                continue
            left = max(
                left_groups,
                key=lambda group: cls._logical_bbox(group[-1])[3],
            )
            right = min(
                right_groups,
                key=lambda group: cls._logical_bbox(group[0])[1],
            )
            if cls._can_continue_across_column(left[-1], right[0]):
                left.extend(right)
                groups.remove(right)
        return groups

    @staticmethod
    def _can_continue_across_column(
        first: _RawBlock,
        second: _RawBlock,
    ) -> bool:
        if (
            first.page_number != second.page_number
            or first.block_type != "paragraph"
            or second.block_type != "paragraph"
            or first.is_table_cell
            or second.is_table_cell
            or first.column_id <= 0
            or second.column_id <= first.column_id
            or not PDFParser._rotations_match(first, second)
        ):
            return False
        first_bbox = PDFParser._logical_bbox(first)
        second_bbox = PDFParser._logical_bbox(second)
        _, first_height = PDFParser._logical_page_size(first)
        _, second_height = PDFParser._logical_page_size(second)
        if first_bbox[3] < first_height * 0.58 or second_bbox[1] > second_height * 0.42:
            return False
        return PDFParser._has_textual_continuity(first, second)

    @staticmethod
    def _can_continue_across_page(
        first: _RawBlock,
        second: _RawBlock,
    ) -> bool:
        if (
            second.page_number != first.page_number + 1
            or first.block_type != "paragraph"
            or second.block_type != "paragraph"
            or first.is_table_cell
            or second.is_table_cell
            or not PDFParser._rotations_match(first, second)
        ):
            return False
        first_bbox = PDFParser._logical_bbox(first)
        second_bbox = PDFParser._logical_bbox(second)
        _, first_height = PDFParser._logical_page_size(first)
        _, second_height = PDFParser._logical_page_size(second)
        if first_bbox[3] < first_height * 0.72 or second_bbox[1] > second_height * 0.28:
            return False
        return PDFParser._has_textual_continuity(first, second)

    @staticmethod
    def _has_textual_continuity(first: _RawBlock, second: _RawBlock) -> bool:
        base_size = max(first.median_font_size, second.median_font_size, 1.0)
        if abs(first.median_font_size - second.median_font_size) > max(
            1.5, base_size * 0.22
        ):
            return False
        first_text = normalize_whitespace(first.text)
        second_text = normalize_whitespace(second.text)
        if not first_text or not second_text:
            return False
        if (
            len(first_text) <= 30
            and len(first_text.split()) <= 5
            and re.fullmatch(r"[A-Z][A-Z0-9 /&+.-]*", first_text)
        ):
            return False
        if min(len(first_text), len(second_text)) < 8 and not (
            first_text.endswith("-") and second_text[:1].islower()
        ):
            return False
        if first_text.endswith("-") and second_text[:1].islower():
            return True
        if second_text[:1].islower():
            return True
        return not re.search(r"[.!?。！？][\"'”’)]?$", first_text)

    @staticmethod
    def _merge_group_text(group: list[_RawBlock]) -> str:
        merged = ""
        for block in group:
            text = normalize_whitespace(block.text)
            if not merged:
                merged = text
            elif merged.endswith("-") and text[:1].islower():
                merged = merged[:-1] + text
            else:
                merged += " " + text
        return merged

    @staticmethod
    def _to_segments(
        blocks: list[_RawBlock] | list[list[_RawBlock]],
        task_id: str,
        settings: TaskSettings,
        stable_row_keys: bool = False,
    ) -> list[Segment]:
        placeholder_service = PlaceholderService()
        segments: list[Segment] = []
        page_orders: Counter[int] = Counter()
        translation_index = 0

        groups: list[list[_RawBlock]]
        if blocks and isinstance(blocks[0], _RawBlock):
            groups = [[block] for block in blocks]  # type: ignore[list-item]
        else:
            groups = blocks  # type: ignore[assignment]

        for group in groups:
            block = group[0]
            source_text = PDFParser._merge_group_text(group)
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
            elif not re.search(r"[A-Za-z]", source_text):
                # Numeric labels, formulas made only from punctuation, and
                # page references should remain untouched. Sending them
                # through document translation only creates false overflow
                # warnings without adding Chinese content.
                should_translate = False
            elif re.fullmatch(r"[A-Z]{1,6}", normalize_whitespace(source_text)):
                # Short all-caps fragments are normally index markers,
                # acronyms, or pieces of curved decorative lettering. Some
                # translators turn them into unrelated Chinese words, so
                # preserve the source glyph.
                should_translate = False

            if should_translate:
                translation_index += 1
                row_key = segment_id if stable_row_keys else f"{translation_index:08d}"
                protected = placeholder_service.protect(
                    source_text,
                    task_id=task_id,
                    segment_id=segment_id,
                    protected_terms=settings.protected_terms,
                )
            else:
                row_key = ""
                protected = placeholder_service.protect(
                    source_text,
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
                    source_text=source_text,
                    protected_text=protected.protected_text,
                    source_text_hash=sha256_text(source_text),
                    bbox=block.bbox,
                    erase_bboxes=block.erase_bboxes or [block.bbox],
                    font_name=block.font_name,
                    font_size=round(block.median_font_size, 2),
                    should_translate=should_translate,
                    placeholders=protected.placeholders,
                    status="pending" if should_translate else "skipped",
                    paragraph_id=segment_id,
                    anchors=[
                        ParagraphAnchor(
                            anchor_id=f"{segment_id}-A{index:03d}",
                            page_number=anchor.page_number,
                            reading_order=anchor.order,
                            bbox=list(anchor.bbox),
                            erase_bboxes=[
                                list(box)
                                for box in (anchor.erase_bboxes or [anchor.bbox])
                            ],
                            source_text=anchor.text,
                            column_id=anchor.column_id,
                            layout_label=anchor.block_type,
                            rotation_degrees=anchor.rotation_degrees,
                            source_kind=anchor.source_kind,
                        )
                        for index, anchor in enumerate(group, start=1)
                    ],
                    layout_label=block.block_type,
                    column_id=block.column_id,
                    rotation_degrees=block.rotation_degrees,
                    source_kind=block.source_kind,
                    continuation=(
                        "cross_page"
                        if len({item.page_number for item in group}) > 1
                        else "cross_column"
                        if len({item.column_id for item in group}) > 1
                        else "visual_fragments"
                        if len(group) > 1
                        else None
                    ),
                )
            )
        return segments
