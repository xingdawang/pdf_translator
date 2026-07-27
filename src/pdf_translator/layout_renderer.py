from __future__ import annotations

import html
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import threading
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import cv2
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from .exceptions import ValidationBlockedError
from .models import Segment, TranslationTask, ValidationReport
from .pdf_renderer import ChineseFontResolver
from .pdf_parser import PDFParser
from .placeholders import PlaceholderService
from .utils import atomic_write_json, normalize_whitespace, safe_stem, utc_now


ProgressCallback = Callable[[int, int], None]
LAYOUT_RENDERER_VERSION = "5.1.2-compact-repair-layers"
PARALLEL_PAGE_THRESHOLD = 4
MAX_REPAIR_TILES_PER_PAGE = 16

# PDF streams are binary-safe. ASCII85 adds roughly 25% transport overhead to
# every raster repair layer without improving compatibility for the files we
# generate.
rl_config.useA85 = 0


@dataclass
class LayoutRenderOutputs:
    layout_pdf: Path
    quality_json: Path
    quality_html: Path
    replaced_segments: int
    overflow_errors: int
    cache_hit: bool = False
    render_workers: int = 1


@dataclass
class _Placement:
    segment_id: str
    page_number: int
    bbox: list[float]
    font_size: float
    source_font_size: float
    rotated: bool
    contour_flow: bool
    fitted: bool
    background_mode: str


@dataclass
class _PreparedTranslation:
    segment: Segment
    text: str
    text_color: tuple[float, float, float]
    background_mode: str
    fragments: list["_LayoutFragment"]


@dataclass
class _LayoutFragment:
    source_text: str
    bbox: list[float]
    font_size: float
    line_bboxes: list[list[float]] = field(default_factory=list)


@dataclass
class _PageRenderRequest:
    selected_index: int
    page_number: int
    width: float
    height: float
    segments: list[Segment]
    ignore_number_warnings: bool


@dataclass
class _RepairImage:
    path: str
    bbox: list[float]


@dataclass
class _PageRenderResult:
    selected_index: int
    page_number: int
    width: float
    height: float
    repair_images: list[_RepairImage]
    dense_cleanup: tuple[list[float], tuple[float, float, float]] | None
    prepared: list[_PreparedTranslation]
    placements_expected: int
    mask_ratio: float
    dense_detail: dict[str, object] | None
    complex_backgrounds: int


class LayoutPreservingRenderer:
    """Replace source text in-place while retaining the original PDF page."""

    def __init__(
        self,
        font_path: str | Path | None = None,
        render_dpi: int = 170,
        minimum_font_size: float = 3.5,
        max_workers: int | None = None,
    ):
        self.font_resolver = ChineseFontResolver(font_path)
        self.render_dpi = max(120, min(300, int(render_dpi)))
        self.minimum_font_size = minimum_font_size
        self.max_workers = max_workers
        self.placeholders = PlaceholderService()

    def _worker_count(self, page_count: int) -> int:
        if page_count < PARALLEL_PAGE_THRESHOLD:
            return 1
        configured = os.getenv("PDF_TRANSLATOR_RENDER_WORKERS", "").strip()
        if self.max_workers is not None:
            available = self.max_workers
        elif configured:
            try:
                available = int(configured)
            except ValueError:
                available = os.cpu_count() or 1
        else:
            available = os.cpu_count() or 1
        return max(1, min(page_count, available))

    def _render_pages(
        self,
        source_path: Path,
        requests: list[_PageRenderRequest],
        temp_dir: Path,
        progress: ProgressCallback | None,
    ) -> tuple[list[_PageRenderResult], int, str]:
        worker_count = self._worker_count(len(requests))
        if worker_count == 1:
            import fitz

            document = fitz.open(source_path)
            results: list[_PageRenderResult] = []
            try:
                for request in requests:
                    results.append(
                        self._prepare_page(document, request, temp_dir)
                    )
                    if progress:
                        progress(len(results), len(requests))
            finally:
                document.close()
            return results, 1, "sequential"

        context = multiprocessing.get_context("spawn")
        try:
            results = []
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_initialize_page_worker,
                initargs=(
                    str(source_path),
                    self.render_dpi,
                    self.minimum_font_size,
                    str(temp_dir),
                ),
            ) as pool:
                futures = {
                    pool.submit(
                        _render_page_worker, request
                    ): request.selected_index
                    for request in requests
                }
                for completed, future in enumerate(
                    as_completed(futures), start=1
                ):
                    results.append(future.result())
                    if progress:
                        progress(completed, len(requests))
            results.sort(key=lambda item: item.selected_index)
            return results, worker_count, "process"
        except (
            NotImplementedError,
            PermissionError,
            BrokenProcessPool,
        ):
            return self._render_pages_with_threads(
                source_path,
                requests,
                temp_dir,
                progress,
                worker_count,
            )

    def _render_pages_with_threads(
        self,
        source_path: Path,
        requests: list[_PageRenderRequest],
        temp_dir: Path,
        progress: ProgressCallback | None,
        worker_count: int,
    ) -> tuple[list[_PageRenderResult], int, str]:
        import fitz

        state = threading.local()
        documents = []
        documents_lock = threading.Lock()
        previous_opencv_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)

        def initialize() -> None:
            state.document = fitz.open(source_path)
            state.renderer = LayoutPreservingRenderer(
                render_dpi=self.render_dpi,
                minimum_font_size=self.minimum_font_size,
                max_workers=1,
            )
            with documents_lock:
                documents.append(state.document)

        def render(request: _PageRenderRequest) -> _PageRenderResult:
            return state.renderer._prepare_page(
                state.document,
                request,
                temp_dir,
            )

        results = []
        try:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="layout-page",
                initializer=initialize,
            ) as pool:
                futures = [pool.submit(render, request) for request in requests]
                for completed, future in enumerate(
                    as_completed(futures), start=1
                ):
                    results.append(future.result())
                    if progress:
                        progress(completed, len(requests))
        finally:
            for document in documents:
                document.close()
            cv2.setNumThreads(previous_opencv_threads)
        results.sort(key=lambda item: item.selected_index)
        return results, worker_count, "thread"

    def _prepare_page(
        self,
        source_document,
        request: _PageRenderRequest,
        temp_dir: Path,
    ) -> _PageRenderResult:
        fitz_page = source_document.load_page(request.page_number - 1)
        pixmap = fitz_page.get_pixmap(dpi=self.render_dpi, alpha=False)
        mode = "RGB" if pixmap.n < 4 else "RGBA"
        page_image = Image.frombytes(
            mode, (pixmap.width, pixmap.height), pixmap.samples
        ).convert("RGB")
        page_array = np.asarray(page_image)
        scale_x = pixmap.width / max(request.width, 1)
        scale_y = pixmap.height / max(request.height, 1)
        candidate_segments = sorted(
            request.segments,
            key=lambda item: item.reading_order,
        )
        fallback_segments = [
            segment
            for segment in candidate_segments
            if segment.status == "source_fallback"
        ]
        replaceable_segments = [
            segment
            for segment in candidate_segments
            if segment.status != "source_fallback"
        ]
        dense_detail = self._dense_page_detail(
            request.page_number,
            candidate_segments,
            request.width,
            request.height,
        )
        dense_cleanup = (
            self._dense_flat_page_cleanup(
                page_array,
                candidate_segments,
                page_width=request.width,
                page_height=request.height,
            )
            if dense_detail and not fallback_segments
            else None
        )
        page_segments = [
            segment
            for segment in replaceable_segments
            if dense_cleanup is not None
            or self._contains_translatable_english(segment.source_text)
        ]
        source_blocks = self._page_source_blocks(fitz_page)
        layout_fragments = {
            segment.segment_id: self._layout_fragments_for_segment(
                segment, source_blocks
            )
            for segment in page_segments
        }
        repair_boxes = {
            segment.segment_id: (
                [
                    box
                    for fragment in layout_fragments[segment.segment_id]
                    for box in (fragment.line_bboxes or [fragment.bbox])
                ]
                if len(layout_fragments[segment.segment_id]) > 1
                else segment.erase_bboxes or [segment.bbox]
            )
            for segment in page_segments
        }
        if dense_cleanup is not None:
            cleanup_box, _ = dense_cleanup
            inpainted_page = page_array
            text_mask = np.zeros(page_array.shape[:2], dtype=np.uint8)
            mask_ratio = (
                (cleanup_box[2] - cleanup_box[0])
                * (cleanup_box[3] - cleanup_box[1])
                / max(request.width * request.height, 1)
            )
        else:
            inpainted_page, text_mask, mask_ratio = self._inpaint_source_text(
                page_array,
                page_segments,
                page_width=request.width,
                page_height=request.height,
                scale_x=scale_x,
                scale_y=scale_y,
                erase_boxes=repair_boxes,
            )

        repair_patches: list[
            tuple[tuple[int, int, int, int], np.ndarray]
        ] = []
        prepared: list[_PreparedTranslation] = []
        complex_backgrounds = 0
        for segment in page_segments:
            restored = self.placeholders.restore(
                segment.translated_text, segment.placeholders
            )
            if not self.placeholders.output_restore_ok(
                restored,
                segment.placeholders,
                request.ignore_number_warnings,
            ):
                raise ValidationBlockedError(
                    f"{segment.segment_id} 的占位符无法恢复。"
                )

            if dense_cleanup is not None:
                background_mode = "dense-flat-cleanup"
            else:
                background_mode, pixel_bbox, patch = (
                    self._build_repaired_patch(
                        page_array,
                        inpainted_page,
                        text_mask,
                        segment,
                        repair_boxes[segment.segment_id],
                        page_width=request.width,
                        page_height=request.height,
                        scale_x=scale_x,
                        scale_y=scale_y,
                    )
                )
                if patch is not None and pixel_bbox is not None:
                    repair_patches.append((pixel_bbox, patch))
            complex_backgrounds += int("complex" in background_mode)
            color = (
                (0.06, 0.06, 0.06)
                if dense_cleanup is not None
                else self._text_color(
                    page_array,
                    repair_boxes[segment.segment_id],
                    scale_x,
                    scale_y,
                )
            )
            replacement_text = (
                segment.source_text
                if not self._contains_translatable_english(segment.source_text)
                else restored.restored_text
            )
            prepared.append(
                _PreparedTranslation(
                    segment=segment,
                    text=replacement_text,
                    text_color=color,
                    background_mode=background_mode,
                    fragments=layout_fragments[segment.segment_id],
                )
            )

        repair_images = (
            self._write_repair_tiles(
                repair_patches,
                temp_dir=temp_dir,
                selected_index=request.selected_index,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            if dense_cleanup is None
            else []
        )

        return _PageRenderResult(
            selected_index=request.selected_index,
            page_number=request.page_number,
            width=request.width,
            height=request.height,
            repair_images=repair_images,
            dense_cleanup=dense_cleanup,
            prepared=prepared,
            placements_expected=len(prepared),
            mask_ratio=mask_ratio,
            dense_detail=dense_detail,
            complex_backgrounds=complex_backgrounds,
        )

    def _generation_fingerprint(
        self,
        task: TranslationTask,
        validation: ValidationReport,
        font_path: Path,
        source_path: Path,
    ) -> str:
        font_stat = font_path.stat()
        source_stat = source_path.stat()
        payload = {
            "renderer": LAYOUT_RENDERER_VERSION,
            "source_hash": task.source_file_hash,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "pages": task.selected_page_numbers,
            "render_dpi": self.render_dpi,
            "minimum_font_size": self.minimum_font_size,
            "font": {
                "path": str(font_path.resolve()),
                "size": font_stat.st_size,
                "mtime_ns": font_stat.st_mtime_ns,
            },
            "ignore_number_warnings": task.settings.ignore_number_warnings,
            "validation": {
                "total_segments": validation.total_segments,
                "translated_segments": validation.translated_segments,
                "blocking_errors": validation.blocking_errors,
                "warnings": validation.warnings,
                "issues": [
                    issue.to_dict() for issue in validation.issues
                ],
                "can_generate": validation.can_generate,
            },
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "page_number": segment.page_number,
                    "reading_order": segment.reading_order,
                    "block_type": segment.block_type,
                    "source_text": segment.source_text,
                    "translated_text": segment.translated_text,
                    "status": segment.status,
                    "bbox": segment.bbox,
                    "erase_bboxes": segment.erase_bboxes,
                    "font_name": segment.font_name,
                    "font_size": segment.font_size,
                    "placeholders": [
                        {
                            "token": item.token,
                            "original": item.original,
                            "kind": item.kind,
                        }
                        for item in segment.placeholders
                    ],
                }
                for segment in task.translatable_segments
            ],
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _cached_outputs(
        self,
        fingerprint: str,
        output_path: Path,
        quality_json: Path,
        quality_html: Path,
        selected_source_pages: list,
    ) -> LayoutRenderOutputs | None:
        if not (
            output_path.is_file()
            and quality_json.is_file()
            and quality_html.is_file()
        ):
            return None
        try:
            report = json.loads(quality_json.read_text(encoding="utf-8"))
            if report.get("generation_fingerprint") != fingerprint:
                return None
            if report.get("output", {}).get("size_bytes") != output_path.stat().st_size:
                return None
            verified = self._verify_output(selected_source_pages, output_path)
            if not verified["valid"]:
                return None
            return LayoutRenderOutputs(
                layout_pdf=output_path,
                quality_json=quality_json,
                quality_html=quality_html,
                replaced_segments=int(report.get("replaced_segments", 0)),
                overflow_errors=int(report.get("overflow_errors", 0)),
                cache_hit=True,
                render_workers=int(report.get("render_workers", 1)),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def generate(
        self,
        task: TranslationTask,
        validation: ValidationReport,
        task_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> LayoutRenderOutputs:
        if not validation.can_generate:
            raise ValidationBlockedError(
                f"存在 {validation.blocking_errors} 个阻断性错误，不能生成原版面中文版。"
            )

        source_path = Path(task.source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"原始 PDF 已移动或删除：{source_path}")

        try:
            import fitz
        except ImportError as exc:
            raise ValidationBlockedError("缺少 PyMuPDF，无法生成原版面中文版。") from exc

        font_name, font_path = self.font_resolver.register()
        output_dir = task_dir / "outputs"
        report_dir = task_dir / "reports"
        temp_dir = task_dir / "tmp" / "layout-replacement"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_stem(task.source_filename)
        output_path = output_dir / f"{stem}_原版面中文版.pdf"
        quality_json = report_dir / "layout_replacement_report.json"
        quality_html = report_dir / "layout_replacement_report.html"
        overlay_path = temp_dir / "overlay.pdf"
        source_reader = PdfReader(str(source_path), strict=False)
        if source_reader.is_encrypted:
            raise ValidationBlockedError("原始 PDF 已加密，无法生成原版面中文版。")
        expected_source_pages = task.source_page_count or task.page_count
        if len(source_reader.pages) != expected_source_pages:
            raise ValidationBlockedError("原始 PDF 页数已经变化，请重新创建任务。")
        page_numbers = task.selected_page_numbers
        if len(page_numbers) != task.page_count or any(
            page_number > len(source_reader.pages) for page_number in page_numbers
        ):
            raise ValidationBlockedError("任务页码范围无效，请重新创建任务。")
        selected_source_pages = [
            source_reader.pages[page_number - 1] for page_number in page_numbers
        ]

        fingerprint = self._generation_fingerprint(
            task,
            validation,
            font_path,
            source_path,
        )
        cached = self._cached_outputs(
            fingerprint,
            output_path,
            quality_json,
            quality_html,
            selected_source_pages,
        )
        if cached is not None:
            if progress:
                progress(task.page_count, task.page_count)
            return cached

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)
        segments_by_page: dict[int, list[Segment]] = {}
        for segment in task.translatable_segments:
            segments_by_page.setdefault(segment.page_number, []).append(segment)

        requests = []
        for selected_index, page_number in enumerate(page_numbers, start=1):
            source_page = source_reader.pages[page_number - 1]
            requests.append(
                _PageRenderRequest(
                    selected_index=selected_index,
                    page_number=page_number,
                    width=float(source_page.mediabox.width),
                    height=float(source_page.mediabox.height),
                    segments=segments_by_page.get(page_number, []),
                    ignore_number_warnings=(
                        task.settings.ignore_number_warnings
                    ),
                )
            )

        page_results, render_workers, render_backend = self._render_pages(
            source_path,
            requests,
            temp_dir,
            progress,
        )
        overlay = canvas.Canvas(str(overlay_path), pageCompression=1)
        placements: list[_Placement] = []
        page_mask_ratios = [item.mask_ratio for item in page_results]
        dense_page_details = [
            item.dense_detail
            for item in page_results
            if item.dense_detail is not None
        ]
        complex_backgrounds = sum(
            item.complex_backgrounds for item in page_results
        )
        try:
            for page_result in page_results:
                overlay.setPageSize((page_result.width, page_result.height))
                if page_result.dense_cleanup is not None:
                    cleanup_box, cleanup_color = page_result.dense_cleanup
                    self._draw_dense_flat_cleanup(
                        overlay,
                        cleanup_box,
                        cleanup_color,
                        page_height=page_result.height,
                    )
                else:
                    for repair_image in page_result.repair_images:
                        x0, y0, x1, y1 = repair_image.bbox
                        overlay.drawImage(
                            repair_image.path,
                            x0,
                            page_result.height - y1,
                            x1 - x0,
                            y1 - y0,
                            mask="auto",
                        )

                for item in page_result.prepared:
                    if len(item.fragments) > 1:
                        placement = self._draw_fragmented_translation(
                            overlay,
                            item.segment,
                            item.text,
                            item.fragments,
                            page_height=page_result.height,
                            font_name=font_name,
                            text_color=item.text_color,
                            background_mode=item.background_mode,
                        )
                    else:
                        placement = self._draw_translation(
                            overlay,
                            item.segment,
                            item.text,
                            page_height=page_result.height,
                            font_name=font_name,
                            text_color=item.text_color,
                            background_mode=item.background_mode,
                        )
                    placements.append(placement)

                overlay.showPage()
        finally:
            overlay.save()

        temporary_output = output_path.with_suffix(".pdf.tmp")
        overlay_reader = PdfReader(str(overlay_path), strict=False)
        writer = PdfWriter()
        for overlay_index, source_page in enumerate(selected_source_pages):
            writer.add_page(source_page)
            output_page = writer.pages[-1]
            output_page.merge_page(overlay_reader.pages[overlay_index])
            merged_content = output_page.get_contents()
            if merged_content is not None:
                # pypdf's convenience method can leave the original
                # /Contents array untouched after merge_page(). Replacing it
                # explicitly guarantees that the combined source and overlay
                # operators are stored as one Flate-compressed stream.
                output_page.replace_contents(
                    merged_content.flate_encode(level=9)
                )
        writer.add_metadata(
            {
                "/Title": f"{task.source_filename} - 原版面中文版",
                "/Author": "Local PDF Translator",
                "/Subject": f"Task {task.task_id}",
            }
        )
        with temporary_output.open("wb") as stream:
            writer.write(stream)
            stream.flush()
        temporary_output.replace(output_path)

        overlap_pairs = self._overlapping_placements(placements)
        overflow = [item for item in placements if not item.fitted]
        verified = self._verify_output(selected_source_pages, output_path)
        payload = {
            "task_id": task.task_id,
            "generated_at": utc_now(),
            "renderer_version": LAYOUT_RENDERER_VERSION,
            "generation_fingerprint": fingerprint,
            "source_filename": task.source_filename,
            "layout_strategy": "in-place-background-repair-and-vector-text",
            "render_dpi": self.render_dpi,
            "render_workers": render_workers,
            "render_backend": render_backend,
            "content_stream_compression": "flate-level-9",
            "repair_image_encoding": "binary-flate-rgba-tiles",
            "repair_image_layers": sum(
                bool(item.repair_images) for item in page_results
            ),
            "repair_image_tiles": sum(
                len(item.repair_images) for item in page_results
            ),
            "source_pages": task.page_count,
            "source_page_numbers": page_numbers,
            "output_pages": verified["page_count"],
            "page_sizes_preserved": verified["page_sizes_preserved"],
            "replaced_segments": len(placements),
            "source_fallback_segments": len(task.source_fallback_segments),
            "source_fallback_segment_ids": [
                segment.segment_id for segment in task.source_fallback_segments
            ],
            "overflow_errors": len(overflow),
            "overflow_segment_ids": [item.segment_id for item in overflow],
            "source_bbox_overlap_warnings": len(overlap_pairs),
            "source_bbox_overlap_pairs": overlap_pairs[:100],
            "complex_background_repairs": complex_backgrounds,
            "masked_page_area_ratio": round(
                float(np.mean(page_mask_ratios)), 6
            )
            if page_mask_ratios
            else 0.0,
            "maximum_page_mask_area_ratio": round(
                max(page_mask_ratios), 6
            )
            if page_mask_ratios
            else 0.0,
            "minimum_rendered_font_size": min(
                (item.font_size for item in placements), default=0
            ),
            "font_reductions": sum(
                item.font_size + 0.1 < item.source_font_size for item in placements
            ),
            "rotated_segments": sum(item.rotated for item in placements),
            "contour_flow_segments": sum(
                item.contour_flow for item in placements
            ),
            "dense_pages_requiring_review": [
                item["page"] for item in dense_page_details
            ],
            "dense_page_details": dense_page_details,
            "dense_flat_page_cleanups": [
                item["page"]
                for item in dense_page_details
                if item["page"] in {
                    placement.page_number
                    for placement in placements
                    if placement.background_mode.startswith(
                        "dense-flat-cleanup"
                    )
                }
            ],
            "font": str(font_path),
            "validation": validation.to_dict(),
            "output": {
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                **verified,
            },
            "status": (
                "needs_review"
                if overflow
                or not verified["page_sizes_preserved"]
                or dense_page_details
                else "passed_with_warnings"
                if overlap_pairs or validation.warnings
                else "passed"
            ),
            "note": (
                "原 PDF 页面作为底图原样保留；仅在 OCR 行坐标处修补背景并写入中文。"
                "多行中文沿原英文逐行轮廓排入，以保留绕图和缩进区域。"
                "扫描页使用英文笔画掩膜和局部图像修复；照片上的文字仍需逐页肉眼抽查。"
                "高密度索引或目录页会被单独标为 needs_review，不能仅凭零溢出判定通过。"
            ),
        }
        atomic_write_json(quality_json, payload)
        self._write_html_report(quality_html, payload)
        shutil.rmtree(temp_dir)
        return LayoutRenderOutputs(
            layout_pdf=output_path,
            quality_json=quality_json,
            quality_html=quality_html,
            replaced_segments=len(placements),
            overflow_errors=len(overflow),
            cache_hit=False,
            render_workers=render_workers,
        )

    def _inpaint_source_text(
        self,
        page_array: np.ndarray,
        segments: list[Segment],
        page_width: float,
        page_height: float,
        scale_x: float,
        scale_y: float,
        erase_boxes: dict[str, list[list[float]]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        mask = np.zeros(page_array.shape[:2], dtype=np.uint8)
        for segment in segments:
            supplied_boxes = (
                erase_boxes.get(segment.segment_id)
                if erase_boxes is not None
                else None
            )
            for supplied in supplied_boxes or segment.erase_bboxes or [segment.bbox]:
                x0, y0, x1, y1 = self._clamp_bbox(
                    supplied, page_width, page_height
                )
                x_padding, y_padding = self._erase_padding(
                    segment, y1 - y0
                )
                core_x0 = max(0, int(math.floor(x0 * scale_x)))
                core_y0 = max(0, int(math.floor(y0 * scale_y)))
                core_x1 = min(
                    page_array.shape[1], int(math.ceil(x1 * scale_x))
                )
                core_y1 = min(
                    page_array.shape[0], int(math.ceil(y1 * scale_y))
                )
                x0 = max(0, x0 - x_padding)
                y0 = max(0, y0 - y_padding)
                x1 = min(page_width, x1 + x_padding)
                y1 = min(page_height, y1 + y_padding)
                px0 = max(0, int(math.floor(x0 * scale_x)))
                py0 = max(0, int(math.floor(y0 * scale_y)))
                px1 = min(page_array.shape[1], int(math.ceil(x1 * scale_x)))
                py1 = min(page_array.shape[0], int(math.ceil(y1 * scale_y)))
                if px1 <= px0 or py1 <= py0:
                    continue

                border = self._border_pixels(page_array, px0, py0, px1, py1)
                background, _ = self._robust_background(border)
                crop = page_array[py0:py1, px0:px1]
                luminance = (
                    crop[:, :, 0].astype(np.float32) * 0.2126
                    + crop[:, :, 1].astype(np.float32) * 0.7152
                    + crop[:, :, 2].astype(np.float32) * 0.0722
                )
                background_luminance = float(
                    background[0] * 0.2126
                    + background[1] * 0.7152
                    + background[2] * 0.0722
                )
                polarity = self._ink_polarity(
                    crop, background_luminance
                )
                if polarity == "dark":
                    local_mask = luminance < background_luminance - 14
                else:
                    local_mask = luminance > background_luminance + 14
                if not np.any(local_mask):
                    local_mask[:, :] = True
                else:
                    # Vision frequently clips the ascenders/descenders of large
                    # serif titles. Keep dark connected components that touch the
                    # original OCR box, allowing the mask to grow into the padded
                    # area without also deleting a nearby rule or image edge.
                    component_count, labels = cv2.connectedComponents(
                        local_mask.astype(np.uint8), connectivity=8
                    )
                    local_core_x0 = max(0, core_x0 - px0)
                    local_core_y0 = max(0, core_y0 - py0)
                    local_core_x1 = min(px1 - px0, core_x1 - px0)
                    local_core_y1 = min(py1 - py0, core_y1 - py0)
                    core_labels = np.unique(
                        labels[
                            local_core_y0:local_core_y1,
                            local_core_x0:local_core_x1,
                        ]
                    )
                    core_labels = core_labels[core_labels != 0]
                    if component_count > 1 and core_labels.size:
                        local_mask = np.isin(labels, core_labels)
                mask[py0:py1, px0:px1] = np.maximum(
                    mask[py0:py1, px0:px1],
                    local_mask.astype(np.uint8) * 255,
                )

        # High-resolution scans often contain thin italic serifs that extend a
        # pixel beyond Vision's line geometry.  A slightly wider high-DPI mask
        # removes those residual strokes without replacing the whole text box
        # or flattening the photograph underneath it.
        kernel_size = max(5, int(round(self.render_dpi / 38)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
        inpaint_radius = max(2.0, self.render_dpi / 70.0)
        repaired = cv2.inpaint(
            page_array,
            mask,
            inpaint_radius,
            cv2.INPAINT_TELEA,
        )
        return repaired, mask, float(np.mean(mask > 0))

    def _build_repaired_patch(
        self,
        page_array: np.ndarray,
        inpainted_page: np.ndarray,
        text_mask: np.ndarray,
        segment: Segment,
        line_bboxes: list[list[float]],
        page_width: float,
        page_height: float,
        scale_x: float,
        scale_y: float,
    ) -> tuple[str, tuple[int, int, int, int] | None, np.ndarray | None]:
        all_x0 = min(float(box[0]) for box in line_bboxes)
        all_y0 = min(float(box[1]) for box in line_bboxes)
        all_x1 = max(float(box[2]) for box in line_bboxes)
        all_y1 = max(float(box[3]) for box in line_bboxes)
        x_padding, y_padding = self._erase_padding(
            segment, all_y1 - all_y0
        )
        x0, y0, x1, y1 = self._clamp_bbox(
            [
                all_x0 - x_padding,
                all_y0 - y_padding,
                all_x1 + x_padding,
                all_y1 + y_padding,
            ],
            page_width,
            page_height,
        )
        px0 = max(0, int(math.floor(x0 * scale_x)))
        py0 = max(0, int(math.floor(y0 * scale_y)))
        px1 = min(page_array.shape[1], int(math.ceil(x1 * scale_x)))
        py1 = min(page_array.shape[0], int(math.ceil(y1 * scale_y)))
        if px1 <= px0 or py1 <= py0:
            return "inpaint-empty", None, None

        border = self._border_pixels(page_array, px0, py0, px1, py1)
        _, spread = self._robust_background(border)
        original_patch = page_array[py0:py1, px0:px1]
        patch = inpainted_page[py0:py1, px0:px1]
        alpha = text_mask[py0:py1, px0:px1]
        if not np.any(alpha):
            return "inpaint-empty", None, None
        dominant_background, dominant_ratio = self._dominant_background(
            original_patch
        )
        if dominant_ratio >= 0.32:
            patch = original_patch.copy()
            patch[alpha > 0] = dominant_background
            spread = 0.0
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.55)
        patch = np.dstack((patch, alpha)).astype(np.uint8)
        if dominant_ratio >= 0.32:
            mode = "flat-background"
        else:
            mode = "inpaint-complex" if spread > 20.0 else "inpaint"
        return mode, (px0, py0, px1, py1), patch

    def _write_repair_tiles(
        self,
        patches: list[tuple[tuple[int, int, int, int], np.ndarray]],
        temp_dir: Path,
        selected_index: int,
        scale_x: float,
        scale_y: float,
    ) -> list[_RepairImage]:
        if not patches:
            return []

        gap = max(6, int(round(self.render_dpi * 0.08)))
        groups = self._cluster_repair_patches(
            patches,
            gap=gap,
            maximum_groups=MAX_REPAIR_TILES_PER_PAGE,
        )
        repair_images: list[_RepairImage] = []
        for tile_index, (tile_bbox, tile_patches) in enumerate(
            groups, start=1
        ):
            left, top, right, bottom = tile_bbox
            tile = Image.new(
                "RGBA",
                (right - left, bottom - top),
                (0, 0, 0, 0),
            )
            for patch_bbox, patch in tile_patches:
                tile.alpha_composite(
                    Image.fromarray(patch, mode="RGBA"),
                    dest=(patch_bbox[0] - left, patch_bbox[1] - top),
                )

            alpha_bbox = tile.getbbox()
            if alpha_bbox is None:
                continue
            crop_left, crop_top, crop_right, crop_bottom = alpha_bbox
            tile = tile.crop(alpha_bbox)
            absolute_bbox = (
                left + crop_left,
                top + crop_top,
                left + crop_right,
                top + crop_bottom,
            )
            repair_path = temp_dir / (
                f"repair-page-{selected_index:06d}"
                f"-tile-{tile_index:03d}.png"
            )
            tile.save(
                repair_path,
                format="PNG",
                optimize=True,
                compress_level=9,
            )
            repair_images.append(
                _RepairImage(
                    path=str(repair_path),
                    bbox=[
                        absolute_bbox[0] / scale_x,
                        absolute_bbox[1] / scale_y,
                        absolute_bbox[2] / scale_x,
                        absolute_bbox[3] / scale_y,
                    ],
                )
            )
        return repair_images

    @classmethod
    def _cluster_repair_patches(
        cls,
        patches: list[tuple[tuple[int, int, int, int], np.ndarray]],
        gap: int,
        maximum_groups: int,
    ) -> list[
        tuple[
            tuple[int, int, int, int],
            list[tuple[tuple[int, int, int, int], np.ndarray]],
        ]
    ]:
        groups = [(bbox, [(bbox, patch)]) for bbox, patch in patches]
        changed = True
        while changed:
            changed = False
            merged = []
            while groups:
                bbox, items = groups.pop(0)
                match_indexes = [
                    index
                    for index, (candidate_bbox, _) in enumerate(groups)
                    if cls._repair_boxes_are_close(
                        bbox, candidate_bbox, gap
                    )
                ]
                if match_indexes:
                    for index in reversed(match_indexes):
                        candidate_bbox, candidate_items = groups.pop(index)
                        bbox = cls._union_pixel_boxes(bbox, candidate_bbox)
                        items.extend(candidate_items)
                    groups.insert(0, (bbox, items))
                    changed = True
                else:
                    merged.append((bbox, items))
            groups = merged

        while len(groups) > maximum_groups:
            best_pair: tuple[int, int] | None = None
            best_cost: int | None = None
            for left_index in range(len(groups) - 1):
                left_bbox = groups[left_index][0]
                for right_index in range(left_index + 1, len(groups)):
                    right_bbox = groups[right_index][0]
                    union = cls._union_pixel_boxes(left_bbox, right_bbox)
                    cost = (
                        cls._pixel_box_area(union)
                        - cls._pixel_box_area(left_bbox)
                        - cls._pixel_box_area(right_bbox)
                    )
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_pair = (left_index, right_index)
            if best_pair is None:
                break
            left_index, right_index = best_pair
            left_bbox, left_items = groups[left_index]
            right_bbox, right_items = groups[right_index]
            groups[left_index] = (
                cls._union_pixel_boxes(left_bbox, right_bbox),
                left_items + right_items,
            )
            groups.pop(right_index)

        return sorted(groups, key=lambda item: (item[0][1], item[0][0]))

    @staticmethod
    def _repair_boxes_are_close(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
        gap: int,
    ) -> bool:
        return not (
            left[2] + gap < right[0]
            or right[2] + gap < left[0]
            or left[3] + gap < right[1]
            or right[3] + gap < left[1]
        )

    @staticmethod
    def _union_pixel_boxes(
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        return (
            min(left[0], right[0]),
            min(left[1], right[1]),
            max(left[2], right[2]),
            max(left[3], right[3]),
        )

    @staticmethod
    def _pixel_box_area(box: tuple[int, int, int, int]) -> int:
        return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

    @staticmethod
    def _erase_padding(segment: Segment, height: float) -> tuple[float, float]:
        x_padding = max(0.9, min(3.0, height * 0.20))
        if segment.block_type == "title":
            return x_padding, max(3.0, min(14.0, height * 0.58))
        if segment.block_type == "heading":
            return x_padding, max(1.5, min(5.0, height * 0.34))
        return x_padding, max(0.9, min(3.0, height * 0.20))

    @staticmethod
    def _page_source_blocks(fitz_page) -> list[dict[str, object]]:
        import fitz

        page_dict = fitz_page.get_text(
            "dict", flags=fitz.TEXTFLAGS_TEXT, sort=True
        )
        blocks: list[dict[str, object]] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            block_text, _ = PDFParser._block_text_and_spans(block)
            fragments: list[_LayoutFragment] = []
            for item in PDFParser.text_block_fragments(block):
                spans = list(item["spans"])
                sizes = [
                    float(span.get("size", 0.0))
                    for span in spans
                    if span.get("size")
                ]
                fragments.append(
                    _LayoutFragment(
                        source_text=normalize_whitespace(str(item["text"])),
                        bbox=[float(value) for value in item["bbox"]],
                        font_size=float(np.median(sizes)) if sizes else 0.0,
                        line_bboxes=[
                            [float(value) for value in box]
                            for box in item["line_bboxes"]
                        ],
                    )
                )
            blocks.append(
                {
                    "text": normalize_whitespace(block_text),
                    "bbox": [
                        float(value)
                        for value in block.get("bbox", (0, 0, 0, 0))
                    ],
                    "fragments": fragments,
                }
            )
        return blocks

    @classmethod
    def _layout_fragments_for_segment(
        cls,
        segment: Segment,
        source_blocks: list[dict[str, object]],
    ) -> list[_LayoutFragment]:
        source_text = normalize_whitespace(segment.source_text)
        fragment_matches: list[tuple[float, _LayoutFragment]] = []
        for block in source_blocks:
            for fragment in block["fragments"]:
                if fragment.source_text == source_text:
                    fragment_matches.append(
                        (
                            cls._bbox_distance(segment.bbox, fragment.bbox),
                            fragment,
                        )
                    )
        if fragment_matches:
            distance, fragment = min(fragment_matches, key=lambda item: item[0])
            if distance <= 8.0:
                return [fragment]

        block_matches: list[tuple[float, dict[str, object]]] = []
        for block in source_blocks:
            if block["text"] == source_text:
                block_matches.append(
                    (
                        cls._bbox_distance(segment.bbox, block["bbox"]),
                        block,
                    )
                )
        if block_matches:
            distance, block = min(block_matches, key=lambda item: item[0])
            if distance <= 12.0:
                return list(block["fragments"])

        geometry_matches = sorted(
            (
                (cls._bbox_distance(segment.bbox, block["bbox"]), block)
                for block in source_blocks
            ),
            key=lambda item: item[0],
        )
        if geometry_matches and geometry_matches[0][0] <= 4.0:
            return list(geometry_matches[0][1]["fragments"])
        return []

    @staticmethod
    def _bbox_distance(first: list[float], second: list[float]) -> float:
        return sum(
            abs(float(left) - float(right))
            for left, right in zip(first, second)
        )

    def _draw_fragmented_translation(
        self,
        pdf_canvas: canvas.Canvas,
        segment: Segment,
        text: str,
        fragments: list[_LayoutFragment],
        page_height: float,
        font_name: str,
        text_color: tuple[float, float, float],
        background_mode: str,
    ) -> _Placement:
        chunks = self._partition_translation(text, fragments)
        if len(chunks) != len(fragments):
            return self._draw_translation(
                pdf_canvas,
                segment,
                text,
                page_height,
                font_name,
                text_color,
                background_mode,
            )

        fragment_placements: list[_Placement] = []
        for index, (fragment, chunk) in enumerate(zip(fragments, chunks)):
            draw_bbox = list(fragment.bbox)
            if index + 1 < len(fragments):
                draw_bbox[2] = max(
                    draw_bbox[2],
                    fragments[index + 1].bbox[0] - 1.5,
                )
            else:
                preceding_gap = (
                    fragment.bbox[0] - fragments[0].bbox[2]
                    if len(fragments) == 2
                    else float("inf")
                )
                is_numbered_label_pair = (
                    len(fragments) == 2
                    and fragments[0].bbox[2] - fragments[0].bbox[0] < 60
                    and preceding_gap < 30
                )
                draw_bbox[2] += (
                    max(12.0, fragment.font_size * 1.4)
                    if is_numbered_label_pair
                    else 3.0
                )
            fragment_segment = replace(
                segment,
                source_text=fragment.source_text,
                bbox=draw_bbox,
                erase_bboxes=[
                    list(box)
                    for box in (
                        fragment.line_bboxes
                        if len(fragment.line_bboxes) > 1
                        else [draw_bbox]
                    )
                ],
                font_size=fragment.font_size or segment.font_size,
            )
            fragment_placements.append(
                self._draw_translation(
                    pdf_canvas,
                    fragment_segment,
                    chunk,
                    page_height,
                    font_name,
                    text_color,
                    background_mode,
                )
            )
        return _Placement(
            segment_id=segment.segment_id,
            page_number=segment.page_number,
            bbox=list(segment.bbox),
            font_size=round(
                min(item.font_size for item in fragment_placements), 2
            ),
            source_font_size=round(
                min(item.source_font_size for item in fragment_placements), 2
            ),
            rotated=any(item.rotated for item in fragment_placements),
            contour_flow=any(
                item.contour_flow for item in fragment_placements
            ),
            fitted=all(item.fitted for item in fragment_placements),
            background_mode=background_mode + "-fragmented",
        )

    @classmethod
    def _partition_translation(
        cls, text: str, fragments: list[_LayoutFragment]
    ) -> list[str]:
        tokens = text.split()
        fragment_count = len(fragments)
        if fragment_count < 2 or len(tokens) < fragment_count or len(tokens) > 80:
            return []

        source_lengths = [
            max(0.5, cls._visual_text_length(fragment.source_text))
            for fragment in fragments
        ]
        source_total = sum(source_lengths)
        target_lengths = [cls._visual_text_length(token) for token in tokens]
        target_total = max(sum(target_lengths), 0.5)

        anchors_by_fragment = [
            cls._layout_anchors(fragment.source_text) for fragment in fragments
        ]
        anchor_owners: dict[str, set[int]] = {}
        for index, anchors in enumerate(anchors_by_fragment):
            for anchor in anchors:
                anchor_owners.setdefault(anchor, set()).add(index)
        unique_anchors = {
            anchor: next(iter(owners))
            for anchor, owners in anchor_owners.items()
            if len(owners) == 1 and anchor in text.casefold()
        }

        token_prefix = [0.0]
        for length in target_lengths:
            token_prefix.append(token_prefix[-1] + length)

        def chunk_cost(fragment_index: int, start: int, end: int) -> float:
            chunk = " ".join(tokens[start:end]).casefold()
            target_ratio = (
                token_prefix[end] - token_prefix[start]
            ) / target_total
            source_ratio = source_lengths[fragment_index] / source_total
            cost = abs(target_ratio - source_ratio) * 12.0
            for anchor, owner in unique_anchors.items():
                if anchor not in chunk:
                    continue
                cost += -2.0 if owner == fragment_index else 8.0
            for anchor in anchors_by_fragment[fragment_index]:
                if anchor in unique_anchors and anchor not in chunk:
                    cost += 3.0
                if (
                    anchor in unique_anchors
                    and fragments[fragment_index]
                    .source_text.casefold()
                    .startswith(anchor)
                    and not chunk.startswith(anchor)
                ):
                    cost += 12.0
            return cost

        infinity = float("inf")
        costs = [
            [infinity] * (len(tokens) + 1)
            for _ in range(fragment_count + 1)
        ]
        previous = [
            [-1] * (len(tokens) + 1)
            for _ in range(fragment_count + 1)
        ]
        costs[0][0] = 0.0
        for fragment_index in range(fragment_count):
            minimum_end = fragment_index + 1
            maximum_end = len(tokens) - (
                fragment_count - fragment_index - 1
            )
            for start in range(fragment_index, len(tokens)):
                if not math.isfinite(costs[fragment_index][start]):
                    continue
                for end in range(max(start + 1, minimum_end), maximum_end + 1):
                    candidate = (
                        costs[fragment_index][start]
                        + chunk_cost(fragment_index, start, end)
                    )
                    if candidate < costs[fragment_index + 1][end]:
                        costs[fragment_index + 1][end] = candidate
                        previous[fragment_index + 1][end] = start

        if not math.isfinite(costs[fragment_count][len(tokens)]):
            return []
        boundaries = [len(tokens)]
        cursor = len(tokens)
        for fragment_index in range(fragment_count, 0, -1):
            cursor = previous[fragment_index][cursor]
            if cursor < 0:
                return []
            boundaries.append(cursor)
        boundaries.reverse()
        return [
            " ".join(tokens[boundaries[index] : boundaries[index + 1]])
            for index in range(fragment_count)
        ]

    @staticmethod
    def _visual_text_length(text: str) -> float:
        total = 0.0
        for character in text:
            if "\u3400" <= character <= "\u9fff":
                total += 1.0
            elif character.isspace():
                total += 0.25
            elif character.isalnum():
                total += 0.55
            else:
                total += 0.45
        return total

    @staticmethod
    def _layout_anchors(text: str) -> set[str]:
        candidates = re.findall(
            r"(?:https?://|www\.)\S+|[A-Za-z]*\d[\w./:%+-]*|[A-Z]{2,}",
            text,
        )
        return {
            candidate.strip(".,;:()[]{}").casefold()
            for candidate in candidates
            if candidate.strip(".,;:()[]{}")
        }

    def _draw_translation(
        self,
        pdf_canvas: canvas.Canvas,
        segment: Segment,
        text: str,
        page_height: float,
        font_name: str,
        text_color: tuple[float, float, float],
        background_mode: str,
    ) -> _Placement:
        x0, y0, x1, y1 = [float(value) for value in segment.bbox]
        width = max(x1 - x0, 1)
        height = max(y1 - y0, 1)
        # Vision occasionally returns a two- or three-point-high box for a
        # readable small caption. Expanding only the drawing rectangle avoids
        # clipping the replacement in half without widening the erased image
        # patch into adjacent artwork.
        minimum_draw_height = self.minimum_font_size * 1.20
        if len(segment.erase_bboxes or [segment.bbox]) == 1 and (
            height < minimum_draw_height
        ):
            center_y = (y0 + y1) * 0.5
            y0 = max(0.0, center_y - minimum_draw_height * 0.5)
            y1 = min(page_height, y0 + minimum_draw_height)
            y0 = max(0.0, y1 - minimum_draw_height)
            height = max(y1 - y0, 1)
        # A tall union bbox can still be an ordinary narrow column made from
        # several horizontal lines.  Only rotate when the OCR geometry itself
        # contains one genuinely vertical line.
        is_dense_cleanup = background_mode.startswith("dense-flat-cleanup")
        is_index_marker = bool(
            re.fullmatch(r"[A-Z]", normalize_whitespace(segment.source_text))
        )
        effective_block_type = (
            "paragraph"
            if is_dense_cleanup and not is_index_marker
            else segment.block_type
        )
        rotated = self._is_rotated_segment(segment, text)
        contour_flow = (
            not rotated
            and effective_block_type not in {"title", "heading"}
            and len(segment.erase_bboxes) > 1
        )
        source_size = self._source_font_size(segment)
        if is_dense_cleanup and not is_index_marker:
            # Index OCR occasionally labels a malformed multi-line entry as a
            # title, yielding a giant replacement in an otherwise tiny column.
            # Dense pages use one restrained body scale while standalone
            # alphabetic dividers retain their larger source size.
            source_size = min(
                source_size,
                max(7.5, self.minimum_font_size * 2.15),
            )
        if contour_flow:
            font_size, fitted = self._draw_contour_text(
                pdf_canvas,
                segment,
                text,
                page_height,
                font_name,
                source_size,
                text_color,
            )
            return _Placement(
                segment_id=segment.segment_id,
                page_number=segment.page_number,
                bbox=list(segment.bbox),
                font_size=round(font_size, 2),
                source_font_size=round(source_size, 2),
                rotated=False,
                contour_flow=True,
                fitted=fitted,
                background_mode=background_mode,
            )

        available_width = height if rotated else width
        available_height = width if rotated else height
        font_size, paragraph, wrapped_height, fitted = self._fit_paragraph(
            text,
            available_width,
            available_height,
            font_name,
            source_size,
            text_color,
            effective_block_type,
        )

        pdf_canvas.saveState()
        if not fitted:
            path = pdf_canvas.beginPath()
            path.rect(x0, page_height - y1, width, height)
            pdf_canvas.clipPath(path, stroke=0, fill=0)
        if rotated:
            pdf_canvas.translate(x1, page_height - y1)
            pdf_canvas.rotate(90)
            paragraph.drawOn(
                pdf_canvas, 0, max(0, available_height - wrapped_height)
            )
        else:
            paragraph.drawOn(
                pdf_canvas,
                x0,
                page_height - y0 - wrapped_height,
            )
        pdf_canvas.restoreState()
        return _Placement(
            segment_id=segment.segment_id,
            page_number=segment.page_number,
            bbox=list(segment.bbox),
            font_size=round(font_size, 2),
            source_font_size=round(source_size, 2),
            rotated=rotated,
            contour_flow=False,
            fitted=fitted,
            background_mode=background_mode,
        )

    @staticmethod
    def _is_rotated_segment(segment: Segment, text: str) -> bool:
        orientation_boxes = segment.erase_bboxes or [segment.bbox]
        first_box = orientation_boxes[0]
        first_width = max(float(first_box[2]) - float(first_box[0]), 1.0)
        first_height = max(float(first_box[3]) - float(first_box[1]), 1.0)
        return (
            len(orientation_boxes) == 1
            and first_height > first_width * 1.45
            and len(text) <= 80
        )

    @staticmethod
    def _dense_page_detail(
        page_number: int,
        segments: list[Segment],
        width: float,
        height: float,
    ) -> dict[str, object] | None:
        source_characters = sum(len(segment.source_text) for segment in segments)
        source_lines = sum(
            len(segment.erase_bboxes or [segment.bbox])
            for segment in segments
        )
        character_density = source_characters / max(width * height, 1)
        if source_lines < 240 and character_density < 0.012:
            return None
        return {
            "page": page_number,
            "source_characters": source_characters,
            "ocr_lines": source_lines,
            "character_density": round(character_density, 6),
            "reason": (
                "高密度索引或目录页需要逐页人工检查；"
                "局部原位替换可能保留 OCR 漏掉的小字。"
            ),
        }

    @staticmethod
    def _contains_translatable_english(text: str) -> bool:
        normalized = normalize_whitespace(text)
        return bool(
            re.search(r"[A-Za-z]", normalized)
            and not re.fullmatch(r"[A-Z]", normalized)
        )

    @classmethod
    def _dense_flat_page_cleanup(
        cls,
        page_array: np.ndarray,
        segments: list[Segment],
        page_width: float,
        page_height: float,
    ) -> tuple[list[float], tuple[float, float, float]] | None:
        """Return a safe clean text slab for light, image-free index pages."""
        if page_array.size == 0 or not segments:
            return None
        channel_min = page_array.min(axis=2)
        channel_spread = page_array.max(axis=2) - channel_min
        neutral_light = (channel_min >= 235) & (channel_spread <= 18)
        if float(np.mean(neutral_light)) < 0.80:
            return None

        boxes = []
        for segment in segments:
            box = [float(value) for value in segment.bbox]
            if cls._is_rotated_segment(segment, segment.source_text):
                continue
            if (box[0] + box[2]) * 0.5 >= page_width * 0.94:
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
        if not boxes:
            return None

        padding_x = max(3.0, page_width * 0.007)
        padding_y = max(3.0, page_height * 0.005)
        cleanup_box = cls._clamp_bbox(
            [
                min(box[0] for box in boxes) - padding_x,
                min(box[1] for box in boxes) - padding_y,
                max(box[2] for box in boxes) + padding_x,
                max(box[3] for box in boxes) + padding_y,
            ],
            page_width,
            page_height,
        )
        background_pixels = page_array[neutral_light]
        background = np.median(background_pixels, axis=0) / 255.0
        return cleanup_box, tuple(float(value) for value in background)

    @staticmethod
    def _draw_dense_flat_cleanup(
        pdf_canvas: canvas.Canvas,
        cleanup_box: list[float],
        cleanup_color: tuple[float, float, float],
        page_height: float,
    ) -> None:
        x0, y0, x1, y1 = cleanup_box
        pdf_canvas.saveState()
        pdf_canvas.setFillColor(colors.Color(*cleanup_color))
        pdf_canvas.rect(
            x0,
            page_height - y1,
            x1 - x0,
            y1 - y0,
            stroke=0,
            fill=1,
        )
        pdf_canvas.restoreState()

    def _draw_contour_text(
        self,
        pdf_canvas: canvas.Canvas,
        segment: Segment,
        text: str,
        page_height: float,
        font_name: str,
        source_size: float,
        text_color: tuple[float, float, float],
    ) -> tuple[float, bool]:
        slots = sorted(
            (
                [float(value) for value in box]
                for box in segment.erase_bboxes
                if len(box) == 4
                and float(box[2]) > float(box[0])
                and float(box[3]) > float(box[1])
            ),
            key=lambda box: (box[1], box[0]),
        )
        font_size, lines, fitted = self._fit_text_to_slots(
            text,
            slots,
            font_name,
            source_size,
        )

        pdf_canvas.saveState()
        pdf_canvas.setFillColor(colors.Color(*text_color))
        pdf_canvas.setFont(font_name, font_size)
        for box, line in zip(slots, lines):
            if not line:
                continue
            x0, y0, _, y1 = box
            line_height = y1 - y0
            baseline = (
                page_height
                - y1
                + max(0.0, (line_height - font_size) * 0.5)
                + font_size * 0.10
            )
            pdf_canvas.drawString(x0, baseline, line)
        pdf_canvas.restoreState()
        return font_size, fitted

    def _fit_text_to_slots(
        self,
        text: str,
        slots: list[list[float]],
        font_name: str,
        source_size: float,
    ) -> tuple[float, list[str], bool]:
        if not slots:
            return self.minimum_font_size, [], False
        size = max(
            self.minimum_font_size,
            min(34.0, source_size * 1.02),
        )
        last_lines: list[str] = []
        while size >= self.minimum_font_size - 0.01:
            lines, consumed = self._pack_text_into_slots(
                text,
                slots,
                font_name,
                size,
            )
            last_lines = lines
            if consumed >= len(text.rstrip()):
                return size, lines, True
            size -= 0.25
        return self.minimum_font_size, last_lines, False

    @staticmethod
    def _pack_text_into_slots(
        text: str,
        slots: list[list[float]],
        font_name: str,
        font_size: float,
    ) -> tuple[list[str], int]:
        content = text.rstrip()
        cursor = 0
        lines: list[str] = []
        for x0, _, x1, _ in slots:
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if cursor >= len(content):
                lines.append("")
                continue

            start = cursor
            end = cursor
            available_width = max(0.5, x1 - x0)
            while end < len(content):
                character = content[end]
                if character == "\n":
                    break
                candidate = content[start : end + 1]
                if (
                    pdfmetrics.stringWidth(
                        candidate, font_name, font_size
                    )
                    > available_width + 0.1
                ):
                    break
                end += 1
            if end == start:
                end = min(len(content), start + 1)

            lines.append(content[start:end].rstrip())
            cursor = end
            if cursor < len(content) and content[cursor] == "\n":
                cursor += 1
        return lines, cursor

    def _fit_paragraph(
        self,
        text: str,
        width: float,
        height: float,
        font_name: str,
        source_size: float,
        text_color: tuple[float, float, float],
        block_type: str,
    ) -> tuple[float, Paragraph, float, bool]:
        start = max(self.minimum_font_size, min(34.0, source_size * 1.08))
        if block_type in {"title", "heading"}:
            start = min(36.0, max(start, source_size))
        size = start
        escaped = html.escape(text).replace("\n", "<br/>")
        last: tuple[Paragraph, float] | None = None
        while size >= self.minimum_font_size - 0.01:
            style = ParagraphStyle(
                f"Replacement-{size:.2f}",
                fontName=font_name,
                fontSize=size,
                leading=size * 1.18,
                textColor=colors.Color(*text_color),
                alignment=TA_LEFT,
                wordWrap="CJK",
                splitLongWords=True,
                allowWidows=0,
                allowOrphans=0,
            )
            paragraph = Paragraph(escaped, style)
            _, wrapped_height = paragraph.wrap(width, 100000)
            last = (paragraph, wrapped_height)
            if wrapped_height <= height + 0.35:
                return size, paragraph, wrapped_height, True
            size -= 0.25

        assert last is not None
        paragraph, wrapped_height = last
        return self.minimum_font_size, paragraph, wrapped_height, False

    @staticmethod
    def _source_font_size(segment: Segment) -> float:
        heights = [
            max(0.1, float(box[3]) - float(box[1]))
            for box in segment.erase_bboxes
            if len(box) == 4
        ]
        geometry_size = float(np.median(heights)) * 0.92 if heights else 0.0
        return max(5.0, segment.font_size or 0.0, geometry_size)

    @classmethod
    def _text_color(
        cls,
        page_array: np.ndarray,
        line_bboxes: list[list[float]],
        scale_x: float,
        scale_y: float,
    ) -> tuple[float, float, float]:
        dark_score = 0.0
        light_score = 0.0
        for box in line_bboxes[:20]:
            x0, y0, x1, y1 = box
            px0 = max(0, int(x0 * scale_x))
            py0 = max(0, int(y0 * scale_y))
            px1 = min(page_array.shape[1], int(math.ceil(x1 * scale_x)))
            py1 = min(page_array.shape[0], int(math.ceil(y1 * scale_y)))
            if px1 > px0 and py1 > py0:
                border = cls._border_pixels(
                    page_array, px0, py0, px1, py1
                )
                background, _ = cls._robust_background(border)
                background_luminance = float(
                    background[0] * 0.2126
                    + background[1] * 0.7152
                    + background[2] * 0.0722
                )
                crop = page_array[py0:py1, px0:px1]
                dark, light = cls._ink_scores(
                    crop, background_luminance
                )
                dark_score += dark
                light_score += light
        return (
            (0.96, 0.97, 0.96)
            if light_score > dark_score * 1.08
            else (0.07, 0.09, 0.09)
        )

    @classmethod
    def _ink_polarity(
        cls, crop: np.ndarray, background_luminance: float
    ) -> str:
        dark_score, light_score = cls._ink_scores(
            crop, background_luminance
        )
        return "light" if light_score > dark_score * 1.08 else "dark"

    @staticmethod
    def _ink_scores(
        crop: np.ndarray, background_luminance: float
    ) -> tuple[float, float]:
        if not crop.size:
            return 0.0, 0.0
        luminance = (
            crop[:, :, 0].astype(np.float32) * 0.2126
            + crop[:, :, 1].astype(np.float32) * 0.7152
            + crop[:, :, 2].astype(np.float32) * 0.0722
        )
        dark_delta = np.maximum(background_luminance - luminance - 14.0, 0.0)
        light_delta = np.maximum(luminance - background_luminance - 14.0, 0.0)
        return float(np.sum(dark_delta)), float(np.sum(light_delta))

    @staticmethod
    def _border_pixels(
        page_array: np.ndarray, x0: int, y0: int, x1: int, y1: int
    ) -> np.ndarray:
        margin = max(2, min(6, (y1 - y0) // 2 or 2))
        ox0 = max(0, x0 - margin)
        oy0 = max(0, y0 - margin)
        ox1 = min(page_array.shape[1], x1 + margin)
        oy1 = min(page_array.shape[0], y1 + margin)
        pieces = [
            page_array[oy0:y0, ox0:ox1].reshape(-1, 3),
            page_array[y1:oy1, ox0:ox1].reshape(-1, 3),
            page_array[y0:y1, ox0:x0].reshape(-1, 3),
            page_array[y0:y1, x1:ox1].reshape(-1, 3),
        ]
        available = [piece for piece in pieces if piece.size]
        return np.concatenate(available, axis=0) if available else np.empty((0, 3))

    @staticmethod
    def _robust_background(border: np.ndarray) -> tuple[np.ndarray, float]:
        if not border.size:
            return np.array([255.0, 255.0, 255.0]), 0.0
        pixels = border.astype(np.float32)
        luminance = (
            pixels[:, 0] * 0.2126
            + pixels[:, 1] * 0.7152
            + pixels[:, 2] * 0.0722
        )
        median_luminance = float(np.median(luminance))
        if median_luminance >= 120:
            cutoff = np.percentile(luminance, 45)
            candidates = pixels[luminance >= cutoff]
        else:
            cutoff = np.percentile(luminance, 55)
            candidates = pixels[luminance <= cutoff]
        if not candidates.size:
            candidates = pixels
        background = np.median(candidates, axis=0)
        spread = float(np.mean(np.std(candidates, axis=0)))
        return background, spread

    @staticmethod
    def _dominant_background(crop: np.ndarray) -> tuple[np.ndarray, float]:
        if not crop.size:
            return np.array([255, 255, 255], dtype=np.uint8), 0.0
        pixels = crop.reshape(-1, 3).astype(np.int16)
        quantized = (pixels // 12).astype(np.int16)
        bin_count = 22
        encoded = (
            quantized[:, 0] * bin_count * bin_count
            + quantized[:, 1] * bin_count
            + quantized[:, 2]
        )
        counts = np.bincount(
            encoded,
            minlength=bin_count * bin_count * bin_count,
        )
        dominant_code = int(np.argmax(counts))
        dominant = np.array(
            [
                dominant_code // (bin_count * bin_count),
                (dominant_code // bin_count) % bin_count,
                dominant_code % bin_count,
            ],
            dtype=np.int16,
        )
        distance = np.max(np.abs(quantized - dominant), axis=1)
        candidates = pixels[distance <= 1]
        if not candidates.size:
            candidates = pixels
        background = np.median(candidates, axis=0).astype(np.uint8)
        ratio = float(counts[dominant_code] / max(len(pixels), 1))
        return background, ratio

    @staticmethod
    def _clamp_bbox(
        bbox: list[float], page_width: float, page_height: float
    ) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        x0 = max(0.0, min(page_width, x0))
        x1 = max(x0, min(page_width, x1))
        y0 = max(0.0, min(page_height, y0))
        y1 = max(y0, min(page_height, y1))
        return x0, y0, x1, y1

    @staticmethod
    def _overlapping_placements(
        placements: list[_Placement],
    ) -> list[dict[str, object]]:
        by_page: dict[int, list[_Placement]] = {}
        for placement in placements:
            by_page.setdefault(placement.page_number, []).append(placement)
        overlaps: list[dict[str, object]] = []
        for page_number, page_items in by_page.items():
            for index, first in enumerate(page_items):
                ax0, ay0, ax1, ay1 = first.bbox
                for second in page_items[index + 1 :]:
                    bx0, by0, bx1, by1 = second.bbox
                    width = min(ax1, bx1) - max(ax0, bx0)
                    height = min(ay1, by1) - max(ay0, by0)
                    if width > 0.75 and height > 0.75:
                        overlaps.append(
                            {
                                "page": page_number,
                                "first": first.segment_id,
                                "second": second.segment_id,
                                "intersection_area": round(width * height, 2),
                            }
                        )
        return overlaps

    @staticmethod
    def _verify_output(source_pages: list, output_path: Path) -> dict[str, object]:
        output_reader = PdfReader(str(output_path), strict=False)
        same_count = len(source_pages) == len(output_reader.pages)
        same_sizes = same_count and all(
            abs(float(source.mediabox.width) - float(output.mediabox.width)) < 0.1
            and abs(float(source.mediabox.height) - float(output.mediabox.height)) < 0.1
            for source, output in zip(source_pages, output_reader.pages)
        )
        return {
            "valid": same_count and same_sizes,
            "page_count": len(output_reader.pages),
            "page_sizes_preserved": same_sizes,
        }

    @staticmethod
    def _write_html_report(path: Path, payload: dict[str, object]) -> None:
        validation = payload["validation"]
        assert isinstance(validation, dict)
        issues = validation.get("issues", [])
        issue_rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(item.get('severity', '')))}</td>"
            f"<td>{html.escape(str(item.get('segment_id', '')))}</td>"
            f"<td>{html.escape(str(item.get('message', '')))}</td>"
            "</tr>"
            for item in issues
        )
        document = f"""<!doctype html>
<meta charset="utf-8">
<title>原版面替换质量报告</title>
<style>
body{{font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;
max-width:960px;margin:48px auto;padding:0 24px;color:#172321}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border:1px solid #dcded5;border-radius:12px;padding:16px}}
.card b{{display:block;font-size:24px}} table{{width:100%;border-collapse:collapse;margin-top:24px}}
th,td{{text-align:left;border-bottom:1px solid #ddd;padding:9px;vertical-align:top}}
.passed{{color:#216557}} .review{{color:#a23d34}}
</style>
<h1>原版面替换质量报告</h1>
<p class="{'passed' if str(payload['status']).startswith('passed') else 'review'}">
状态：{html.escape(str(payload['status']))}</p>
<div class="grid">
<div class="card">替换段落<b>{payload['replaced_segments']}</b></div>
<div class="card">溢出错误<b>{payload['overflow_errors']}</b></div>
<div class="card">复杂背景修补<b>{payload['complex_background_repairs']}</b></div>
<div class="card">绕图段落<b>{payload.get('contour_flow_segments', 0)}</b></div>
<div class="card">最小字号<b>{payload['minimum_rendered_font_size']}</b></div>
<div class="card">高密度待复核页<b>{len(payload.get('dense_pages_requiring_review', []))}</b></div>
</div>
<p>{html.escape(str(payload['note']))}</p>
<table><thead><tr><th>级别</th><th>段落</th><th>信息</th></tr></thead>
<tbody>{issue_rows or '<tr><td colspan="3">无校验问题</td></tr>'}</tbody></table>
<details><summary>完整 JSON</summary><pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre></details>
"""
        path.write_text(document, encoding="utf-8")


_PAGE_WORKER_DOCUMENT = None
_PAGE_WORKER_RENDERER: LayoutPreservingRenderer | None = None
_PAGE_WORKER_TEMP_DIR: Path | None = None


def _initialize_page_worker(
    source_path: str,
    render_dpi: int,
    minimum_font_size: float,
    temp_dir: str,
) -> None:
    import fitz

    global _PAGE_WORKER_DOCUMENT
    global _PAGE_WORKER_RENDERER
    global _PAGE_WORKER_TEMP_DIR
    cv2.setNumThreads(1)
    _PAGE_WORKER_DOCUMENT = fitz.open(source_path)
    _PAGE_WORKER_RENDERER = LayoutPreservingRenderer(
        render_dpi=render_dpi,
        minimum_font_size=minimum_font_size,
        max_workers=1,
    )
    _PAGE_WORKER_TEMP_DIR = Path(temp_dir)


def _render_page_worker(request: _PageRenderRequest) -> _PageRenderResult:
    if (
        _PAGE_WORKER_DOCUMENT is None
        or _PAGE_WORKER_RENDERER is None
        or _PAGE_WORKER_TEMP_DIR is None
    ):
        raise RuntimeError("页面渲染进程未正确初始化。")
    return _PAGE_WORKER_RENDERER._prepare_page(
        _PAGE_WORKER_DOCUMENT,
        request,
        _PAGE_WORKER_TEMP_DIR,
    )
