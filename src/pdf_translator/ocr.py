from __future__ import annotations

import json
import platform
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import PDFAnalysisError
from .utils import normalize_whitespace


@dataclass
class OCRBlock:
    text: str
    bbox: list[float]
    line_bboxes: list[list[float]]
    font_size: float
    block_type: str
    confidence: float


@dataclass
class _OCRLine:
    text: str
    confidence: float
    bbox: list[float]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class MacVisionOCR:
    """Optional local OCR powered by the macOS Vision framework."""

    def __init__(self, cache_dir: Path, dpi: int = 180):
        self.cache_dir = cache_dir
        self.dpi = dpi
        self.source_path = (
            Path(__file__).resolve().parent / "assets" / "vision_ocr.swift"
        )
        self.binary_path = cache_dir / "vision_ocr"
        self.module_cache = cache_dir / "swift-module-cache"

    def available(self) -> bool:
        return (
            platform.system() == "Darwin"
            and self.source_path.is_file()
            and shutil.which("swiftc") is not None
        )

    def recognize_page(self, page: Any, page_number: int) -> list[OCRBlock]:
        if not self.available():
            raise PDFAnalysisError(
                "扫描件 OCR 需要 macOS Vision 和 Swift 命令行工具；当前环境不可用。"
            )
        self._ensure_binary()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        image_path: Path | None = None
        try:
            image_path, source_rect = self._source_image_or_render(
                page, page_number
            )
            completed = subprocess.run(
                [str(self.binary_path), str(image_path), "en-US"],
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "未知错误"
                raise PDFAnalysisError(f"macOS Vision OCR 失败：{detail}")
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                raise PDFAnalysisError("macOS Vision OCR 返回了无效结果。") from exc
            lines = self._lines_from_payload(
                payload,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                source_rect=source_rect,
            )
            return self._group_lines(lines, float(page.rect.width))
        finally:
            if image_path is not None:
                image_path.unlink(missing_ok=True)

    def _source_image_or_render(
        self, page: Any, page_number: int
    ) -> tuple[Path, list[float]]:
        """Prefer the original full-page scan to avoid rasterizer-softened text."""
        try:
            images = page.get_images(full=True)
            if len(images) == 1:
                xref = images[0][0]
                extracted = page.parent.extract_image(xref)
                width = int(extracted.get("width", 0))
                height = int(extracted.get("height", 0))
                placements = page.get_image_rects(xref, transform=True)
                if len(placements) != 1:
                    raise ValueError("full-page scan must have one placement")
                placement_rect, matrix = placements[0]
                if (
                    abs(float(matrix.b)) > 0.01
                    or abs(float(matrix.c)) > 0.01
                    or float(matrix.a) <= 0
                    or float(matrix.d) <= 0
                ):
                    raise ValueError("rotated scan image uses rendered OCR")
                page_ratio = float(placement_rect.width) / max(
                    float(placement_rect.height), 1
                )
                image_ratio = width / max(height, 1)
                if (
                    width >= 900
                    and height >= 900
                    and abs(page_ratio - image_ratio) <= 0.035
                ):
                    extension = str(extracted.get("ext", "png")).lower()
                    if extension not in {"png", "jpg", "jpeg", "tiff", "heic"}:
                        extension = "png"
                    image_path = self.cache_dir / (
                        f"ocr_page_{page_number:05d}.{extension}"
                    )
                    image_path.write_bytes(extracted["image"])
                    return image_path, [
                        float(placement_rect.x0),
                        float(placement_rect.y0),
                        float(placement_rect.x1),
                        float(placement_rect.y1),
                    ]
        except Exception:
            pass

        image_path = self.cache_dir / f"ocr_page_{page_number:05d}.png"
        pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
        pixmap.save(str(image_path))
        return image_path, [
            float(page.rect.x0),
            float(page.rect.y0),
            float(page.rect.x1),
            float(page.rect.y1),
        ]

    def _ensure_binary(self) -> None:
        if (
            self.binary_path.is_file()
            and self.binary_path.stat().st_mtime >= self.source_path.stat().st_mtime
        ):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.module_cache.mkdir(parents=True, exist_ok=True)
        temporary = self.binary_path.with_suffix(".tmp")
        completed = subprocess.run(
            [
                "swiftc",
                "-module-cache-path",
                str(self.module_cache),
                str(self.source_path),
                "-o",
                str(temporary),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "未知错误"
            raise PDFAnalysisError(f"无法编译 macOS Vision OCR 辅助程序：{detail}")
        temporary.chmod(0o755)
        temporary.replace(self.binary_path)

    @staticmethod
    def _lines_from_payload(
        payload: list[dict],
        page_width: float,
        page_height: float,
        source_rect: list[float] | None = None,
    ) -> list[_OCRLine]:
        source_x0, source_y0, source_x1, source_y1 = source_rect or [
            0.0,
            0.0,
            page_width,
            page_height,
        ]
        source_width = source_x1 - source_x0
        source_height = source_y1 - source_y0
        lines: list[_OCRLine] = []
        for item in payload:
            text = normalize_whitespace(str(item.get("text", "")))
            confidence = float(item.get("confidence", 0.0))
            # Below this confidence level, Vision results on illustrated pages
            # are dominated by image textures, decorative rules, and rotated
            # margin fragments. Keeping them creates bogus translation boxes.
            if not text or confidence < 0.35:
                continue
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            width = float(item.get("width", 0.0))
            height = float(item.get("height", 0.0))
            x0 = source_x0 + x * source_width
            x1 = source_x0 + (x + width) * source_width
            y0 = source_y0 + (1.0 - y - height) * source_height
            y1 = source_y0 + (1.0 - y) * source_height
            lines.append(
                _OCRLine(
                    text=text,
                    confidence=confidence,
                    bbox=[
                        round(x0, 2),
                        round(y0, 2),
                        round(x1, 2),
                        round(y1, 2),
                    ],
                )
            )
        return lines

    @staticmethod
    def _group_lines(lines: list[_OCRLine], page_width: float) -> list[OCRBlock]:
        if not lines:
            return []
        horizontal = [line for line in lines if line.width >= line.height * 1.2]
        vertical = [line for line in lines if line not in horizontal]
        groups: list[list[_OCRLine]] = []
        current: list[_OCRLine] = []

        for line in horizontal:
            if not current:
                current = [line]
                continue
            previous = current[-1]
            x_tolerance = max(14.0, page_width * 0.065)
            vertical_gap = line.y0 - previous.y1
            same_column = abs(line.x0 - previous.x0) <= x_tolerance
            follows_downward = line.y0 >= previous.y0 - 1.0
            close_enough = vertical_gap <= max(previous.height, line.height) * 1.8
            similar_scale = (
                max(previous.height, line.height)
                / max(min(previous.height, line.height), 0.1)
                < 1.55
            )
            same_text_role = (
                MacVisionOCR._looks_like_heading(previous.text)
                == MacVisionOCR._looks_like_heading(line.text)
            )
            if (
                same_column
                and follows_downward
                and close_enough
                and similar_scale
                and same_text_role
            ):
                current.append(line)
            else:
                groups.append(current)
                current = [line]
        if current:
            groups.append(current)
        groups.extend([[line] for line in vertical])

        median_height = statistics.median(line.height for line in horizontal) if horizontal else 8
        blocks: list[OCRBlock] = []
        for group in groups:
            text = " ".join(line.text for line in group)
            x0 = min(line.x0 for line in group)
            y0 = min(line.y0 for line in group)
            x1 = max(line.x1 for line in group)
            y1 = max(line.y1 for line in group)
            average_height = statistics.mean(line.height for line in group)
            uppercase_ratio = sum(character.isupper() for character in text) / max(
                sum(character.isalpha() for character in text), 1
            )
            if re.fullmatch(r"\d{1,5}", text):
                block_type = "page_number"
            elif average_height >= median_height * 2.1 and len(text) <= 100:
                block_type = "title"
            elif uppercase_ratio > 0.72 and len(text) <= 120:
                block_type = "heading"
            elif len(group) == 1 and group[0].width < group[0].height * 1.2:
                block_type = "other"
            else:
                block_type = "paragraph"
            blocks.append(
                OCRBlock(
                    text=text,
                    bbox=[x0, y0, x1, y1],
                    line_bboxes=[line.bbox for line in group],
                    font_size=round(max(7.5, average_height * 0.9), 2),
                    block_type=block_type,
                    confidence=round(
                        statistics.mean(line.confidence for line in group), 3
                    ),
                )
            )
        return blocks

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        letters = [character for character in text if character.isalpha()]
        if len(letters) < 3 or len(text) > 90:
            return False
        return sum(character.isupper() for character in letters) / len(letters) >= 0.9
