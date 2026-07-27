from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw


CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def _page_array(page: fitz.Page, dpi: int) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    return np.asarray(
        Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples).convert(
            "RGB"
        )
    )


def _text_boxes(page: fitz.Page) -> list[list[float]]:
    page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, sort=False)
    return [
        [float(value) for value in span["bbox"]]
        for block in page_dict.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if str(span.get("text", "")).strip()
    ]


def _cjk_boxes(page: fitz.Page) -> list[list[float]]:
    page_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT, sort=False)
    return [
        [float(value) for value in span["bbox"]]
        for block in page_dict.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if CJK_PATTERN.search(str(span.get("text", "")))
    ]


def _allowed_text_mask(
    shape: tuple[int, int],
    page_width: float,
    page_height: float,
    boxes: list[list[float]],
    padding: int = 4,
) -> np.ndarray:
    height, width = shape
    scale_x = width / max(page_width, 1)
    scale_y = height / max(page_height, 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        px0 = max(0, math.floor(x0 * scale_x) - padding)
        py0 = max(0, math.floor(y0 * scale_y) - padding)
        px1 = min(width, math.ceil(x1 * scale_x) + padding)
        py1 = min(height, math.ceil(y1 * scale_y) + padding)
        if px1 > px0 and py1 > py0:
            cv2.rectangle(mask, (px0, py0), (px1, py1), 255, -1)
    return mask


def _overlap_metrics(boxes: list[list[float]]) -> tuple[int, float]:
    severe_pairs = 0
    maximum_ratio = 0.0
    for index, first in enumerate(boxes):
        first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
        if first_area <= 0:
            continue
        for second in boxes[index + 1 :]:
            intersection_width = max(
                0.0, min(first[2], second[2]) - max(first[0], second[0])
            )
            intersection_height = max(
                0.0, min(first[3], second[3]) - max(first[1], second[1])
            )
            intersection = intersection_width * intersection_height
            if intersection <= 0:
                continue
            second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
            ratio = intersection / max(min(first_area, second_area), 0.01)
            maximum_ratio = max(maximum_ratio, ratio)
            severe_pairs += int(ratio >= 0.15)
    return severe_pairs, maximum_ratio


def audit(
    source_path: Path,
    output_path: Path,
    dpi: int,
) -> tuple[dict[str, object], list[tuple[int, np.ndarray]]]:
    source = fitz.open(source_path)
    output = fitz.open(output_path)
    try:
        if source.page_count != output.page_count:
            raise ValueError(
                f"页数不一致：原文 {source.page_count}，输出 {output.page_count}"
            )
        pages: list[dict[str, object]] = []
        rendered: list[tuple[int, np.ndarray]] = []
        for page_index in range(source.page_count):
            source_page = source.load_page(page_index)
            output_page = output.load_page(page_index)
            source_array = _page_array(source_page, dpi)
            output_array = _page_array(output_page, dpi)
            if source_array.shape != output_array.shape:
                raise ValueError(f"第 {page_index + 1} 页渲染尺寸不一致")

            difference = (
                np.max(
                    np.abs(
                        source_array.astype(np.int16) - output_array.astype(np.int16)
                    ),
                    axis=2,
                )
                >= 24
            )
            source_boxes = _text_boxes(source_page)
            output_boxes = _text_boxes(output_page)
            allowed_mask = _allowed_text_mask(
                difference.shape,
                float(source_page.rect.width),
                float(source_page.rect.height),
                source_boxes + output_boxes,
            )
            outside_difference = difference & (allowed_mask == 0)
            changed_pixels = int(np.count_nonzero(difference))
            outside_pixels = int(np.count_nonzero(outside_difference))
            page_pixels = difference.size
            overlap_pairs, maximum_overlap = _overlap_metrics(_cjk_boxes(output_page))
            outside_ratio = outside_pixels / max(page_pixels, 1)
            outside_share = outside_pixels / max(changed_pixels, 1)
            risk_score = (
                outside_ratio * 10_000
                + outside_share * 10
                + min(overlap_pairs, 30) * 0.25
                + maximum_overlap
            )
            pages.append(
                {
                    "page": page_index + 1,
                    "changed_ratio": round(changed_pixels / page_pixels, 6),
                    "outside_text_change_ratio": round(outside_ratio, 6),
                    "outside_text_change_share": round(outside_share, 6),
                    "cjk_overlap_pairs": overlap_pairs,
                    "maximum_cjk_overlap_ratio": round(maximum_overlap, 4),
                    "risk_score": round(risk_score, 4),
                }
            )
            rendered.append((page_index + 1, output_array))

        ranked = sorted(
            pages,
            key=lambda item: float(item["risk_score"]),
            reverse=True,
        )
        report: dict[str, object] = {
            "source": str(source_path.resolve()),
            "output": str(output_path.resolve()),
            "dpi": dpi,
            "page_count": source.page_count,
            "pages_checked": len(pages),
            "pages_with_outside_text_change_over_0_1_percent": sum(
                float(item["outside_text_change_ratio"]) > 0.001 for item in pages
            ),
            "pages_with_cjk_overlap_pairs": sum(
                int(item["cjk_overlap_pairs"]) > 0 for item in pages
            ),
            "maximum_outside_text_change_ratio": max(
                (float(item["outside_text_change_ratio"]) for item in pages),
                default=0.0,
            ),
            "top_risk_pages": ranked[:30],
            "pages": pages,
        }
        return report, rendered
    finally:
        output.close()
        source.close()


def _write_contact_sheet(
    path: Path,
    ranked_pages: list[dict[str, object]],
    rendered: list[tuple[int, np.ndarray]],
    maximum_pages: int,
) -> None:
    images_by_page = dict(rendered)
    selected = ranked_pages[:maximum_pages]
    columns = 4
    tile_width = 320
    label_height = 34
    tiles: list[Image.Image] = []
    for item in selected:
        page_number = int(item["page"])
        image = Image.fromarray(images_by_page[page_number]).convert("RGB")
        image.thumbnail((tile_width, 420))
        tile = Image.new(
            "RGB",
            (tile_width, image.height + label_height),
            "white",
        )
        tile.paste(image, ((tile_width - image.width) // 2, label_height))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 8),
            (
                f"Page {page_number}  risk={item['risk_score']}  "
                f"outside={item['outside_text_change_ratio']}  "
                f"overlaps={item['cjk_overlap_pairs']}"
            ),
            fill="black",
        )
        tiles.append(tile)
    rows = math.ceil(len(tiles) / columns)
    row_heights = [
        max(
            (
                tiles[index].height
                for index in range(row * columns, min((row + 1) * columns, len(tiles)))
            ),
            default=0,
        )
        for row in range(rows)
    ]
    sheet = Image.new(
        "RGB",
        (columns * tile_width, sum(row_heights)),
        "#d9dde1",
    )
    y = 0
    for row, row_height in enumerate(row_heights):
        for column in range(columns):
            index = row * columns + column
            if index >= len(tiles):
                break
            sheet.paste(tiles[index], (column * tile_width, y))
        y += row_height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐页检查原版面翻译 PDF 的背景变化和中文文字框重叠。"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--top", type=int, default=24)
    args = parser.parse_args()

    report, rendered = audit(args.source, args.output, args.dpi)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.contact_sheet:
        _write_contact_sheet(
            args.contact_sheet,
            list(report["top_risk_pages"]),
            rendered,
            max(1, args.top),
        )
    print(json.dumps({key: value for key, value in report.items() if key != "pages"}))


if __name__ == "__main__":
    main()
