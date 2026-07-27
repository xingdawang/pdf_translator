from pathlib import Path
from dataclasses import replace
import json

import numpy as np
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from pdf_translator.layout_renderer import (
    LayoutPreservingRenderer,
    _LayoutFragment,
)
from pdf_translator.models import (
    Segment,
    TaskSettings,
    TranslationTask,
    ValidationReport,
)
from pdf_translator.placeholders import PlaceholderService
from pdf_translator.utils import sha256_text, utc_now
from scripts.generate_layout_stress_text import make_stress_text


def _source_pdf(path: Path) -> Path:
    pdf = canvas.Canvas(str(path), pagesize=(300, 200))
    pdf.setFillColor(HexColor("#d8edf0"))
    pdf.rect(0, 0, 300, 200, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#397c86"))
    pdf.circle(245, 65, 38, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#101817"))
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(24, 151, "MINERAL")
    pdf.drawString(24, 132, "GUIDE")
    pdf.save()
    return path


def _task(source: Path) -> TranslationTask:
    now = utc_now()
    source_text = "MINERAL GUIDE"
    segment = Segment(
        segment_id="P0001-S000001",
        row_key="00000001",
        page_number=1,
        reading_order=1,
        block_type="paragraph",
        source_text=source_text,
        protected_text=source_text,
        source_text_hash=sha256_text(source_text),
        bbox=[24, 33, 165, 70],
        erase_bboxes=[[24, 33, 90, 50], [24, 52, 165, 70]],
        font_name="Helvetica-Bold",
        font_size=16,
        translated_text="矿物指南避开图片区域",
    )
    return TranslationTask(
        task_id="20260726-layout-test",
        source_path=str(source),
        source_filename=source.name,
        source_file_hash="a" * 64,
        source_size_bytes=source.stat().st_size,
        created_at=now,
        updated_at=now,
        status="translated",
        page_count=1,
        text_page_count=1,
        scanned_page_count=0,
        image_count=0,
        parser_version="test",
        settings=TaskSettings(),
        segments=[segment],
    )


def test_layout_renderer_preserves_page_and_reports_fit(tmp_path):
    source = _source_pdf(tmp_path / "source.pdf")
    task = _task(source)
    validation = ValidationReport(
        task_id=task.task_id,
        created_at=utc_now(),
        total_segments=1,
        translated_segments=1,
        blocking_errors=0,
        warnings=0,
        issues=[],
        can_generate=True,
    )

    outputs = LayoutPreservingRenderer(render_dpi=150).generate(
        task,
        validation,
        tmp_path / "task",
    )

    reader = PdfReader(str(outputs.layout_pdf))
    assert len(reader.pages) == 1
    assert float(reader.pages[0].mediabox.width) == 300
    assert float(reader.pages[0].mediabox.height) == 200
    contents = reader.pages[0]["/Contents"].get_object()
    assert contents.get("/Filter") == "/FlateDecode"
    assert len(contents._data) < len(contents.get_data())
    image_filters = [
        str(item.get_object().get("/Filter"))
        for item in reader.pages[0]["/Resources"].get("/XObject", {}).values()
        if item.get_object().get("/Subtype") == "/Image"
    ]
    assert image_filters
    assert all("ASCII85Decode" not in item for item in image_filters)
    assert outputs.replaced_segments == 1
    assert outputs.overflow_errors == 0
    report = json.loads(outputs.quality_json.read_text(encoding="utf-8"))
    assert report["page_sizes_preserved"]
    assert report["status"] == "passed"
    assert report["contour_flow_segments"] == 1
    assert report["repair_image_layers"] == 1
    assert report["repair_image_tiles"] >= 1
    assert outputs.quality_html.exists()


def test_layout_renderer_allows_google_changed_number_unit_placeholder(tmp_path):
    source = _source_pdf(tmp_path / "source.pdf")
    task = _task(source)
    segment = task.segments[0]
    protected = PlaceholderService().protect(
        "MINERAL GUIDE 15 km",
        task.task_id,
        segment.segment_id,
    )
    segment.source_text = "MINERAL GUIDE 15 km"
    segment.protected_text = protected.protected_text
    segment.placeholders = protected.placeholders
    segment.translated_text = "矿物指南，距离约为 24 公里"
    validation = ValidationReport(
        task_id=task.task_id,
        created_at=utc_now(),
        total_segments=1,
        translated_segments=1,
        blocking_errors=0,
        warnings=1,
        issues=[],
        can_generate=True,
    )

    outputs = LayoutPreservingRenderer(render_dpi=150).generate(
        task,
        validation,
        tmp_path / "task",
    )

    assert outputs.layout_pdf.exists()
    assert outputs.overflow_errors == 0


def test_layout_renderer_leaves_source_fallback_region_untouched(tmp_path):
    source = _source_pdf(tmp_path / "source.pdf")
    task = _task(source)
    segment = task.segments[0]
    segment.translated_text = segment.source_text
    segment.status = "source_fallback"
    segment.fallback_reason = "翻译件中缺少对应译文"
    validation = ValidationReport(
        task_id=task.task_id,
        created_at=utc_now(),
        total_segments=1,
        translated_segments=1,
        blocking_errors=0,
        warnings=1,
        issues=[],
        can_generate=True,
    )

    outputs = LayoutPreservingRenderer(render_dpi=150).generate(
        task,
        validation,
        tmp_path / "task",
    )

    report = outputs.quality_json.read_text(encoding="utf-8")
    assert outputs.replaced_segments == 0
    assert '"source_fallback_segments": 1' in report
    assert "MINERAL" in PdfReader(str(outputs.layout_pdf)).pages[0].extract_text()


def test_narrow_multiline_column_is_not_mistaken_for_vertical_text(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.bbox = [24, 20, 70, 150]
    segment.erase_bboxes = [
        [24, 20, 68, 32],
        [24, 36, 69, 48],
        [24, 52, 67, 64],
    ]

    assert not LayoutPreservingRenderer._is_rotated_segment(segment, "窄栏多行中文")


def test_single_vertical_ocr_line_is_rotated(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.bbox = [24, 20, 34, 100]
    segment.erase_bboxes = [[24, 20, 34, 100]]

    assert LayoutPreservingRenderer._is_rotated_segment(segment, "竖排标题")


def test_layout_stress_text_does_not_count_index_numbers_twice():
    text = make_stress_text("agate 88, 96, 108, 219", [], 0.4)

    assert text == "排版"


def test_repair_patch_clusters_stay_local_and_bounded():
    patch = np.zeros((10, 10, 4), dtype=np.uint8)
    patch[:, :, :3] = 240
    patch[:, :, 3] = 255
    patches = [
        ((0, 0, 10, 10), patch),
        ((12, 0, 22, 10), patch),
        ((100, 100, 110, 110), patch),
        ((200, 200, 210, 210), patch),
    ]

    groups = LayoutPreservingRenderer._cluster_repair_patches(
        patches,
        gap=3,
        maximum_groups=3,
    )

    assert len(groups) == 3
    assert any(group[0] == (0, 0, 22, 10) for group in groups)
    assert sum(len(group[1]) for group in groups) == len(patches)
    bounded = LayoutPreservingRenderer._cluster_repair_patches(
        patches,
        gap=3,
        maximum_groups=2,
    )
    assert len(bounded) == 2


def test_dense_index_page_is_identified_for_manual_review(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.source_text = "mineral 123 " * 500
    segment.erase_bboxes = [[24, 33, 165, 40]] * 250

    detail = LayoutPreservingRenderer._dense_page_detail(
        8,
        [segment],
        563.04,
        737.28,
    )

    assert detail is not None
    assert detail["page"] == 8
    assert detail["ocr_lines"] == 250


def test_dense_flat_index_page_gets_one_clean_text_slab(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.bbox = [35, 40, 265, 185]
    segment.erase_bboxes = [[35, 40 + index * 0.5, 265, 44 + index * 0.5] for index in range(250)]
    page = np.full((400, 600, 3), 250, dtype=np.uint8)
    page[30:370:12, 40:560] = 30

    cleanup = LayoutPreservingRenderer._dense_flat_page_cleanup(
        page,
        [segment],
        page_width=300,
        page_height=200,
    )

    assert cleanup is not None
    box, color = cleanup
    assert box[0] < segment.bbox[0]
    assert box[2] > segment.bbox[2]
    assert min(color) > 0.95


def test_photo_heavy_dense_page_does_not_get_flat_cleanup(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    page = np.full((400, 600, 3), (90, 130, 170), dtype=np.uint8)

    assert (
        LayoutPreservingRenderer._dense_flat_page_cleanup(
            page,
            task.segments,
            page_width=300,
            page_height=200,
        )
        is None
    )


def test_table_translation_is_partitioned_by_source_cells():
    fragments = [
        _LayoutFragment("CS5702", [46, 132, 72, 142], 7),
        _LayoutFragment(
            "SOFTWARE ENGINEERING REQUIREMENTS",
            [88, 132, 238, 142],
            7,
        ),
        _LayoutFragment("D1", [401, 132, 410, 142], 7),
        _LayoutFragment("1", [463, 132, 467, 142], 7),
        _LayoutFragment("6", [534, 132, 538, 142], 7),
    ]

    chunks = LayoutPreservingRenderer._partition_translation(
        "CS5702 软件工程需求 D1 1 6",
        fragments,
    )

    assert chunks == ["CS5702", "软件工程需求", "D1", "1", "6"]


def test_white_text_on_blue_background_uses_light_ink_polarity():
    crop = np.full((30, 80, 3), (20, 166, 220), dtype=np.uint8)
    crop[8:22, 20:60] = (255, 255, 255)

    assert (
        LayoutPreservingRenderer._ink_polarity(crop, 145.0) == "light"
    )


def test_numbered_table_headers_keep_each_label_with_its_translation():
    fragments = [
        _LayoutFragment("7.1 Date", [49, 330, 83, 342], 9),
        _LayoutFragment("7.2 Signature", [134, 330, 190, 342], 9),
        _LayoutFragment("7.3 Capacity", [276, 330, 328, 342], 9),
        _LayoutFragment(
            "7.4 Official stamp or seal",
            [417, 330, 523, 342],
            9,
        ),
    ]

    chunks = LayoutPreservingRenderer._partition_translation(
        "7.1 日期 7.2 签名 7.3 身份 7.4 公章或印章",
        fragments,
    )

    assert chunks == [
        "7.1 日期",
        "7.2 签名",
        "7.3 身份",
        "7.4 公章或印章",
    ]


def test_flat_background_detector_prefers_dominant_page_color():
    crop = np.full((40, 100, 3), (157, 204, 213), dtype=np.uint8)
    crop[10:30, 20:80] = (15, 20, 20)

    background, ratio = LayoutPreservingRenderer._dominant_background(crop)

    assert ratio > 0.6
    assert np.allclose(background, (157, 204, 213), atol=2)


def test_linear_background_histogram_matches_previous_exact_result():
    rng = np.random.default_rng(20260726)
    crop = rng.integers(0, 256, size=(117, 83, 3), dtype=np.uint8)
    pixels = crop.reshape(-1, 3).astype(np.int16)
    quantized = (pixels // 12).astype(np.int16)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    candidates = pixels[
        np.max(np.abs(quantized - dominant), axis=1) <= 1
    ]
    expected_background = np.median(candidates, axis=0).astype(np.uint8)
    expected_ratio = float(np.max(counts) / len(pixels))

    background, ratio = LayoutPreservingRenderer._dominant_background(crop)

    assert np.array_equal(background, expected_background)
    assert ratio == expected_ratio


def test_identical_layout_generation_reuses_verified_cache(tmp_path):
    source = _source_pdf(tmp_path / "source.pdf")
    task = _task(source)
    validation = ValidationReport(
        task_id=task.task_id,
        created_at=utc_now(),
        total_segments=1,
        translated_segments=1,
        blocking_errors=0,
        warnings=0,
        issues=[],
        can_generate=True,
    )
    renderer = LayoutPreservingRenderer(render_dpi=150)

    first = renderer.generate(task, validation, tmp_path / "task")
    first_mtime = first.layout_pdf.stat().st_mtime_ns
    second = renderer.generate(task, validation, tmp_path / "task")

    assert not first.cache_hit
    assert second.cache_hit
    assert second.layout_pdf.stat().st_mtime_ns == first_mtime
    report = json.loads(second.quality_json.read_text(encoding="utf-8"))
    assert report["generation_fingerprint"]

    task.segments[0].translated_text = "更新后的矿物指南"
    changed = renderer.generate(task, validation, tmp_path / "task")

    assert not changed.cache_hit


def test_multi_page_layout_uses_requested_parallel_workers(tmp_path):
    source = tmp_path / "four-pages.pdf"
    pdf = canvas.Canvas(str(source), pagesize=(300, 200))
    for page_number in range(1, 5):
        pdf.setFillColor(HexColor("#d8edf0"))
        pdf.rect(0, 0, 300, 200, fill=1, stroke=0)
        pdf.setFillColor(HexColor("#101817"))
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(24, 151, f"MINERAL GUIDE {page_number}")
        pdf.showPage()
    pdf.save()

    task = _task(source)
    task.page_count = 4
    task.source_page_count = 4
    task.settings.page_end = 4
    task.segments = [
        replace(
            task.segments[0],
            segment_id=f"P{page_number:04d}-S000001",
            row_key=f"{page_number:08d}",
            page_number=page_number,
            source_text=f"MINERAL GUIDE {page_number}",
            protected_text=f"MINERAL GUIDE {page_number}",
            source_text_hash=sha256_text(f"MINERAL GUIDE {page_number}"),
        )
        for page_number in range(1, 5)
    ]
    validation = ValidationReport(
        task_id=task.task_id,
        created_at=utc_now(),
        total_segments=4,
        translated_segments=4,
        blocking_errors=0,
        warnings=0,
        issues=[],
        can_generate=True,
    )

    outputs = LayoutPreservingRenderer(
        render_dpi=120,
        max_workers=2,
    ).generate(task, validation, tmp_path / "task")

    report = json.loads(outputs.quality_json.read_text(encoding="utf-8"))
    assert outputs.render_workers == 2
    assert report["render_workers"] == 2
    assert report["repair_image_layers"] == 4
    assert len(PdfReader(str(outputs.layout_pdf)).pages) == 4
