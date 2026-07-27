from pathlib import Path
from dataclasses import replace
import json

import fitz
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

from pdf_translator.layout_renderer import (
    LayoutPreservingRenderer,
    _LayoutFragment,
)
from pdf_translator.models import (
    ParagraphAnchor,
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
    assert outputs.replaced_segments == 1
    assert outputs.overflow_errors == 0
    report = json.loads(outputs.quality_json.read_text(encoding="utf-8"))
    assert report["page_sizes_preserved"]
    assert report["status"] == "passed"
    assert report["contour_flow_segments"] == 1
    assert report["repair_image_layers"] == 0
    assert report["repair_image_tiles"] == 0
    assert report["vector_text_redactions"] >= 1
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


def test_layout_preflight_preserves_source_when_translation_cannot_fit(tmp_path):
    source = _source_pdf(tmp_path / "source.pdf")
    task = _task(source)
    segment = task.segments[0]
    segment.bbox = [24, 33, 36, 38]
    segment.erase_bboxes = [[24, 33, 36, 38]]
    segment.translated_text = "无法放进极小文本框的很长译文"
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

    report = json.loads(outputs.quality_json.read_text(encoding="utf-8"))
    assert outputs.replaced_segments == 0
    assert outputs.overflow_errors == 0
    assert report["layout_fallback_segments"] == 1
    assert report["layout_fallback_segment_ids"] == [segment.segment_id]
    assert "MINERAL" in PdfReader(str(outputs.layout_pdf)).pages[0].extract_text()


def test_layout_preflight_keeps_irregular_contour_instead_of_using_union_box(
    tmp_path,
):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.bbox = [24, 33, 170, 110]
    segment.erase_bboxes = [
        [120, 33, 170, 42],
        [120, 45, 170, 54],
        [24, 57, 170, 66],
    ]
    segment.translated_text = (
        "这段译文非常长，无法沿着绕开图片的三行轮廓完整排入，"
        "因此必须在擦除原文之前被安全检查识别出来。"
        "即使外接矩形仍有空白，也不能越过逐行轮廓占用旁边的插图区域。"
    )
    renderer = LayoutPreservingRenderer()

    fallbacks = renderer._layout_preflight_fallbacks(
        [segment],
        {segment.segment_id: segment.translated_text},
        {segment.segment_id: []},
    )

    assert segment.segment_id in fallbacks


def test_layout_preflight_draws_duplicate_compact_region_only_once(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    first = task.segments[0]
    first.translated_text = "矿物指南"
    first.erase_bboxes = [list(first.bbox)]
    duplicate = replace(
        first,
        segment_id="P0001-S000002",
        row_key="00000002",
        reading_order=2,
    )
    renderer = LayoutPreservingRenderer()

    fallbacks = renderer._layout_preflight_fallbacks(
        [first, duplicate],
        {
            first.segment_id: first.translated_text,
            duplicate.segment_id: duplicate.translated_text,
        },
        {
            first.segment_id: [],
            duplicate.segment_id: [],
        },
    )

    assert first.segment_id not in fallbacks
    assert duplicate.segment_id in fallbacks
    assert "重复文本区域" in fallbacks[duplicate.segment_id]


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


def test_oriented_source_font_size_uses_line_thickness_not_line_length(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.bbox = [24, 20, 32, 140]
    segment.erase_bboxes = [[24, 20, 32, 140]]
    segment.font_size = 8
    segment.rotation_degrees = -90

    assert LayoutPreservingRenderer._source_font_size(segment) == 8


def test_explicit_rotation_is_preserved_for_non_right_angle_text(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.rotation_degrees = -35

    assert LayoutPreservingRenderer._segment_rotation(segment, "标签") == -35


def test_rotated_axis_bbox_is_converted_back_to_line_dimensions():
    angle = np.deg2rad(20)
    source_length = 120
    source_thickness = 10
    axis_width = source_length * abs(np.cos(angle)) + source_thickness * abs(
        np.sin(angle)
    )
    axis_height = source_length * abs(np.sin(angle)) + source_thickness * abs(
        np.cos(angle)
    )

    length, thickness = LayoutPreservingRenderer._oriented_box_dimensions(
        axis_width,
        axis_height,
        20,
        source_font_size=8,
    )

    assert length == pytest.approx(source_length)
    assert thickness == pytest.approx(source_thickness)


def test_thin_rotated_single_line_label_accounts_for_paragraph_leading():
    renderer = LayoutPreservingRenderer()
    font_name, _ = renderer.font_resolver.register()

    expanded = renderer._rotated_single_line_height(
        "脉冲信号发送至丘脑底核",
        available_width=151.82,
        available_height=2.68,
        font_name=font_name,
        minimum_font_size=3.5,
    )

    assert expanded == pytest.approx(3.5 * 1.02)
    assert (
        renderer._rotated_single_line_height(
            "脉冲信号发送至丘脑底核",
            available_width=20,
            available_height=2.68,
            font_name=font_name,
            minimum_font_size=3.5,
        )
        == 2.68
    )
    assert (
        renderer._rotated_single_line_height(
            "第一行\n第二行",
            available_width=151.82,
            available_height=2.68,
            font_name=font_name,
            minimum_font_size=3.5,
        )
        == 2.68
    )


def test_irregular_heading_contour_preserves_artwork_indentation():
    artwork_contour = [
        [416, 293, 485, 311],
        [416, 310, 507, 328],
        [362, 327, 515, 345],
        [362, 344, 503, 362],
    ]
    ordinary_lines = [
        [100, 100, 220, 112],
        [102, 112, 218, 124],
    ]

    assert LayoutPreservingRenderer._has_irregular_line_contour(artwork_contour)
    assert not LayoutPreservingRenderer._has_irregular_line_contour(ordinary_lines)


def test_contour_line_break_does_not_split_ascii_number():
    renderer = LayoutPreservingRenderer()
    font_name, _ = renderer.font_resolver.register()
    font_size = 15
    first_line_width = 91
    slots = [
        [0, 0, first_line_width, 18],
        [0, 18, 200, 36],
    ]

    lines, consumed = renderer._pack_text_into_slots(
        "食餐诞生于1953年",
        slots,
        font_name,
        font_size,
    )

    assert lines == ["食餐诞生于", "1953年"]
    assert consumed == len("食餐诞生于1953年")


def test_contour_fit_avoids_orphan_closing_punctuation():
    renderer = LayoutPreservingRenderer()
    font_name, _ = renderer.font_resolver.register()
    text = "三种比例尺的密西西比河地图（略微放大以便复制）。"
    slots = [
        [0, 0, 231, 9],
        [0, 11, 65, 20],
    ]

    _, lines, fitted = renderer._fit_text_to_slots(
        text,
        slots,
        font_name,
        source_size=8.5,
    )

    assert fitted
    assert not renderer._has_orphan_punctuation_line(lines)


def test_short_lowercase_diagram_fragments_are_preserved(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.source_text = "or at q"
    segment.translated_text = "或在 q"
    segment.bbox = [24, 33, 55, 45]
    segment.erase_bboxes = [list(segment.bbox)]
    renderer = LayoutPreservingRenderer()

    fallbacks = renderer._layout_preflight_fallbacks(
        [segment],
        {segment.segment_id: segment.translated_text},
        {segment.segment_id: []},
    )

    assert segment.segment_id in fallbacks
    assert "过短片段" in fallbacks[segment.segment_id]


def test_rotated_compact_labels_with_axis_overlap_are_preserved(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    first = task.segments[0]
    first.source_text = "Satellite track"
    first.translated_text = "卫星轨迹"
    first.bbox = [20, 20, 35, 70]
    first.erase_bboxes = [list(first.bbox)]
    first.rotation_degrees = 83
    second = replace(
        first,
        segment_id="P0001-S000002",
        row_key="00000002",
        reading_order=2,
        source_text="March 1st",
        translated_text="3月1日",
        bbox=[29, 25, 43, 68],
        erase_bboxes=[[29, 25, 43, 68]],
    )
    renderer = LayoutPreservingRenderer()

    fallbacks = renderer._layout_preflight_fallbacks(
        [first, second],
        {
            first.segment_id: first.translated_text,
            second.segment_id: second.translated_text,
        },
        {
            first.segment_id: [],
            second.segment_id: [],
        },
    )

    assert set(fallbacks) == {first.segment_id, second.segment_id}


def test_semantic_paragraph_translation_is_distributed_to_visual_anchors(
    tmp_path,
):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.paragraph_id = segment.segment_id
    segment.source_text = "First source part second source part"
    segment.translated_text = "第一部分译文，第二部分译文。"
    segment.anchors = [
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A001",
            page_number=1,
            reading_order=1,
            bbox=[24, 33, 100, 50],
            erase_bboxes=[[24, 33, 100, 50]],
            source_text="First source part",
            column_id=1,
        ),
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A002",
            page_number=2,
            reading_order=1,
            bbox=[24, 33, 130, 50],
            erase_bboxes=[[24, 33, 130, 50]],
            source_text="second source part",
            column_id=1,
        ),
    ]
    segment.continuation = "cross_page"

    expanded = LayoutPreservingRenderer()._expand_visual_segments(
        [segment],
        ignore_number_warnings=True,
    )

    assert len(expanded) == 2
    assert [item.page_number for item in expanded] == [1, 2]
    assert (
        "".join(item.translated_text for item in expanded).replace(
            " ",
            "",
        )
        == segment.translated_text
    )
    assert all(item.paragraph_id == segment.segment_id for item in expanded)


def test_visual_anchor_split_preserves_labels_when_identifier_would_break(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.source_text = "Sounded Nov. 19, 1937 R 602.5 OIL MILL"
    segment.translated_text = "1937年11月19日测量，R 602.5 油厂"
    segment.continuation = "visual_fragments"
    segment.anchors = [
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A001",
            page_number=1,
            reading_order=1,
            bbox=[20, 20, 90, 32],
            source_text="Sounded Nov. 19, 1937",
        ),
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A002",
            page_number=1,
            reading_order=2,
            bbox=[70, 32, 95, 44],
            source_text="R 602.5",
        ),
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A003",
            page_number=1,
            reading_order=3,
            bbox=[95, 44, 130, 56],
            source_text="OIL MILL",
        ),
    ]

    expanded = LayoutPreservingRenderer()._expand_visual_segments(
        [segment],
        ignore_number_warnings=True,
    )

    assert all(item.status == "source_fallback" for item in expanded)
    assert [item.translated_text for item in expanded] == [
        anchor.source_text for anchor in segment.anchors
    ]
    assert all("标识符" in item.fallback_reason for item in expanded)


def test_short_translation_split_stays_near_source_weight_ratio():
    chunks = LayoutPreservingRenderer._split_text_across_anchors(
        "隆升、埋藏和重结晶侵蚀",
        [
            "uplift and burial and recrystallization",
            "erosion",
        ],
    )

    assert chunks == ["隆升、埋藏和重结晶", "侵蚀"]


def test_placeholder_only_url_is_not_redrawn(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    protected = PlaceholderService().protect(
        "www.example.com",
        task.task_id,
        segment.segment_id,
    )
    segment.source_text = "www.example.com"
    segment.translated_text = protected.protected_text
    segment.placeholders = protected.placeholders

    assert not LayoutPreservingRenderer()._translation_changes_source(segment)


def test_short_source_label_is_not_fed_with_cross_column_body_translation(
    tmp_path,
):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.source_text = "FUEL TANK A separate explanatory paragraph."
    segment.translated_text = "这是一段独立的说明正文。"
    segment.continuation = "cross_column"
    segment.anchors = [
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A001",
            page_number=1,
            reading_order=1,
            bbox=[20, 140, 70, 155],
            source_text="FUEL TANK",
        ),
        ParagraphAnchor(
            anchor_id=f"{segment.segment_id}-A002",
            page_number=1,
            reading_order=2,
            bbox=[100, 20, 280, 100],
            source_text="A separate explanatory paragraph.",
        ),
    ]

    expanded = LayoutPreservingRenderer()._expand_visual_segments(
        [segment],
        ignore_number_warnings=True,
    )

    assert expanded[0].status == "source_fallback"
    assert expanded[0].translated_text == "FUEL TANK"
    assert expanded[1].translated_text == segment.translated_text


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
    segment.erase_bboxes = [
        [35, 40 + index * 0.5, 265, 44 + index * 0.5] for index in range(250)
    ]
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

    assert LayoutPreservingRenderer._ink_polarity(crop, 145.0) == "light"


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


def test_ocr_background_does_not_treat_large_black_title_as_background():
    border_background = np.array([247, 236, 223], dtype=np.float32)
    title_black = np.array([28, 28, 26], dtype=np.uint8)

    background, threshold = LayoutPreservingRenderer._select_ocr_background(
        border_background,
        border_spread=6.7,
        dominant_background=title_black,
        dominant_ratio=0.20,
    )

    assert np.array_equal(background, border_background)
    assert threshold == 14


def test_flat_scan_background_uses_lower_antialias_threshold():
    background, threshold = LayoutPreservingRenderer._select_ocr_background(
        np.array([255, 255, 255], dtype=np.float32),
        border_spread=0,
        dominant_background=np.array([254, 254, 254], dtype=np.uint8),
        dominant_ratio=0.67,
    )

    assert np.allclose(background, (254, 254, 254))
    assert threshold == 6


def test_dominant_flat_color_overrides_icon_contaminated_border():
    background, threshold = LayoutPreservingRenderer._select_ocr_background(
        np.array([208, 208, 208], dtype=np.float32),
        border_spread=37,
        dominant_background=np.array([0, 158, 227], dtype=np.uint8),
        dominant_ratio=0.43,
    )

    assert np.array_equal(background, (0, 158, 227))
    assert threshold == 6


def test_flat_color_text_repair_does_not_create_inpaint_clouds(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.source_kind = "vision_ocr"
    segment.font_size = 18
    segment.bbox = [20, 20, 100, 50]
    segment.erase_bboxes = [[20, 20, 100, 50]]
    blue = np.array([0, 158, 227], dtype=np.uint8)
    page = np.full((100, 160, 3), blue, dtype=np.uint8)
    page[30:40, 35:85] = (255, 255, 255)

    repaired, mask, _ = LayoutPreservingRenderer(render_dpi=200)._inpaint_source_text(
        page,
        [segment],
        page_width=160,
        page_height=100,
        scale_x=1,
        scale_y=1,
    )

    assert np.any(mask > 0)
    assert np.array_equal(
        repaired[mask > 0], np.broadcast_to(blue, repaired[mask > 0].shape)
    )


def test_large_ocr_title_gets_wider_bounded_growth_margin():
    small = LayoutPreservingRenderer._ocr_growth_margin_pixels(
        font_size=8,
        scale_x=2.78,
        scale_y=2.78,
        render_dpi=200,
    )
    title = LayoutPreservingRenderer._ocr_growth_margin_pixels(
        font_size=90,
        scale_x=2.78,
        scale_y=2.78,
        render_dpi=200,
    )

    assert small >= 2
    assert title > small
    assert title <= round(8 * 2.78)


def test_large_display_title_gets_horizontal_erase_padding(tmp_path):
    task = _task(_source_pdf(tmp_path / "source.pdf"))
    segment = task.segments[0]
    segment.block_type = "title"
    segment.font_size = 90

    x_padding, y_padding = LayoutPreservingRenderer._erase_padding(
        segment,
        height=100,
    )

    assert x_padding == pytest.approx(9)
    assert y_padding == 14


def test_full_page_jpeg_is_extracted_before_poppler_render(tmp_path):
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (1500, 1000), (210, 225, 235)).save(
        image_path,
        format="JPEG",
        quality=88,
    )
    source_path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(source_path), pagesize=(300, 200))
    pdf.drawImage(str(image_path), 0, 0, width=300, height=200)
    pdf.save()
    document = fitz.open(source_path)
    pixmap = document[0].get_pixmap(dpi=150, alpha=False)
    rendered = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    raster_base = LayoutPreservingRenderer._write_raster_base_image(
        document[0],
        rendered,
        tmp_path,
        selected_index=1,
    )

    assert Path(raster_base.path).suffix.lower() in {".jpg", ".jpeg"}
    assert Path(raster_base.path).stat().st_size < 100_000
    document.close()


def test_repaired_raster_base_is_written_as_bounded_jpeg(tmp_path):
    source_path = tmp_path / "page.pdf"
    pdf = canvas.Canvas(str(source_path), pagesize=(300, 200))
    pdf.showPage()
    pdf.save()
    document = fitz.open(source_path)
    page_array = np.full((1000, 1500, 3), (225, 235, 242), dtype=np.uint8)
    page_array[300:700, 500:1000] = (70, 90, 110)

    raster_base = LayoutPreservingRenderer._write_repaired_raster_base_image(
        page_array,
        document[0],
        tmp_path,
        selected_index=1,
    )

    assert Path(raster_base.path).suffix.lower() == ".jpg"
    assert Path(raster_base.path).stat().st_size < 150_000
    assert raster_base.bbox == [0.0, 0.0, 300.0, 200.0]
    document.close()


def test_linear_background_histogram_matches_previous_exact_result():
    rng = np.random.default_rng(20260726)
    crop = rng.integers(0, 256, size=(117, 83, 3), dtype=np.uint8)
    pixels = crop.reshape(-1, 3).astype(np.int16)
    quantized = (pixels // 12).astype(np.int16)
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    candidates = pixels[np.max(np.abs(quantized - dominant), axis=1) <= 1]
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
    assert report["repair_image_layers"] == 0
    assert report["vector_text_redactions"] >= 4
    assert len(PdfReader(str(outputs.layout_pdf)).pages) == 4
