from pdf_translator.models import TaskSettings
from pdf_translator.pdf_parser import PDFParser, _RawBlock


def _span(text, bbox, size=8):
    return {
        "text": text,
        "bbox": bbox,
        "size": size,
        "font": "Helvetica",
    }


def test_table_row_block_is_split_into_visual_cells():
    block = {
        "type": 0,
        "bbox": [40, 100, 540, 112],
        "lines": [
            {
                "bbox": [40, 100, 72, 112],
                "spans": [_span("CS5702", [40, 100, 72, 112])],
            },
            {
                "bbox": [90, 100, 260, 112],
                "spans": [
                    _span(
                        "SOFTWARE ENGINEERING REQUIREMENTS",
                        [90, 100, 260, 112],
                    )
                ],
            },
            {
                "bbox": [400, 100, 412, 112],
                "spans": [_span("D1", [400, 100, 412, 112])],
            },
        ],
    }

    fragments = PDFParser.text_block_fragments(block)

    assert [item["text"] for item in fragments] == [
        "CS5702",
        "SOFTWARE ENGINEERING REQUIREMENTS",
        "D1",
    ]
    assert [item["bbox"] for item in fragments] == [
        [40.0, 100.0, 72.0, 112.0],
        [90.0, 100.0, 260.0, 112.0],
        [400.0, 100.0, 412.0, 112.0],
    ]


def test_wrapped_paragraph_remains_one_block():
    block = {
        "type": 0,
        "bbox": [40, 100, 300, 130],
        "lines": [
            {
                "bbox": [40, 100, 300, 112],
                "spans": [_span("First wrapped line", [40, 100, 300, 112])],
            },
            {
                "bbox": [40, 116, 220, 128],
                "spans": [_span("second wrapped line", [40, 116, 220, 128])],
            },
        ],
    }

    fragments = PDFParser.text_block_fragments(block)

    assert len(fragments) == 1
    assert fragments[0]["text"] == "First wrapped line second wrapped line"
    assert fragments[0]["bbox"] == [40.0, 100.0, 300.0, 130.0]


def test_rotated_block_keeps_visual_lines_independent_and_records_direction():
    block = {
        "type": 0,
        "bbox": [100, 200, 126, 500],
        "lines": [
            {
                "bbox": [100, 200, 110, 500],
                "dir": [0, -1],
                "spans": [_span("first vertical line", [100, 200, 110, 500])],
            },
            {
                "bbox": [116, 220, 126, 500],
                "dir": [0, -1],
                "spans": [_span("second vertical line", [116, 220, 126, 500])],
            },
        ],
    }

    fragments = PDFParser.text_block_fragments(block)

    assert [item["text"] for item in fragments] == [
        "first vertical line",
        "second vertical line",
    ]
    assert [item["rotation_degrees"] for item in fragments] == [-90.0, -90.0]


def test_distant_marker_and_multiline_title_are_separate_fragments():
    block = {
        "type": 0,
        "bbox": [40, 100, 540, 180],
        "lines": [
            {
                "bbox": [40, 100, 52, 114],
                "spans": [_span("4", [40, 100, 52, 114])],
            },
            {
                "bbox": [320, 116, 500, 134],
                "spans": [_span("THE FIRST READY MEAL", [320, 116, 500, 134])],
            },
            {
                "bbox": [320, 136, 540, 154],
                "spans": [_span("WAS CREATED IN 1953", [320, 136, 540, 154])],
            },
        ],
    }

    fragments = PDFParser.text_block_fragments(block)

    assert [item["text"] for item in fragments] == [
        "4",
        "THE FIRST READY MEAL WAS CREATED IN 1953",
    ]


def test_numeric_form_value_and_bottom_section_are_not_skipped_as_margins():
    blocks = [
        _RawBlock(
            page_number=1,
            order=1,
            text="13054201",
            bbox=[48, 340, 90, 352],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
        _RawBlock(
            page_number=1,
            order=2,
            text="6.2 Further information sources:",
            bbox=[318, 765, 455, 778],
            page_width=595,
            page_height=842,
            max_font_size=9,
            median_font_size=9,
            font_name="Helvetica",
        ),
        _RawBlock(
            page_number=1,
            order=3,
            text="1",
            bbox=[292, 822, 298, 834],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
    ]

    PDFParser._classify_page_blocks(blocks)

    assert blocks[0].block_type == "paragraph"
    assert blocks[1].block_type == "paragraph"
    assert blocks[2].block_type == "page_number"


def test_export_skips_numeric_and_short_decorative_capital_fragments():
    blocks = [
        _RawBlock(
            page_number=1,
            order=1,
            text="11/9-7112",
            bbox=[40, 100, 90, 110],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
        _RawBlock(
            page_number=1,
            order=2,
            text="Q",
            bbox=[40, 120, 50, 135],
            page_width=595,
            page_height=842,
            max_font_size=12,
            median_font_size=12,
            font_name="Helvetica",
            block_type="heading",
        ),
        _RawBlock(
            page_number=1,
            order=3,
            text="CTI",
            bbox=[40, 140, 62, 152],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
        _RawBlock(
            page_number=1,
            order=4,
            text="KXIXI",
            bbox=[40, 158, 72, 170],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
        _RawBlock(
            page_number=1,
            order=5,
            text="quartz 89",
            bbox=[40, 180, 100, 190],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
    ]

    segments = PDFParser._to_segments(blocks, "task-index-markers", TaskSettings())

    assert [segment.should_translate for segment in segments] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert [segment.row_key for segment in segments] == [
        "",
        "",
        "",
        "",
        "00000001",
    ]


def _raw(
    page: int,
    order: int,
    text: str,
    bbox: list[float],
    *,
    block_type: str = "paragraph",
    table: bool = False,
    rotation: float = 0.0,
) -> _RawBlock:
    return _RawBlock(
        page_number=page,
        order=order,
        text=text,
        bbox=bbox,
        page_width=600,
        page_height=800,
        max_font_size=10,
        median_font_size=10,
        font_name="Helvetica",
        block_type=block_type,
        is_table_cell=table,
        rotation_degrees=rotation,
    )


def test_nearby_visual_blocks_form_one_semantic_paragraph():
    groups = PDFParser._assemble_paragraph_groups(
        [
            _raw(1, 1, "A paragraph that", [40, 100, 260, 112]),
            _raw(1, 2, "continues on the next line.", [40, 116, 270, 128]),
        ]
    )

    assert len(groups) == 1
    assert PDFParser._merge_group_text(groups[0]) == (
        "A paragraph that continues on the next line."
    )


def test_nearby_rotated_lines_form_one_semantic_paragraph():
    groups = PDFParser._assemble_paragraph_groups(
        [
            _raw(
                1,
                1,
                "A vertical paragraph that",
                [100, 250, 110, 500],
                rotation=-90,
            ),
            _raw(
                1,
                2,
                "continues on its next line.",
                [115, 250, 125, 500],
                rotation=-90,
            ),
        ]
    )

    assert len(groups) == 1
    segments = PDFParser._to_segments(
        groups,
        "task-rotated-ir",
        TaskSettings(),
        stable_row_keys=True,
    )
    assert segments[0].rotation_degrees == -90
    assert all(anchor.rotation_degrees == -90 for anchor in segments[0].anchors)


def test_different_text_directions_never_merge_across_pages():
    groups = PDFParser._assemble_paragraph_groups(
        [
            _raw(
                1,
                1,
                "A paragraph continues",
                [40, 690, 270, 790],
            ),
            _raw(
                2,
                1,
                "into a rotated infographic label",
                [430, 40, 440, 180],
                rotation=-90,
            ),
        ]
    )

    assert len(groups) == 2


def test_cross_column_and_cross_page_continuations_keep_visual_anchors():
    blocks = [
        _raw(1, 1, "A long thought that", [40, 690, 270, 780]),
        _raw(1, 2, "continues in the right column", [330, 45, 560, 780]),
        _raw(1, 3, "running footer", [40, 785, 180, 798], block_type="footer"),
        _raw(2, 1, "running header", [40, 2, 180, 15], block_type="header"),
        _raw(2, 2, "and continues onto the following page.", [40, 55, 270, 130]),
    ]

    groups = PDFParser._assemble_paragraph_groups(blocks)
    paragraph = next(group for group in groups if len(group) == 3)
    segments = PDFParser._to_segments(
        groups,
        "task-ir-test",
        TaskSettings(),
        stable_row_keys=True,
    )

    assert [item.page_number for item in paragraph] == [1, 1, 2]
    semantic = next(segment for segment in segments if len(segment.anchors) == 3)
    assert semantic.row_key == semantic.segment_id
    assert semantic.continuation == "cross_page"
    assert [anchor.page_number for anchor in semantic.anchors] == [1, 1, 2]


def test_short_all_caps_label_does_not_merge_into_cross_column_body():
    groups = PDFParser._assemble_paragraph_groups(
        [
            _raw(1, 1, "FUEL TANK", [40, 700, 90, 780]),
            _raw(
                1,
                2,
                "A separate explanatory paragraph continues here.",
                [330, 45, 560, 130],
            ),
        ]
    )

    assert len(groups) == 2


def test_table_cells_remain_independent_semantic_segments():
    groups = PDFParser._assemble_paragraph_groups(
        [
            _raw(1, 1, "Course code", [40, 100, 150, 112], table=True),
            _raw(1, 2, "Course title", [160, 100, 300, 112], table=True),
        ]
    )

    assert len(groups) == 2
    assert all(len(group) == 1 for group in groups)
