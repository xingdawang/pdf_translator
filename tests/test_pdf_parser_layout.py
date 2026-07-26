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


def test_export_skips_numeric_and_single_letter_index_markers():
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
            text="quartz 89",
            bbox=[40, 145, 100, 155],
            page_width=595,
            page_height=842,
            max_font_size=8,
            median_font_size=8,
            font_name="Helvetica",
        ),
    ]

    segments = PDFParser._to_segments(
        blocks, "task-index-markers", TaskSettings()
    )

    assert [segment.should_translate for segment in segments] == [
        False,
        False,
        True,
    ]
    assert [segment.row_key for segment in segments] == ["", "", "00000001"]
