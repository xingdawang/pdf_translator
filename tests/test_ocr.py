from pdf_translator.ocr import MacVisionOCR


def test_vision_payload_maps_to_pdf_coordinates_and_groups_lines():
    payload = [
        {
            "text": "First line",
            "confidence": 0.99,
            "x": 0.1,
            "y": 0.8,
            "width": 0.3,
            "height": 0.03,
        },
        {
            "text": "second line",
            "confidence": 0.98,
            "x": 0.1,
            "y": 0.75,
            "width": 0.32,
            "height": 0.03,
        },
    ]

    lines = MacVisionOCR._lines_from_payload(payload, 600, 800)
    blocks = MacVisionOCR._group_lines(lines, 600)

    assert lines[0].bbox == [60.0, 136.0, 240.0, 160.0]
    assert len(blocks) == 1
    assert blocks[0].text == "First line second line"


def test_vision_rejects_low_confidence_image_noise():
    payload = [
        {
            "text": "decorative image noise",
            "confidence": 0.34,
            "x": 0.1,
            "y": 0.8,
            "width": 0.2,
            "height": 0.01,
        },
        {
            "text": "real caption",
            "confidence": 0.35,
            "x": 0.1,
            "y": 0.7,
            "width": 0.1,
            "height": 0.01,
        },
    ]

    lines = MacVisionOCR._lines_from_payload(payload, 600, 800)

    assert [line.text for line in lines] == ["real caption"]


def test_heading_is_not_merged_into_following_body():
    payload = [
        {
            "text": "ANCIENT USES",
            "confidence": 1,
            "x": 0.1,
            "y": 0.8,
            "width": 0.2,
            "height": 0.02,
        },
        {
            "text": "Malachite has a long history.",
            "confidence": 1,
            "x": 0.1,
            "y": 0.77,
            "width": 0.35,
            "height": 0.02,
        },
    ]
    lines = MacVisionOCR._lines_from_payload(payload, 600, 800)
    blocks = MacVisionOCR._group_lines(lines, 600)

    assert [block.text for block in blocks] == [
        "ANCIENT USES",
        "Malachite has a long history.",
    ]
    assert blocks[0].block_type == "heading"


def test_vision_payload_uses_embedded_image_placement_rect():
    payload = [
        {
            "text": "Placed scan line",
            "confidence": 0.99,
            "x": 0.1,
            "y": 0.8,
            "width": 0.3,
            "height": 0.03,
        }
    ]

    lines = MacVisionOCR._lines_from_payload(
        payload,
        page_width=600,
        page_height=800,
        source_rect=[0, 9, 600, 709],
    )

    assert lines[0].bbox == [60.0, 128.0, 240.0, 149.0]
