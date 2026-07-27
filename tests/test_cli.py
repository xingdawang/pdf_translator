import pytest

from pdf_translator.cli import build_parser


def test_ocr_and_layout_defaults_are_both_170_dpi():
    parser = build_parser()

    analyze = parser.parse_args(["analyze", "/tmp/book.pdf"])
    generate = parser.parse_args(["generate", "task-id"])

    assert analyze.ocr_dpi == 170
    assert generate.layout_dpi == 170


def test_150_dpi_is_not_a_cli_option():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["analyze", "/tmp/book.pdf", "--ocr-dpi", "150"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["generate", "task-id", "--layout-dpi", "150"]
        )
