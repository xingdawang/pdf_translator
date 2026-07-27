from pathlib import Path

from pdf_translator.pdf_renderer import SYSTEM_CJK_FONT_PATHS


def test_system_font_candidates_include_ci_true_type_font():
    assert (
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
        in SYSTEM_CJK_FONT_PATHS
    )
