from pathlib import Path
import shutil
import subprocess

import pytest
from openpyxl import load_workbook
from pypdf import PdfReader

from pdf_translator.config import AppConfig
from pdf_translator.models import TaskSettings
from pdf_translator.workflow import TranslationWorkflow
from scripts.sample_assets import create_sample_pdf, mock_translate_xlsx


pytest.importorskip("fitz")


def test_end_to_end_flow_layout_and_bilingual_output(tmp_path):
    source = create_sample_pdf(tmp_path / "sample.pdf")
    workflow = TranslationWorkflow(AppConfig.from_env(tmp_path / "data"))
    task = workflow.create_task(source)

    assert task.page_count == 2
    assert task.text_page_count == 2
    assert len(task.translatable_segments) > 5
    assert len({segment.segment_id for segment in task.segments}) == len(task.segments)

    task, package_paths = workflow.export_packages(task.task_id)
    returned = tmp_path / "translated.xlsx"
    mock_translate_xlsx(package_paths[0], returned)
    task, results, report = workflow.import_packages(task.task_id, [returned])
    assert results[0].exact_matches == len(task.translatable_segments)
    assert report.can_generate

    first_segment = task.translatable_segments[0]
    workflow.update_translations(
        task.task_id,
        {
            first_segment.segment_id: (
                "这是用于验证自动分页的长篇中文内容。" * 700
            )
        },
    )
    task, outputs = workflow.generate(
        task.task_id,
        chinese=True,
        bilingual=True,
        show_segment_ids=True,
    )

    chinese_reader = PdfReader(str(outputs.chinese_pdf))
    bilingual_reader = PdfReader(str(outputs.bilingual_pdf))
    assert len(chinese_reader.pages) >= 4
    assert len(bilingual_reader.pages) == task.page_count + outputs.translation_pages
    assert outputs.quality_json.exists()
    assert outputs.quality_html.exists()

    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        render_prefix = tmp_path / "rendered"
        subprocess.run(
            [
                pdftoppm,
                "-f",
                "1",
                "-singlefile",
                "-png",
                "-r",
                "90",
                str(outputs.chinese_pdf),
                str(render_prefix),
            ],
            check=True,
            capture_output=True,
        )
        assert render_prefix.with_suffix(".png").exists()


def test_quoted_path_and_page_range_generate_only_selected_pages(tmp_path):
    source = create_sample_pdf(tmp_path / "sample book.pdf")
    workflow = TranslationWorkflow(AppConfig.from_env(tmp_path / "data"))
    task = workflow.create_task(
        f"'{source}'",
        settings=TaskSettings(page_start=2, page_end=2),
    )

    assert task.source_page_count == 2
    assert task.page_count == 1
    assert task.selected_page_numbers == [2]
    assert task.page_range_label == "第 2–2 页，共 1 页"
    assert {segment.page_number for segment in task.segments} == {2}

    workflow.update_translations(
        task.task_id,
        {
            segment.segment_id: "中文译文：" + segment.protected_text
            for segment in task.translatable_segments
        },
    )
    _, outputs = workflow.generate(
        task.task_id,
        chinese=False,
        bilingual=False,
        layout=True,
        layout_dpi=150,
    )

    assert len(PdfReader(str(outputs.layout_pdf)).pages) == 1


def test_source_and_layout_output_interleaves_matching_pages_and_reuses_cache(
    tmp_path,
):
    source = create_sample_pdf(tmp_path / "sample.pdf")
    workflow = TranslationWorkflow(AppConfig.from_env(tmp_path / "data"))
    task = workflow.create_task(source)
    workflow.update_translations(
        task.task_id,
        {
            segment.segment_id: "中文译文：" + segment.protected_text
            for segment in task.translatable_segments
        },
    )

    task, outputs = workflow.generate(
        task.task_id,
        chinese=False,
        bilingual=False,
        layout=False,
        source_layout=True,
        layout_dpi=150,
    )

    assert outputs.source_layout_pdf is not None
    assert outputs.source_layout_pdf.exists()
    assert outputs.source_layout_pdf.name.endswith(
        "_原文+原版面中文版.pdf"
    )
    assert task.outputs == {
        "source_layout_pdf": str(outputs.source_layout_pdf)
    }
    assert list(
        workflow.repository.task_dir(task.task_id).joinpath("outputs").glob("*.pdf")
    ) == [outputs.source_layout_pdf]
    assert not {
        "quality_json",
        "quality_html",
        "layout_quality_json",
        "layout_quality_html",
    }.intersection(task.outputs)

    source_reader = PdfReader(str(source))
    combined_reader = PdfReader(str(outputs.source_layout_pdf))
    assert len(combined_reader.pages) == 4
    assert "A Practical Study" in combined_reader.pages[0].extract_text()
    assert "Two-column extraction example" in (
        combined_reader.pages[2].extract_text()
    )
    for source_index, combined_index in ((0, 0), (0, 1), (1, 2), (1, 3)):
        source_page = source_reader.pages[source_index]
        combined_page = combined_reader.pages[combined_index]
        assert float(combined_page.mediabox.width) == pytest.approx(
            float(source_page.mediabox.width)
        )
        assert float(combined_page.mediabox.height) == pytest.approx(
            float(source_page.mediabox.height)
        )

    first_mtime = outputs.source_layout_pdf.stat().st_mtime_ns
    _, cached = workflow.generate(
        task.task_id,
        chinese=False,
        bilingual=False,
        layout=False,
        source_layout=True,
        layout_dpi=150,
    )
    assert cached.source_layout_cache_hit
    assert cached.source_layout_pdf.stat().st_mtime_ns == first_mtime
