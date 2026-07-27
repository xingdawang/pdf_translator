import io
import time
import zipfile

from openpyxl import load_workbook
from reportlab.pdfgen import canvas

from pdf_translator.config import AppConfig
from pdf_translator.flask_app import (
    BackgroundJob,
    _format_eta,
    _rounded_eta_seconds,
    create_app,
)
from scripts.sample_assets import create_sample_pdf, mock_translate_xlsx


def test_eta_is_displayed_in_ten_second_steps():
    assert _rounded_eta_seconds(0) == 0
    assert _rounded_eta_seconds(0.1) == 10
    assert _rounded_eta_seconds(10) == 10
    assert _rounded_eta_seconds(10.1) == 20
    assert _format_eta(70) == "1分10秒"
    assert _format_eta(60) == "1分钟"
    assert _format_eta(10) == "10秒"


def test_generate_job_payload_appends_dynamic_eta():
    job = BackgroundJob(
        job_id="eta-test",
        kind="generate",
        state="running",
        current=20,
        total=100,
        message="正在生成页面 20/100",
    )
    job._eta_deadline = time.monotonic() + 61

    payload = job.payload()

    assert payload["eta_seconds"] == 70
    assert payload["message"] == (
        "正在生成页面 20/100（预计剩余时间：1分10秒）"
    )
    assert "_eta_deadline" not in payload


def test_index_renders_lightweight_local_ui(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "本地 PDF 英译中" in response.get_data(as_text=True)
    assert "Vision OCR" in response.get_data(as_text=True)
    assert "streamlit" not in response.get_data(as_text=True).lower()
    assert "选择 PDF 文件" in response.get_data(as_text=True)
    assert "保留原版面，把英文 PDF 转成中文" in response.get_data(as_text=True)
    assert 'name="page_start"' in response.get_data(as_text=True)
    assert 'name="page_end"' in response.get_data(as_text=True)
    assert '<option value="vision" selected>' in response.get_data(as_text=True)
    assert "自动识别（推荐）" in response.get_data(as_text=True)
    assert '<option value="170" selected>170 DPI · 推荐</option>' in response.get_data(
        as_text=True
    )
    assert "150 DPI" not in response.get_data(as_text=True)
    assert "翻译范围与术语" not in response.get_data(as_text=True)
    assert 'name="protected_terms"' not in response.get_data(as_text=True)


def test_native_pdf_picker_returns_selected_path(tmp_path, monkeypatch):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    monkeypatch.setattr(
        "pdf_translator.flask_app._choose_pdf_file",
        lambda: "/Users/test/Downloads/book.pdf",
    )

    with app.test_client() as client:
        response = client.post("/api/select-pdf")

    assert response.status_code == 200
    assert response.get_json()["path"] == "/Users/test/Downloads/book.pdf"


def test_pdf_inspection_reports_text_layer_and_page_range(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    source = create_sample_pdf(tmp_path / "sample book.pdf")

    with app.test_client() as client:
        response = client.post(
            "/api/inspect-pdf",
            data={
                "source_path": f"'{source}'",
                "page_start": "2",
                "page_end": "2",
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["page_count"] == 2
    assert payload["page_start"] == 2
    assert payload["page_end"] == 2
    assert payload["sampled_pages"] == [2]
    assert payload["text_layer_pages"] == [2]
    assert payload["recommended_mode"] == "off"
    assert payload["status"] == "text"


def test_analyze_requires_local_path(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.post("/api/analyze", data={})

    assert response.status_code == 400
    assert "绝对路径" in response.get_json()["error"]


def test_analyze_rejects_reversed_page_range(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.post(
            "/api/analyze",
            data={
                "source_path": "/tmp/book.pdf",
                "page_start": "20",
                "page_end": "1",
            },
        )

    assert response.status_code == 400
    assert "结束页不能小于起始页" in response.get_json()["error"]


def test_text_only_analysis_failure_has_vision_retry_code(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    source = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(source))
    pdf.showPage()
    pdf.save()

    with app.test_client() as client:
        response = client.post(
            "/api/analyze",
            data={
                "source_path": str(source),
                "page_start": "1",
                "page_end": "1",
                "ocr_mode": "off",
            },
        )
        assert response.status_code == 202
        job_id = response.get_json()["job_id"]
        payload = {}
        for _ in range(100):
            payload = client.get(f"/api/jobs/{job_id}").get_json()
            if payload["state"] in {"completed", "failed"}:
                break
            time.sleep(0.01)

    assert payload["state"] == "failed"
    assert payload["error_code"] == "NO_TEXT_LAYER"
    assert "自动识别" in payload["error"]


def test_task_page_uses_simplified_four_step_workflow(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    workflow = app.extensions["translation_workflow"]
    task = workflow.create_task(create_sample_pdf(tmp_path / "sample.pdf"))

    with app.test_client() as client:
        content = client.get(f"/tasks/{task.task_id}").get_data(as_text=True)

    assert 'class="workflow-nav"' in content
    assert 'id="export"' in content
    assert 'id="google"' in content
    assert 'id="import"' in content
    assert 'id="generate"' in content
    assert "分析结果" not in content
    assert "导入、检查与修复" not in content
    assert "保存修改并重新检查" not in content
    assert "确认无误" not in content
    assert "选择 Google 下载的 XLSX" in content
    assert "正式输出 · 170 DPI（推荐）" in content
    assert "高清输出 · 200 DPI" in content
    assert "精细输出 · 240 DPI（小字/复杂背景）" in content
    assert "150 DPI" not in content
    assert "不影响 OCR 识别或中文矢量文字清晰度" in content
    assert "自动使用当前机器的全部" in content
    assert content.count('name="output_mode"') == 2
    assert 'value="layout" checked' in content
    assert 'value="source_layout"' in content
    assert "原版面中文版（推荐）" in content
    assert "原文 + 原版面中文版" in content
    assert "原第 1 页 → 中文第 1 页 → 原第 2 页 → 中文第 2 页" in content
    assert "重排中文阅读版" not in content
    assert "原文 + 重排双语版" not in content
    assert "显示 Segment ID" not in content
    assert "重排质量检查报告" not in content
    assert "重排质量数据 JSON" not in content
    assert "自定义中文字体" not in content
    assert "原版面替换质量报告" not in content
    assert "原版面质量数据 JSON" not in content


def test_generate_rejects_unknown_output_mode_and_reports_are_not_downloadable(
    tmp_path,
):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    workflow = app.extensions["translation_workflow"]
    task = workflow.create_task(create_sample_pdf(tmp_path / "sample.pdf"))

    with app.test_client() as client:
        response = client.post(
            f"/api/tasks/{task.task_id}/generate",
            data={"output_mode": "unknown"},
        )
        report_response = client.get(
            f"/tasks/{task.task_id}/outputs/quality_json"
        )

    assert response.status_code == 400
    assert "有效的 PDF 输出方式" in response.get_json()["error"]
    assert report_response.status_code == 404


def test_export_downloads_package_and_import_redirects_to_generate(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    workflow = app.extensions["translation_workflow"]
    task = workflow.create_task(create_sample_pdf(tmp_path / "sample.pdf"))

    with app.test_client() as client:
        export_response = client.post(f"/tasks/{task.task_id}/export")

    assert export_response.status_code == 200
    assert export_response.headers["Content-Disposition"].startswith("attachment;")
    assert export_response.headers["X-Package-Count"] == "1"

    task = workflow.repository.load(task.task_id)
    package_path = (
        workflow.repository.task_dir(task.task_id)
        / "exports"
        / task.export_packages[0].filename
    )
    translated = mock_translate_xlsx(
        package_path,
        tmp_path / "translated.xlsx",
    )
    with translated.open("rb") as stream, app.test_client() as client:
        import_response = client.post(
            f"/tasks/{task.task_id}/import",
            data={"translations": (stream, translated.name)},
            content_type="multipart/form-data",
        )

    assert import_response.status_code == 302
    assert import_response.headers["Location"].endswith("#generate")
    with app.test_client() as client:
        content = client.get(f"/tasks/{task.task_id}").get_data(as_text=True)
    assert "✓ 准备就绪" in content
    assert "准备就绪，可以生成最终 PDF" in content
    assert "人工确认" not in content


def test_missing_translation_is_listed_as_source_fallback(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    workflow = app.extensions["translation_workflow"]
    task = workflow.create_task(create_sample_pdf(tmp_path / "sample.pdf"))
    _, packages = workflow.export_packages(task.task_id)
    translated = mock_translate_xlsx(
        packages[0],
        tmp_path / "translated-missing-one.xlsx",
    )
    workbook = load_workbook(translated)
    workbook.worksheets[0].delete_rows(2)
    workbook.save(translated)

    with translated.open("rb") as stream, app.test_client() as client:
        response = client.post(
            f"/tasks/{task.task_id}/import",
            data={"translations": (stream, translated.name)},
            content_type="multipart/form-data",
        )
        page = client.get(f"/tasks/{task.task_id}")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("#generate")
    content = page.get_data(as_text=True)
    assert "查看保留英文内容（1 段）" in content
    assert "翻译件中缺少对应译文" in content
    assert "P0001-S000001" in content
    assert "准备就绪，可以生成最终 PDF" in content


def test_multiple_translation_packages_download_as_one_zip(tmp_path):
    config = AppConfig(
        data_dir=(tmp_path / "data").resolve(),
        max_package_rows=2,
    )
    app = create_app(config)
    app.config.update(TESTING=True)
    workflow = app.extensions["translation_workflow"]
    task = workflow.create_task(create_sample_pdf(tmp_path / "sample.pdf"))

    with app.test_client() as client:
        response = client.post(f"/tasks/{task.task_id}/export")

    assert response.status_code == 200
    assert response.headers["X-Package-Count"] != "1"
    assert response.headers["Content-Disposition"].endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(response.data)) as bundle:
        names = bundle.namelist()
    assert len(names) == int(response.headers["X-Package-Count"])
    assert all(name.endswith(".xlsx") for name in names)


def test_task_page_refreshes_saved_number_warnings(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    workflow = app.extensions["translation_workflow"]
    task = workflow.create_task(create_sample_pdf(tmp_path / "sample.pdf"))
    workflow.update_translations(
        task.task_id,
        {
            segment.segment_id: "中文译文：" + segment.protected_text
            for segment in task.translatable_segments
        },
    )
    task = workflow.repository.load(task.task_id)
    task.last_validation["warnings"] += 1
    task.last_validation["issues"].append(
        {
            "severity": "warning",
            "code": "NUMBER_CHANGED",
            "message": "历史数字警告",
            "segment_id": task.translatable_segments[0].segment_id,
            "page_number": 1,
        }
    )
    workflow.repository.save(task)

    with app.test_client() as client:
        response = client.get(f"/tasks/{task.task_id}")

    assert response.status_code == 200
    refreshed = workflow.repository.load(task.task_id)
    assert not any(
        issue["code"] == "NUMBER_CHANGED"
        for issue in refreshed.last_validation["issues"]
    )
    assert "NUMBER_CHANGED" not in response.get_data(as_text=True)
