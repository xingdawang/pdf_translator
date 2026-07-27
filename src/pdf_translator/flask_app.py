from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename

from .config import (
    AppConfig,
    DEFAULT_DPI,
    DPI_CHOICES,
    DPI_PROFILES,
    parse_dpi,
)
from .exceptions import NoTextLayerError
from .models import TaskSettings
from .utils import file_size_label, normalize_local_path, safe_stem, utc_now
from .workflow import TranslationWorkflow


STATUS_LABELS = {
    "analyzed": "分析完成",
    "package_exported": "等待译文",
    "needs_review": "需要修复",
    "ready": "可以生成",
    "generated": "生成完成",
}
ETA_QUANTUM_SECONDS = 10
JOB_RETENTION_SECONDS = 24 * 60 * 60
logger = logging.getLogger(__name__)


def _rounded_eta_seconds(seconds: float) -> int:
    if seconds <= 0:
        return 0
    return max(
        ETA_QUANTUM_SECONDS,
        int(math.ceil(seconds / ETA_QUANTUM_SECONDS))
        * ETA_QUANTUM_SECONDS,
    )


def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "即将完成"
    minutes, remainder = divmod(seconds, 60)
    if minutes and remainder:
        return f"{minutes}分{remainder}秒"
    if minutes:
        return f"{minutes}分钟"
    return f"{remainder}秒"


@dataclass
class BackgroundJob:
    job_id: str
    kind: str
    state: str = "queued"
    current: int = 0
    total: int = 0
    message: str = "等待开始"
    task_id: str | None = None
    error: str | None = None
    error_code: str | None = None
    created_at: str = field(default_factory=utc_now)
    _eta_deadline: float | None = field(default=None, repr=False)
    _finished_monotonic: float | None = field(default=None, repr=False)
    _dedupe_key: str | None = field(default=None, repr=False)

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        eta_deadline = data.pop("_eta_deadline", None)
        data.pop("_finished_monotonic", None)
        data.pop("_dedupe_key", None)
        data["progress"] = (
            round(self.current / self.total * 100, 1) if self.total else 0
        )
        data["eta_seconds"] = None
        if (
            self.kind == "generate"
            and self.state == "running"
            and self.current > 0
            and self.total > 0
            and eta_deadline is not None
        ):
            eta_seconds = _rounded_eta_seconds(
                max(0.0, eta_deadline - time.monotonic())
            )
            data["eta_seconds"] = eta_seconds
            data["message"] = (
                f"{self.message}"
                f"（预计剩余时间：{_format_eta(eta_seconds)}）"
            )
        if self.task_id:
            anchor = "#export" if self.kind == "analyze" else "#generate"
            data["redirect_url"] = (
                url_for("task_detail", task_id=self.task_id) + anchor
            )
        return data


class JobManager:
    def __init__(self, workers: int = 2):
        self._jobs: dict[str, BackgroundJob] = {}
        self._active_keys: dict[str, str] = {}
        self._lock = threading.Lock()
        self._heavy_job_slot = threading.Semaphore(1)
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="pdf-translator"
        )

    def submit(
        self,
        kind: str,
        worker: Callable[[Callable[[int, int], None]], str],
        dedupe_key: str | None = None,
    ) -> BackgroundJob:
        active_key = f"{kind}:{dedupe_key}" if dedupe_key else None
        with self._lock:
            self._prune_locked()
            if active_key and active_key in self._active_keys:
                existing = self._jobs.get(self._active_keys[active_key])
                if existing and existing.state in {"queued", "running"}:
                    return existing
                self._active_keys.pop(active_key, None)
            job = BackgroundJob(
                job_id=uuid.uuid4().hex,
                kind=kind,
                _dedupe_key=active_key,
            )
            self._jobs[job.job_id] = job
            if active_key:
                self._active_keys[active_key] = job.job_id

        def run() -> None:
            opening_message = (
                "正在打开并检查 PDF" if kind == "analyze" else "正在准备生成"
            )

            eta_phase_started = time.monotonic()
            eta_last_current = 0
            eta_last_total = 0
            eta_deadline: float | None = None

            def progress(current: int, total: int) -> None:
                nonlocal eta_phase_started
                nonlocal eta_last_current
                nonlocal eta_last_total
                nonlocal eta_deadline
                now = time.monotonic()
                if (
                    eta_last_total
                    and (
                        total != eta_last_total
                        or current < eta_last_current
                    )
                ):
                    eta_phase_started = now
                    eta_deadline = None
                if kind == "generate" and current > 0 and total > 0:
                    elapsed = max(now - eta_phase_started, 0.001)
                    finalization_pages = min(
                        10.0,
                        max(1.0, total * 0.03),
                    )
                    raw_remaining = (
                        elapsed
                        / current
                        * (max(total - current, 0) + finalization_pages)
                    )
                    if eta_deadline is None:
                        smoothed_remaining = raw_remaining
                    else:
                        previous_remaining = max(0.0, eta_deadline - now)
                        smoothed_remaining = (
                            previous_remaining * 0.65
                            + raw_remaining * 0.35
                        )
                    eta_deadline = now + smoothed_remaining
                eta_last_current = current
                eta_last_total = total
                progress_message = (
                    f"正在检查并识别页面 {current}/{total}"
                    if kind == "analyze"
                    else f"正在生成页面 {current}/{total}"
                )
                self._update(
                    job.job_id,
                    current=current,
                    total=total,
                    message=progress_message,
                    _eta_deadline=eta_deadline,
                )

            try:
                with self._heavy_job_slot:
                    eta_phase_started = time.monotonic()
                    self._update(
                        job.job_id,
                        state="running",
                        message=opening_message,
                    )
                    task_id = worker(progress)
                self._update(
                    job.job_id,
                    state="completed",
                    task_id=task_id,
                    message="处理完成",
                )
            except Exception as exc:
                logger.exception("后台 %s 任务失败", kind)
                self._update(
                    job.job_id,
                    state="failed",
                    error=str(exc),
                    error_code=(
                        "NO_TEXT_LAYER"
                        if isinstance(exc, NoTextLayerError)
                        else None
                    ),
                    message="处理失败",
                )
            finally:
                with self._lock:
                    completed_job = self._jobs.get(job.job_id)
                    if completed_job is not None:
                        completed_job._finished_monotonic = time.monotonic()
                    if (
                        active_key
                        and self._active_keys.get(active_key) == job.job_id
                    ):
                        self._active_keys.pop(active_key, None)

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            self._prune_locked()
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in values.items():
                setattr(job, key, value)

    def _prune_locked(self) -> None:
        cutoff = time.monotonic() - JOB_RETENTION_SECONDS
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job._finished_monotonic is not None
            and job._finished_monotonic < cutoff
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)


def create_app(config: AppConfig | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.update(
        SECRET_KEY=os.getenv("PDF_TRANSLATOR_SECRET", uuid.uuid4().hex),
        MAX_CONTENT_LENGTH=128 * 1024 * 1024,
    )
    workflow = TranslationWorkflow(config or AppConfig.from_env())
    jobs = JobManager()
    app.extensions["translation_workflow"] = workflow
    app.extensions["translation_jobs"] = jobs

    @app.template_filter("filesize")
    def filesize_filter(value: int) -> str:
        return file_size_label(value)

    @app.context_processor
    def common_values() -> dict[str, Any]:
        return {
            "status_labels": STATUS_LABELS,
            "default_dpi": DEFAULT_DPI,
            "dpi_choices": DPI_CHOICES,
            "dpi_profiles": DPI_PROFILES,
        }

    @app.get("/")
    def index():
        all_tasks = workflow.repository.list_tasks()
        history_page_size = 5
        history_page_count = max(
            1,
            (len(all_tasks) + history_page_size - 1) // history_page_size,
        )
        requested_history_page = request.args.get(
            "history_page",
            default=1,
            type=int,
        )
        history_page = min(
            max(requested_history_page or 1, 1),
            history_page_count,
        )
        history_start = (history_page - 1) * history_page_size
        return render_template(
            "index.html",
            tasks=all_tasks[
                history_start : history_start + history_page_size
            ],
            task_count=len(all_tasks),
            history_page=history_page,
            history_page_count=history_page_count,
            data_dir=workflow.config.data_dir,
        )

    @app.post("/api/analyze")
    def analyze():
        source_path = normalize_local_path(
            request.form.get("source_path", "")
        )
        if not source_path:
            return jsonify({"error": "请填写本地 PDF 的绝对路径。"}), 400
        page_start = _bounded_int(
            request.form.get("page_start"),
            default=1,
            minimum=1,
            maximum=1_000_000,
        )
        try:
            page_end = _optional_positive_int(request.form.get("page_end"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if page_end is not None and page_end < page_start:
            return jsonify({"error": "结束页不能小于起始页。"}), 400
        settings = TaskSettings(
            translate_headers=request.form.get("translate_headers") == "on",
            translate_footers=request.form.get("translate_footers") == "on",
            ignore_page_numbers=request.form.get("translate_page_numbers") != "on",
            page_start=page_start,
            page_end=page_end,
            protected_terms=[
                line.strip()
                for line in request.form.get("protected_terms", "").splitlines()
                if line.strip()
            ],
            ocr_mode=request.form.get("ocr_mode", "vision"),
            ocr_dpi=parse_dpi(request.form.get("ocr_dpi")),
        )

        def worker(progress):
            task = workflow.create_task(
                source_path,
                settings=settings,
                copy_source=False,
                progress=progress,
            )
            return task.task_id

        analyze_key = "|".join(
            (
                source_path,
                str(page_start),
                str(page_end or ""),
                settings.ocr_mode,
                str(settings.ocr_dpi),
            )
        )
        return jsonify(
            jobs.submit("analyze", worker, dedupe_key=analyze_key).payload()
        ), 202

    @app.post("/api/inspect-pdf")
    def inspect_pdf():
        source_path = normalize_local_path(
            request.form.get("source_path", "")
        )
        if not source_path:
            return jsonify({"error": "请先选择或填写本地 PDF 路径。"}), 400
        page_start = _bounded_int(
            request.form.get("page_start"),
            default=1,
            minimum=1,
            maximum=1_000_000,
        )
        try:
            page_end = _optional_positive_int(request.form.get("page_end"))
            inspection = workflow.inspect_source(
                source_path,
                page_start=page_start,
                page_end=page_end,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        payload = inspection.to_dict()
        payload["size_label"] = file_size_label(inspection.size_bytes)
        return jsonify(payload)

    @app.post("/api/select-pdf")
    def select_pdf():
        try:
            selected = _choose_pdf_file()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 501
        except Exception as exc:
            return jsonify({"error": f"无法打开文件选择器：{exc}"}), 500
        return jsonify({"path": selected, "cancelled": selected is None})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"error": "找不到后台任务。"}), 404
        return jsonify(job.payload())

    @app.get("/tasks/<task_id>")
    def task_detail(task_id: str):
        task = workflow.repository.load(task_id)
        if (
            task.last_validation
            and (
                task.last_validation.get("blocking_errors", 0)
                or (
                    task.settings.ignore_number_warnings
                    and any(
                        issue.get("code") == "NUMBER_CHANGED"
                        for issue in task.last_validation.get("issues", [])
                    )
                )
            )
        ):
            task, refreshed_report = workflow.validate(task_id)
            report = refreshed_report.to_dict()
        else:
            report = task.last_validation
        report = report or {
            "total_segments": len(task.translatable_segments),
            "translated_segments": task.translated_count,
            "blocking_errors": len(task.translatable_segments) - task.translated_count,
            "warnings": 0,
            "issues": [],
            "can_generate": False,
        }
        return render_template(
            "task.html",
            task=task,
            report=report,
            task_dir=workflow.repository.task_dir(task_id),
            render_cores=os.cpu_count() or 1,
        )

    @app.get("/tasks/<task_id>/structure")
    def task_structure(task_id: str):
        task = workflow.repository.load(task_id)
        available_pages = task.selected_page_numbers
        if not available_pages:
            abort(404)
        page_number = _bounded_int(
            request.args.get("page"),
            default=available_pages[0],
            minimum=available_pages[0],
            maximum=available_pages[-1],
        )
        if page_number not in available_pages:
            abort(404)

        page_ir = next(
            (
                page
                for page in (task.document_ir.pages if task.document_ir else [])
                if page.page_number == page_number
            ),
            None,
        )
        if page_ir is None:
            try:
                import fitz

                with fitz.open(task.source_path) as document:
                    page = document.load_page(page_number - 1)
                    page_width = float(page.rect.width)
                    page_height = float(page.rect.height)
            except Exception:
                abort(404)
        else:
            page_width = page_ir.width
            page_height = page_ir.height

        anchors: list[dict[str, Any]] = []
        for segment in task.segments:
            for anchor in segment.visual_anchors:
                if anchor.page_number != page_number:
                    continue
                x0, y0, x1, y1 = anchor.bbox
                anchors.append(
                    {
                        "anchor_id": anchor.anchor_id,
                        "paragraph_id": segment.paragraph_id
                        or segment.segment_id,
                        "reading_order": anchor.reading_order,
                        "layout_label": anchor.layout_label,
                        "column_id": anchor.column_id,
                        "continuation": segment.continuation,
                        "source_text": anchor.source_text,
                        "x": x0,
                        "y": y0,
                        "width": max(0.5, x1 - x0),
                        "height": max(0.5, y1 - y0),
                    }
                )
        anchors.sort(key=lambda item: (item["reading_order"], item["y"], item["x"]))
        page_index = available_pages.index(page_number)
        return render_template(
            "structure.html",
            task=task,
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            anchors=anchors,
            previous_page=(
                available_pages[page_index - 1] if page_index > 0 else None
            ),
            next_page=(
                available_pages[page_index + 1]
                if page_index + 1 < len(available_pages)
                else None
            ),
        )

    @app.get("/tasks/<task_id>/structure/pages/<int:page_number>.png")
    def task_structure_page(task_id: str, page_number: int):
        task = workflow.repository.load(task_id)
        if page_number not in task.selected_page_numbers:
            abort(404)
        try:
            import fitz

            with fitz.open(task.source_path) as document:
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(1.5, 1.5),
                    alpha=False,
                )
                image = BytesIO(pixmap.tobytes("png"))
        except Exception:
            abort(404)
        return send_file(
            image,
            mimetype="image/png",
            download_name=f"page-{page_number}.png",
            max_age=300,
        )

    @app.post("/tasks/<task_id>/cleanup")
    def cleanup_task(task_id: str):
        try:
            released = workflow.cleanup_task(task_id)
            flash(
                f"已清理临时文件、缓存和旧输出，释放 {file_size_label(released)}。",
                "success",
            )
        except Exception as exc:
            flash(f"清理失败：{exc}", "error")
        return redirect(url_for("task_detail", task_id=task_id))

    @app.post("/tasks/<task_id>/delete")
    def delete_task(task_id: str):
        try:
            destination = workflow.delete_task(task_id)
            flash(f"任务已移入本地回收目录：{destination}", "success")
        except Exception as exc:
            flash(f"删除任务失败：{exc}", "error")
            return redirect(url_for("task_detail", task_id=task_id))
        return redirect(url_for("index"))

    @app.post("/tasks/<task_id>/export")
    def export_packages(task_id: str):
        try:
            task, paths = workflow.export_packages(task_id)
            if len(paths) == 1:
                download_path = paths[0]
            else:
                download_path = (
                    workflow.repository.task_dir(task_id)
                    / "exports"
                    / f"{safe_stem(task.source_filename)}_google_translate_packages.zip"
                )
                with zipfile.ZipFile(
                    download_path, "w", compression=zipfile.ZIP_DEFLATED
                ) as bundle:
                    for path in paths:
                        bundle.write(path, arcname=path.name)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        response = send_file(
            download_path,
            as_attachment=True,
            download_name=download_path.name,
        )
        response.headers["X-Package-Count"] = str(len(paths))
        return response

    @app.get("/tasks/<task_id>/exports/<filename>")
    def download_export(task_id: str, filename: str):
        task = workflow.repository.load(task_id)
        allowed = {package.filename for package in task.export_packages}
        if filename not in allowed:
            abort(404)
        path = workflow.repository.task_dir(task_id) / "exports" / filename
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.post("/tasks/<task_id>/import")
    def import_packages(task_id: str):
        uploads = request.files.getlist("translations")
        if not uploads or not any(item.filename for item in uploads):
            flash("请选择至少一个翻译后的 XLSX。", "error")
            return redirect(url_for("task_detail", task_id=task_id) + "#import")
        import_dir = workflow.repository.task_dir(task_id) / "imports"
        import_dir.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        anchor = "#import"
        try:
            for upload in uploads:
                if not upload.filename:
                    continue
                if Path(upload.filename).suffix.lower() != ".xlsx":
                    raise ValueError(f"只支持 XLSX：{upload.filename}")
                clean = secure_filename(upload.filename)
                filename = f"staged_{uuid.uuid4().hex[:8]}_{safe_stem(clean)}.xlsx"
                destination = import_dir / filename
                upload.save(destination)
                staged.append(destination)
            task, results, report = workflow.import_packages(task_id, staged)
            imported = sum(item.imported_segments for item in results)
            if report.can_generate:
                anchor = "#generate"
                flash(
                    f"已导入 {imported} 段；"
                    f"{task.chinese_translation_count} 段使用译文，"
                    f"{len(task.source_fallback_segments)} 段保留英文。准备就绪。",
                    "success",
                )
            else:
                flash(
                    "翻译件无法安全对应当前任务："
                    f"仍有 {report.blocking_errors} 个技术错误。",
                    "error",
                )
        except Exception as exc:
            flash(str(exc), "error")
        return redirect(url_for("task_detail", task_id=task_id) + anchor)

    @app.post("/tasks/<task_id>/translations")
    def update_translations(task_id: str):
        intent = request.form.get("intent", "save")
        updates = {
            key.removeprefix("translation__"): value
            for key, value in request.form.items()
            if key.startswith("translation__")
        }
        next_anchor = "#review-feedback"
        try:
            _, report = workflow.update_translations(task_id, updates)
            if intent == "confirm":
                if report.can_generate:
                    _, report = workflow.confirm_review(task_id)
                    next_anchor = "#generate"
                    flash(
                        "当前修改已保存，并已确认全部译文无误。",
                        "success",
                    )
                else:
                    flash(
                        "修改已保存，但仍有 "
                        f"{report.blocking_errors} 个阻断问题，暂时不能确认无误。",
                        "error",
                    )
            else:
                message = (
                    "修改已保存，检查通过，可以进入生成步骤。"
                    if report.can_generate
                    else f"修改已保存，仍有 {report.blocking_errors} 个阻断问题。"
                )
                if report.can_generate:
                    next_anchor = "#generate"
                flash(message, "success" if report.can_generate else "error")
        except Exception as exc:
            flash(str(exc), "error")
        return redirect(url_for("task_detail", task_id=task_id) + next_anchor)

    @app.post("/api/tasks/<task_id>/generate")
    def generate(task_id: str):
        output_mode = request.form.get("output_mode", "layout")
        layout_dpi = parse_dpi(request.form.get("layout_dpi"))
        if output_mode not in {"layout", "source_layout"}:
            return jsonify({"error": "请选择有效的 PDF 输出方式。"}), 400

        def worker(progress):
            workflow.generate(
                task_id,
                chinese=False,
                bilingual=False,
                layout=output_mode == "layout",
                source_layout=output_mode == "source_layout",
                show_segment_ids=False,
                font_path=None,
                layout_dpi=layout_dpi,
                progress=progress,
            )
            return task_id

        return jsonify(
            jobs.submit("generate", worker, dedupe_key=task_id).payload()
        ), 202

    @app.get("/tasks/<task_id>/outputs/<kind>")
    def download_output(task_id: str, kind: str):
        task = workflow.repository.load(task_id)
        allowed = {
            "layout_pdf",
            "source_layout_pdf",
        }
        if kind not in allowed or kind not in task.outputs:
            abort(404)
        path = Path(task.outputs[kind]).resolve()
        task_root = workflow.repository.task_dir(task_id).resolve()
        if task_root not in path.parents or not path.is_file():
            abort(404)
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(ValueError)
    def friendly_error(error):
        return render_template("error.html", message=str(error)), 404

    return app


def _bounded_int(
    value: str | None, default: int, minimum: int, maximum: int
) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _optional_positive_int(value: str | None) -> int | None:
    supplied = (value or "").strip()
    if not supplied:
        return None
    try:
        parsed = int(supplied)
    except (TypeError, ValueError) as exc:
        raise ValueError("页码必须是整数。") from exc
    if parsed < 1:
        raise ValueError("页码必须大于或等于 1。")
    return parsed


def _choose_pdf_file() -> str | None:
    if sys.platform != "darwin":
        raise RuntimeError("当前系统不支持原生文件选择器，请直接粘贴 PDF 路径。")
    script = """
try
  set selectedFile to choose file with prompt "选择需要翻译的 PDF" of type {"com.adobe.pdf"}
  return POSIX path of selectedFile
on error number -128
  return ""
end try
"""
    result = subprocess.run(
        ["osascript", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = normalize_local_path(result.stdout)
    return selected or None


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 PDF 英译中 Flask 界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5050, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    app = create_app(AppConfig.from_env(args.data_dir))
    if not args.no_browser:
        threading.Timer(
            0.9, lambda: webbrowser.open(f"http://{args.host}:{args.port}")
        ).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
