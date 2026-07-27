from __future__ import annotations

import argparse
import ipaddress
import os
import threading
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .config import AppConfig
from .flask_app import create_app as create_base_app
from .utils import file_size_label, safe_stem

DEFAULT_MAX_UPLOAD_MB = 1024


def _is_local_client(remote_addr: str | None) -> bool:
    """Return whether the HTTP client is running on the server machine."""

    if not remote_addr:
        return False
    address = remote_addr.split("%", 1)[0]
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return address.lower() == "localhost"


def create_app(config: AppConfig | None = None) -> Flask:
    app = create_base_app(config)
    workflow = app.extensions["translation_workflow"]
    max_upload_mb = max(
        1,
        int(os.getenv("PDF_TRANSLATOR_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)),
    )
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024

    @app.before_request
    def protect_native_picker_from_remote_clients():
        if request.path == "/api/select-pdf" and not _is_local_client(
            request.remote_addr
        ):
            return (
                jsonify(
                    {
                        "error": (
                            "远程访问不能打开服务器 Mac 的文件选择器。"
                            "请使用浏览器选择并上传 PDF。"
                        )
                    }
                ),
                403,
            )
        return None

    @app.get("/api/access-mode")
    def access_mode():
        return jsonify(
            {
                "local": _is_local_client(request.remote_addr),
                "max_upload_mb": max_upload_mb,
            }
        )

    @app.post("/api/upload-pdf")
    def upload_pdf():
        upload = request.files.get("pdf")
        if upload is None or not upload.filename:
            return jsonify({"error": "请选择需要上传的 PDF。"}), 400

        original_name = Path(upload.filename).name
        if Path(original_name).suffix.lower() != ".pdf":
            return jsonify({"error": "只支持上传 PDF 文件。"}), 400

        clean_name = secure_filename(original_name)
        stem = safe_stem(clean_name or "document.pdf")
        destination = workflow.config.uploads_dir / (
            f"remote_{uuid.uuid4().hex[:12]}_{stem}.pdf"
        )
        workflow.config.uploads_dir.mkdir(parents=True, exist_ok=True)

        try:
            upload.save(destination)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise ValueError("上传的 PDF 是空文件。")
            with destination.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise ValueError("文件内容不是有效的 PDF。")
        except Exception as exc:
            destination.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            {
                "path": str(destination.resolve()),
                "filename": original_name,
                "size_bytes": destination.stat().st_size,
                "size_label": file_size_label(destination.stat().st_size),
            }
        )

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return (
            jsonify(
                {
                    "error": (
                        f"上传文件超过 {max_upload_mb} MB 限制。"
                        "可以在服务器 Mac 本机使用路径模式，"
                        "或通过 PDF_TRANSLATOR_MAX_UPLOAD_MB 调高限制。"
                    )
                }
            ),
            413,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 PDF 英译中 Flask 界面")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5050, type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    app = create_app(AppConfig.from_env(args.data_dir))
    if not args.no_browser:
        browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        threading.Timer(
            0.9, lambda: webbrowser.open(f"http://{browser_host}:{args.port}")
        ).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
