import io
from pathlib import Path

from pdf_translator.config import AppConfig
from pdf_translator.hybrid_app import _is_local_client, create_app


def test_local_client_detection():
    assert _is_local_client("127.0.0.1")
    assert _is_local_client("::1")
    assert not _is_local_client("192.168.1.20")
    assert not _is_local_client(None)


def test_access_mode_distinguishes_local_and_remote_clients(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)

    with app.test_client() as client:
        local = client.get(
            "/api/access-mode",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"},
        )
        remote = client.get(
            "/api/access-mode",
            environ_overrides={"REMOTE_ADDR": "192.168.1.20"},
        )

    assert local.get_json()["local"] is True
    assert remote.get_json()["local"] is False


def test_remote_client_cannot_trigger_server_native_picker(tmp_path, monkeypatch):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)
    called = False

    def picker():
        nonlocal called
        called = True
        return "/tmp/book.pdf"

    monkeypatch.setattr("pdf_translator.flask_app._choose_pdf_file", picker)

    with app.test_client() as client:
        response = client.post(
            "/api/select-pdf",
            environ_overrides={"REMOTE_ADDR": "192.168.1.20"},
        )

    assert response.status_code == 403
    assert "远程访问" in response.get_json()["error"]
    assert called is False


def test_remote_pdf_upload_is_saved_on_server(tmp_path):
    config = AppConfig.from_env(tmp_path / "data")
    app = create_app(config)
    app.config.update(TESTING=True)
    pdf_bytes = b"%PDF-1.4\n% minimal test upload\n%%EOF\n"

    with app.test_client() as client:
        response = client.post(
            "/api/upload-pdf",
            data={"pdf": (io.BytesIO(pdf_bytes), "My Book.pdf")},
            content_type="multipart/form-data",
            environ_overrides={"REMOTE_ADDR": "192.168.1.20"},
        )

    assert response.status_code == 200
    payload = response.get_json()
    destination = Path(payload["path"])
    assert destination.parent == config.uploads_dir.resolve()
    assert destination.read_bytes() == pdf_bytes
    assert payload["filename"] == "My Book.pdf"


def test_upload_rejects_non_pdf_content(tmp_path):
    app = create_app(AppConfig.from_env(tmp_path / "data"))
    app.config.update(TESTING=True)

    with app.test_client() as client:
        response = client.post(
            "/api/upload-pdf",
            data={"pdf": (io.BytesIO(b"not a pdf"), "fake.pdf")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    assert "有效的 PDF" in response.get_json()["error"]
