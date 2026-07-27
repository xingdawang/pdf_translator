import json

from pdf_translator.config import AppConfig
from pdf_translator.models import (
    ExportPackage,
    Segment,
    TaskSettings,
    TranslationTask,
)
from pdf_translator.repository import TaskRepository
from pdf_translator.utils import sha256_text, utc_now


def _task(task_id: str = "20260727-storage") -> TranslationTask:
    now = utc_now()
    source = "A paragraph stored separately from task metadata."
    return TranslationTask(
        task_id=task_id,
        source_path="/tmp/source.pdf",
        source_filename="source.pdf",
        source_file_hash="a" * 64,
        source_size_bytes=1234,
        created_at=now,
        updated_at=now,
        status="analyzed",
        page_count=1,
        text_page_count=1,
        scanned_page_count=0,
        image_count=0,
        parser_version="test",
        settings=TaskSettings(ocr_dpi=170),
        segments=[
            Segment(
                segment_id="P0001-S000001",
                paragraph_id="P0001-S000001",
                row_key="P0001-S000001",
                page_number=1,
                reading_order=1,
                block_type="paragraph",
                source_text=source,
                protected_text=source,
                source_text_hash=sha256_text(source),
                bbox=[20, 30, 200, 60],
            )
        ],
    )


def test_sqlite_is_primary_store_and_json_is_small_summary(tmp_path):
    config = AppConfig.from_env(tmp_path / "data")
    repository = TaskRepository(config)
    task = _task()
    task.export_packages = [
        ExportPackage(
            package_id="PKG-storage-test",
            filename="translations.xlsx",
            package_index=1,
            package_total=1,
            row_keys=[f"P0001-S{index:06d}" for index in range(10_000)],
            created_at=utc_now(),
        )
    ]

    repository.save(task)

    summary_path = repository.manifest_path(task.task_id)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert config.database_path.is_file()
    assert summary["storage_backend"] == "sqlite"
    assert "segments" not in summary
    assert "row_keys" not in summary_path.read_text(encoding="utf-8")
    assert summary_path.stat().st_size < 5_000
    listed = repository.list_tasks()
    assert listed[0].segments == []
    assert listed[0].segment_count == 1
    loaded = repository.load(task.task_id)
    assert loaded.settings.ocr_dpi == 170
    assert loaded.segments[0].row_key == "P0001-S000001"
    assert len(loaded.export_packages[0].row_keys) == 10_000


def test_legacy_json_migration_keeps_recorded_dpi_and_backup(tmp_path):
    config = AppConfig.from_env(tmp_path / "data")
    task = _task("20260727-legacy")
    task.settings.ocr_dpi = 180
    task_dir = config.tasks_dir / task.task_id
    task_dir.mkdir(parents=True)
    manifest = task_dir / "task.json"
    manifest.write_text(
        json.dumps(task.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    repository = TaskRepository(config)
    migrated = repository.load(task.task_id)

    assert migrated.settings.ocr_dpi == 180
    assert repository.legacy_manifest_path(task.task_id).is_file()
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    assert summary["storage_backend"] == "sqlite"
    assert "segments" not in summary


def test_cleanup_preserves_current_output_and_delete_moves_to_trash(tmp_path):
    config = AppConfig.from_env(tmp_path / "data")
    repository = TaskRepository(config)
    task = _task("20260727-cleanup")
    repository.save(task)
    task_dir = repository.task_dir(task.task_id)
    current = task_dir / "outputs" / "current.pdf"
    stale = task_dir / "outputs" / "stale.pdf"
    cache = task_dir / "cache" / "page.bin"
    current.write_bytes(b"current")
    stale.write_bytes(b"stale")
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    task.outputs = {"layout_pdf": str(current)}
    repository.save(task)

    released = repository.cleanup_task_files(task.task_id)

    assert released > 0
    assert current.is_file()
    assert not stale.exists()
    assert not cache.exists()

    destination = repository.delete_task(task.task_id)
    assert destination.parent == config.trash_dir
    assert destination.is_dir()
    assert not task_dir.exists()
