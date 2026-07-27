from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import AppConfig, ensure_app_directories
from .models import TranslationTask
from .utils import atomic_write_json, utc_now


_LOCKS_GUARD = threading.Lock()
_TASK_LOCKS: dict[tuple[str, str], threading.RLock] = {}
DATABASE_SCHEMA_VERSION = 1


class TaskRepository:
    """Persist task metadata and paragraphs in a local SQLite database.

    Each task directory still contains a small human-readable ``task.json``
    summary.  Legacy all-in-one manifests are backed up and migrated lazily,
    so existing tasks remain recoverable without keeping them on the hot path.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        ensure_app_directories(config)
        self._initialize_database()
        self._migrate_legacy_manifests()

    def task_dir(self, task_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{5,80}", task_id):
            raise ValueError(f"任务 ID 格式不正确：{task_id}")
        return self.config.tasks_dir / task_id

    def manifest_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "task.json"

    def legacy_manifest_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "task.legacy.json"

    @contextmanager
    def task_guard(self, task_id: str) -> Iterator[None]:
        key = (str(self.config.database_path), task_id)
        with _LOCKS_GUARD:
            lock = _TASK_LOCKS.setdefault(key, threading.RLock())
        with lock:
            yield

    def save(self, task: TranslationTask) -> None:
        with self.task_guard(task.task_id):
            directory = self.task_dir(task.task_id)
            for name in ("imports", "exports", "outputs", "reports", "tmp"):
                (directory / name).mkdir(parents=True, exist_ok=True)

            task.segment_count = len(task.segments)
            task.translated_segment_count = sum(
                bool(segment.translated_text.strip())
                for segment in task.segments
                if segment.should_translate
            )
            task.source_fallback_count = sum(
                segment.status == "source_fallback"
                for segment in task.segments
                if segment.should_translate
            )
            task.storage_size_bytes = self.task_storage_bytes(task.task_id)

            metadata = task.to_dict()
            segment_payloads = metadata.pop("segments")
            metadata_json = self._json_dump(metadata)

            with self._connect() as connection:
                existing = {
                    row["segment_id"]: (row["payload_hash"], row["position"])
                    for row in connection.execute(
                        """
                        SELECT segment_id, payload_hash, position
                        FROM segments
                        WHERE task_id = ?
                        """,
                        (task.task_id,),
                    )
                }
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_id, updated_at, status, source_filename,
                        page_count, source_size_bytes, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        status = excluded.status,
                        source_filename = excluded.source_filename,
                        page_count = excluded.page_count,
                        source_size_bytes = excluded.source_size_bytes,
                        payload_json = excluded.payload_json
                    """,
                    (
                        task.task_id,
                        task.updated_at,
                        task.status,
                        task.source_filename,
                        task.page_count,
                        task.source_size_bytes,
                        metadata_json,
                    ),
                )

                seen: set[str] = set()
                changed_rows: list[tuple[str, str, int, str, str]] = []
                for position, payload in enumerate(segment_payloads):
                    segment_id = str(payload["segment_id"])
                    payload_json = self._json_dump(payload)
                    payload_hash = hashlib.sha256(
                        payload_json.encode("utf-8")
                    ).hexdigest()
                    seen.add(segment_id)
                    if existing.get(segment_id) != (payload_hash, position):
                        changed_rows.append(
                            (
                                task.task_id,
                                segment_id,
                                position,
                                payload_json,
                                payload_hash,
                            )
                        )
                if changed_rows:
                    connection.executemany(
                        """
                        INSERT INTO segments (
                            task_id, segment_id, position, payload_json, payload_hash
                        )
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(task_id, segment_id) DO UPDATE SET
                            position = excluded.position,
                            payload_json = excluded.payload_json,
                            payload_hash = excluded.payload_hash
                        """,
                        changed_rows,
                    )
                removed = set(existing) - seen
                if removed:
                    connection.executemany(
                        "DELETE FROM segments WHERE task_id = ? AND segment_id = ?",
                        ((task.task_id, segment_id) for segment_id in removed),
                    )

            atomic_write_json(
                self.manifest_path(task.task_id),
                self._summary_payload(metadata),
            )

    def load(self, task_id: str) -> TranslationTask:
        self.task_dir(task_id)
        with self.task_guard(task_id):
            task = self._load_from_database(task_id)
            if task is not None:
                return task
            if self._migrate_manifest(task_id):
                migrated = self._load_from_database(task_id)
                if migrated is not None:
                    return migrated
        raise FileNotFoundError(f"找不到任务：{task_id}")

    def list_tasks(self) -> list[TranslationTask]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json
                FROM tasks
                ORDER BY updated_at DESC
                """
            ).fetchall()
        tasks: list[TranslationTask] = []
        for row in rows:
            try:
                tasks.append(TranslationTask.from_dict(json.loads(row["payload_json"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        for task in tasks:
            task.storage_size_bytes = self.task_storage_bytes(task.task_id)
        return tasks

    def task_storage_bytes(self, task_id: str) -> int:
        root = self.task_dir(task_id)
        if not root.exists():
            return 0
        total = 0
        for path in root.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def cleanup_task_files(self, task_id: str) -> int:
        """Remove reproducible temporary, cache and superseded output files."""

        with self.task_guard(task_id):
            task = self.load(task_id)
            root = self.task_dir(task_id)
            before = self.task_storage_bytes(task_id)

            tmp_dir = root / "tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            cache_dir = root / "cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

            for temporary in root.rglob("*.tmp"):
                if temporary.is_file() and not temporary.is_symlink():
                    temporary.unlink(missing_ok=True)

            referenced_outputs = {
                Path(path).resolve()
                for path in task.outputs.values()
                if path
            }
            output_dir = root / "outputs"
            if output_dir.exists():
                for path in output_dir.iterdir():
                    if (
                        path.is_file()
                        and path.resolve() not in referenced_outputs
                    ):
                        path.unlink(missing_ok=True)

            allowed_exports = {
                package.filename for package in task.export_packages
            }
            export_dir = root / "exports"
            if export_dir.exists():
                for path in export_dir.iterdir():
                    if (
                        path.is_file()
                        and path.suffix.lower() in {".xlsx", ".zip"}
                        and path.name not in allowed_exports
                    ):
                        path.unlink(missing_ok=True)

            after = self.task_storage_bytes(task_id)
            task.storage_size_bytes = after
            self.save(task)
            return max(0, before - after)

    def delete_task(self, task_id: str) -> Path:
        """Move a task directory to local trash, then remove its database rows."""

        with self.task_guard(task_id):
            task = self.load(task_id)
            source = self.task_dir(task_id)
            suffix = utc_now().replace(":", "").replace("+", "_")
            destination = self.config.trash_dir / f"{task.task_id}-{suffix}"
            if destination.exists():
                raise FileExistsError(f"回收站目标已存在：{destination}")
            shutil.move(str(source), str(destination))
            try:
                with self._connect() as connection:
                    connection.execute(
                        "DELETE FROM tasks WHERE task_id = ?",
                        (task_id,),
                    )
            except Exception:
                shutil.move(str(destination), str(source))
                raise
            return destination

    def _load_from_database(self, task_id: str) -> TranslationTask | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            segment_rows = connection.execute(
                """
                SELECT payload_json
                FROM segments
                WHERE task_id = ?
                ORDER BY position
                """,
                (task_id,),
            ).fetchall()
        payload = json.loads(row["payload_json"])
        payload["segments"] = [
            json.loads(segment_row["payload_json"])
            for segment_row in segment_rows
        ]
        task = TranslationTask.from_dict(payload)
        task.storage_size_bytes = self.task_storage_bytes(task_id)
        return task

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    source_size_bytes INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS segments (
                    task_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    PRIMARY KEY (task_id, segment_id),
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_segments_task_position
                ON segments(task_id, position);
                """
            )
            connection.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(DATABASE_SCHEMA_VERSION),),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.config.database_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def _migrate_legacy_manifests(self) -> None:
        for path in sorted(self.config.tasks_dir.glob("*/task.json")):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    payload = json.load(stream)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if payload.get("storage_backend") == "sqlite":
                continue
            if "segments" not in payload:
                continue
            self._migrate_manifest(path.parent.name, payload)

    def _migrate_manifest(
        self,
        task_id: str,
        supplied_payload: dict | None = None,
    ) -> bool:
        path = self.manifest_path(task_id)
        if supplied_payload is None:
            if not path.is_file():
                return False
            try:
                with path.open("r", encoding="utf-8") as stream:
                    supplied_payload = json.load(stream)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return False
        if "segments" not in supplied_payload:
            return False

        try:
            task = TranslationTask.from_dict(supplied_payload)
        except (TypeError, ValueError):
            return False

        backup = self.legacy_manifest_path(task_id)
        moved_original = False
        if path.exists() and not backup.exists():
            path.replace(backup)
            moved_original = True
        try:
            self.save(task)
        except Exception:
            if moved_original and backup.exists() and not path.exists():
                backup.replace(path)
            raise
        return True

    @staticmethod
    def _json_dump(payload: object) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _summary_payload(metadata: dict) -> dict:
        """Return a human-readable overview without large nested row lists."""

        summary = dict(metadata)
        packages = summary.pop("export_packages", [])
        validation = summary.pop("last_validation", None)
        document_ir = summary.pop("document_ir", None)
        summary["export_package_summaries"] = [
            {
                key: value
                for key, value in package.items()
                if key != "row_keys"
            }
            for package in packages
        ]
        if validation:
            summary["validation_summary"] = {
                key: value
                for key, value in validation.items()
                if key != "issues"
            }
        if document_ir:
            summary["document_ir_summary"] = {
                "version": document_ir.get("version"),
                "page_count": len(document_ir.get("pages", [])),
            }
        return {
            "storage_backend": "sqlite",
            "storage_version": DATABASE_SCHEMA_VERSION,
            **summary,
        }
