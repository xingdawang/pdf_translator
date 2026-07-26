from __future__ import annotations

import json
import re
from pathlib import Path

from .config import AppConfig, ensure_app_directories
from .models import TranslationTask
from .utils import atomic_write_json


class TaskRepository:
    def __init__(self, config: AppConfig):
        self.config = config
        ensure_app_directories(config)

    def task_dir(self, task_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{5,80}", task_id):
            raise ValueError(f"任务 ID 格式不正确：{task_id}")
        return self.config.tasks_dir / task_id

    def manifest_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "task.json"

    def save(self, task: TranslationTask) -> None:
        directory = self.task_dir(task.task_id)
        for name in ("imports", "exports", "outputs", "reports", "tmp"):
            (directory / name).mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path(task.task_id), task.to_dict())

    def load(self, task_id: str) -> TranslationTask:
        path = self.manifest_path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"找不到任务：{task_id}")
        with path.open("r", encoding="utf-8") as stream:
            return TranslationTask.from_dict(json.load(stream))

    def list_tasks(self) -> list[TranslationTask]:
        tasks: list[TranslationTask] = []
        if not self.config.tasks_dir.exists():
            return tasks
        for path in self.config.tasks_dir.glob("*/task.json"):
            try:
                with path.open("r", encoding="utf-8") as stream:
                    tasks.append(TranslationTask.from_dict(json.load(stream)))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(tasks, key=lambda item: item.updated_at, reverse=True)
