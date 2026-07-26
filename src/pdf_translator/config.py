from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    max_package_rows: int = 1_000_000
    max_package_characters: int = 100_000_000
    max_package_bytes: int = 9_500_000
    minimum_text_characters_per_page: int = 20
    minimum_text_page_ratio: float = 0.20
    render_dpi: int = 120

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None) -> "AppConfig":
        configured = data_dir or os.getenv("PDF_TRANSLATOR_DATA_DIR")
        base = Path(configured).expanduser() if configured else PROJECT_ROOT / "data"
        return cls(data_dir=base.resolve())

    @property
    def tasks_dir(self) -> Path:
        return self.data_dir / "tasks"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


def ensure_app_directories(config: AppConfig) -> None:
    config.tasks_dir.mkdir(parents=True, exist_ok=True)
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
