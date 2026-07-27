from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# OCR and layout repair deliberately share one quality scale.  Keep these
# values here so the CLI, web UI, task model, OCR engine and renderer cannot
# silently drift apart again.
DEFAULT_DPI = 200
DPI_PROFILES = {
    170: {
        "ocr_label": "更省资源",
        "layout_label": "轻量输出",
        "layout_note": "",
    },
    200: {
        "ocr_label": "推荐",
        "layout_label": "标准输出",
        "layout_note": "推荐",
    },
    240: {
        "ocr_label": "小字/复杂页面",
        "layout_label": "精细输出",
        "layout_note": "小字/复杂背景",
    },
}
DPI_CHOICES = tuple(DPI_PROFILES)


def parse_dpi(value: object, *, default: int = DEFAULT_DPI) -> int:
    """Return a supported DPI value for new user input.

    Historical tasks may contain older explicit values and remain readable.
    This helper is only used when accepting a new CLI or web selection.
    """

    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError:
            parsed = default
    return parsed if parsed in DPI_CHOICES else default


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    max_package_rows: int = 1_000_000
    max_package_characters: int = 100_000_000
    max_package_bytes: int = 9_500_000
    minimum_text_characters_per_page: int = 20
    minimum_text_page_ratio: float = 0.20
    render_dpi: int = DEFAULT_DPI

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

    @property
    def database_path(self) -> Path:
        return self.data_dir / "tasks.sqlite3"

    @property
    def trash_dir(self) -> Path:
        return self.data_dir / "trash"


def ensure_app_directories(config: AppConfig) -> None:
    config.tasks_dir.mkdir(parents=True, exist_ok=True)
    config.uploads_dir.mkdir(parents=True, exist_ok=True)
    config.trash_dir.mkdir(parents=True, exist_ok=True)
