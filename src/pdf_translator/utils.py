from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00ad", "")
    return re.sub(r"[ \t\r\f\v]+", " ", normalized).strip()


def normalize_local_path(value: str | Path) -> str:
    """Accept Finder/Terminal-style pasted paths, including shell quotes."""
    supplied = str(value).strip()
    if not supplied:
        return supplied
    if (
        len(supplied) >= 2
        and (supplied[0], supplied[-1])
        in {("'", "'"), ('"', '"'), ("‘", "’"), ("“", "”")}
    ):
        supplied = supplied[1:-1].strip()
    try:
        parts = shlex.split(supplied)
    except ValueError:
        parts = []
    if len(parts) == 1:
        supplied = parts[0]
    return supplied


def safe_stem(filename: str, max_length: int = 80) -> str:
    stem = Path(filename).stem
    stem = unicodedata.normalize("NFKC", stem)
    stem = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", stem, flags=re.UNICODE)
    stem = stem.strip("._") or "document"
    return stem[:max_length]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_size_label(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"
