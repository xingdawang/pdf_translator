from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Placeholder:
    token: str
    original: str
    kind: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Placeholder":
        return cls(**data)


@dataclass
class Segment:
    segment_id: str
    row_key: str
    page_number: int
    reading_order: int
    block_type: str
    source_text: str
    protected_text: str
    source_text_hash: str
    bbox: list[float]
    erase_bboxes: list[list[float]] = field(default_factory=list)
    font_name: str = ""
    font_size: float = 0.0
    should_translate: bool = True
    translated_text: str = ""
    placeholders: list[Placeholder] = field(default_factory=list)
    status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    fallback_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        payload = dict(data)
        payload["placeholders"] = [
            Placeholder.from_dict(item) for item in payload.get("placeholders", [])
        ]
        return cls(**payload)


@dataclass
class ExportPackage:
    package_id: str
    filename: str
    package_index: int
    package_total: int
    row_keys: list[str]
    created_at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportPackage":
        return cls(**data)


@dataclass
class TaskSettings:
    translate_headers: bool = False
    translate_footers: bool = False
    ignore_page_numbers: bool = True
    ignore_number_warnings: bool = True
    page_start: int = 1
    page_end: int | None = None
    protected_terms: list[str] = field(default_factory=list)
    ocr_mode: str = "off"
    ocr_dpi: int = 170

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskSettings":
        return cls(**(data or {}))


@dataclass
class TranslationTask:
    task_id: str
    source_path: str
    source_filename: str
    source_file_hash: str
    source_size_bytes: int
    created_at: str
    updated_at: str
    status: str
    page_count: int
    text_page_count: int
    scanned_page_count: int
    image_count: int
    parser_version: str
    settings: TaskSettings
    segments: list[Segment] = field(default_factory=list)
    export_packages: list[ExportPackage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    last_validation: dict[str, Any] | None = None
    review_confirmation: dict[str, Any] | None = None
    source_page_count: int | None = None
    last_generation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationTask":
        payload = dict(data)
        payload["settings"] = TaskSettings.from_dict(payload.get("settings"))
        payload["segments"] = [
            Segment.from_dict(item) for item in payload.get("segments", [])
        ]
        payload["export_packages"] = [
            ExportPackage.from_dict(item)
            for item in payload.get("export_packages", [])
        ]
        return cls(**payload)

    @property
    def translatable_segments(self) -> list[Segment]:
        return [segment for segment in self.segments if segment.should_translate]

    @property
    def translated_count(self) -> int:
        return sum(
            bool(segment.translated_text.strip())
            for segment in self.translatable_segments
        )

    @property
    def source_fallback_segments(self) -> list[Segment]:
        return [
            segment
            for segment in self.translatable_segments
            if segment.status == "source_fallback"
        ]

    @property
    def chinese_translation_count(self) -> int:
        return sum(
            bool(segment.translated_text.strip())
            and segment.status != "source_fallback"
            for segment in self.translatable_segments
        )

    @property
    def selected_page_numbers(self) -> list[int]:
        start = max(1, self.settings.page_start)
        end = self.settings.page_end
        if end is None:
            end = self.source_page_count or (start + self.page_count - 1)
        return list(range(start, end + 1))

    @property
    def page_range_label(self) -> str:
        pages = self.selected_page_numbers
        if not pages:
            return "未选择页面"
        source_total = self.source_page_count or self.page_count
        if len(pages) == source_total and pages[0] == 1:
            return f"全部 {source_total} 页"
        return f"第 {pages[0]}–{pages[-1]} 页，共 {len(pages)} 页"


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    segment_id: str | None = None
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationIssue":
        return cls(**data)


@dataclass
class ValidationReport:
    task_id: str
    created_at: str
    total_segments: int
    translated_segments: int
    blocking_errors: int
    warnings: int
    issues: list[ValidationIssue]
    can_generate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationReport":
        payload = dict(data)
        payload["issues"] = [
            ValidationIssue.from_dict(item) for item in payload.get("issues", [])
        ]
        return cls(**payload)
