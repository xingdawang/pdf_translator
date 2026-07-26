from __future__ import annotations

from .models import (
    TranslationTask,
    ValidationIssue,
    ValidationReport,
)
from .placeholders import (
    PlaceholderService,
    chinese_character_ratio,
    extract_numbers,
)
from .utils import normalize_whitespace, utc_now


class TranslationValidator:
    def __init__(self) -> None:
        self.placeholders = PlaceholderService()

    def validate(
        self,
        task: TranslationTask,
        extra_issues: list[ValidationIssue] | None = None,
    ) -> ValidationReport:
        issues = list(extra_issues or [])
        translated = 0

        for segment in task.translatable_segments:
            translation = segment.translated_text.strip()
            if segment.status == "source_fallback":
                translated += 1
                continue
            if not translation:
                segment.status = "empty"
                issues.append(
                    ValidationIssue(
                        severity="blocking",
                        code="EMPTY_TRANSLATION",
                        message="译文为空。",
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )
                continue

            translated += 1
            restore = self.placeholders.restore(translation, segment.placeholders)
            allowed_missing = self.placeholders.allowed_missing_tokens(
                restore,
                segment.placeholders,
                task.settings.ignore_number_warnings,
            )
            blocking_missing = self.placeholders.blocking_missing_tokens(
                restore,
                segment.placeholders,
                task.settings.ignore_number_warnings,
            )
            has_placeholder_blocker = bool(
                blocking_missing or restore.duplicated or restore.unknown
            )
            if allowed_missing:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="PLACEHOLDER_CHANGED_ALLOWED",
                        message=(
                            "Google 已改写受保护的数字、单位或误识别片段；"
                            "已按当前设置自动放行。"
                        ),
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )
            if blocking_missing:
                segment.status = "blocked"
                issues.append(
                    ValidationIssue(
                        severity="blocking",
                        code="MISSING_PLACEHOLDER",
                        message="缺少占位符：" + ", ".join(blocking_missing),
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )
            if restore.duplicated:
                segment.status = "blocked"
                issues.append(
                    ValidationIssue(
                        severity="blocking",
                        code="DUPLICATED_PLACEHOLDER",
                        message="占位符被重复使用：" + ", ".join(restore.duplicated),
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )
            if restore.unknown:
                segment.status = "blocked"
                issues.append(
                    ValidationIssue(
                        severity="blocking",
                        code="UNKNOWN_PLACEHOLDER",
                        message="发现未知占位符：" + ", ".join(restore.unknown),
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )

            restored = restore.restored_text.strip()
            if (
                not task.settings.ignore_number_warnings
                and extract_numbers(segment.source_text) != extract_numbers(restored)
            ):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="NUMBER_CHANGED",
                        message="原文和译文中的数字不一致，请人工确认。",
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )

            if normalize_whitespace(restored).casefold() == normalize_whitespace(
                segment.source_text
            ).casefold():
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="UNCHANGED_TEXT",
                        message="译文与原文完全相同。",
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )

            source_length = max(len(segment.source_text.strip()), 1)
            length_ratio = len(restored) / source_length
            if length_ratio > 4.0 or length_ratio < 0.08:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="ABNORMAL_LENGTH",
                        message=f"译文长度比例异常：{length_ratio:.2f}。",
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )

            if source_length >= 30 and chinese_character_ratio(restored) < 0.08:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="LOW_CHINESE_RATIO",
                        message="译文中中文字符比例较低，请确认是否已正确翻译。",
                        segment_id=segment.segment_id,
                        page_number=segment.page_number,
                    )
                )

            if not has_placeholder_blocker:
                segment.status = "ready"
                segment.fallback_reason = None

        blocking = sum(issue.severity == "blocking" for issue in issues)
        warning_count = sum(issue.severity == "warning" for issue in issues)
        return ValidationReport(
            task_id=task.task_id,
            created_at=utc_now(),
            total_segments=len(task.translatable_segments),
            translated_segments=translated,
            blocking_errors=blocking,
            warnings=warning_count,
            issues=issues,
            can_generate=blocking == 0
            and translated == len(task.translatable_segments),
        )
