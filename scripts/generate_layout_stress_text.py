"""Generate non-semantic Chinese text for layout regression testing only.

This helper deliberately does not translate source text.  It derives the amount
of CJK text from each current task segment at runtime, preserves placeholders
and numbers, and then stores the result through the normal workflow.  It must
never be used to produce a deliverable translation.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

from pdf_translator.config import AppConfig
from pdf_translator.placeholders import extract_numbers
from pdf_translator.workflow import TranslationWorkflow


STRESS_MARKER = "排版校验"


def _counter_subtract(source: Counter[str], protected: Counter[str]) -> list[str]:
    remaining = source.copy()
    remaining.subtract(protected)
    values: list[str] = []
    for value, count in remaining.items():
        values.extend([value] * max(count, 0))
    return values


def make_stress_text(source_text: str, placeholder_tokens: list[str], ratio: float) -> str:
    """Return dynamically sized CJK layout-test text, not a translation."""
    # Page numbers and protected tokens are appended unchanged below, so only
    # alphabetic source content should determine the synthetic CJK amount.
    # Counting the whole source would double-count the many numbers on indexes.
    alphabetic_length = len(re.findall(r"[A-Za-z]", source_text))
    target_length = max(1, math.ceil(alphabetic_length * ratio))
    body = (STRESS_MARKER * math.ceil(target_length / len(STRESS_MARKER)))[
        :target_length
    ]
    suffix = [*placeholder_tokens]
    return " ".join([body, *suffix]).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="为已有任务生成非语义中文排版压力文本（仅用于测试）"
    )
    parser.add_argument("task_id")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--length-ratio", type=float, default=0.48)
    args = parser.parse_args()

    if not 0.1 <= args.length_ratio <= 1.5:
        parser.error("--length-ratio 必须在 0.1 到 1.5 之间")

    workflow = TranslationWorkflow(AppConfig.from_env(args.data_dir))
    task = workflow.repository.load(args.task_id)
    updates: dict[str, str] = {}
    for segment in task.translatable_segments:
        placeholder_tokens = [item.token for item in segment.placeholders]
        protected_numbers: Counter[str] = Counter()
        for placeholder in segment.placeholders:
            protected_numbers.update(extract_numbers(placeholder.original))
        unprotected_numbers = _counter_subtract(
            extract_numbers(segment.source_text),
            protected_numbers,
        )
        text = make_stress_text(
            segment.source_text,
            placeholder_tokens,
            args.length_ratio,
        )
        if unprotected_numbers:
            text = f"{text} {' '.join(unprotected_numbers)}"
        updates[segment.segment_id] = text

    _, report = workflow.update_translations(task.task_id, updates)
    print(
        f"任务 {task.task_id}：动态生成 {len(updates)} 段排版压力文本；"
        f"阻塞错误 {report.blocking_errors}，警告 {report.warnings}"
    )
    return 0 if report.can_generate else 2


if __name__ == "__main__":
    raise SystemExit(main())
