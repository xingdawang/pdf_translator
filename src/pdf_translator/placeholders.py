from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256

from .models import Placeholder


TOKEN_PATTERN = re.compile(r"⟦X[0-9A-F]{10}⟧")
BARE_TOKEN_PATTERN = re.compile(r"X[0-9A-F]{10}", re.IGNORECASE)


@dataclass
class PlaceholderResult:
    protected_text: str
    placeholders: list[Placeholder]


@dataclass
class RestoreResult:
    restored_text: str
    missing: list[str]
    duplicated: list[str]
    unknown: list[str]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicated or self.unknown)


class PlaceholderService:
    _patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "code",
            re.compile(r"```[\s\S]*?```|`[^`\n]+`", re.MULTILINE),
        ),
        (
            "formula",
            re.compile(
                r"\$\$[\s\S]+?\$\$|(?<!\$)\$[^$\n]{1,200}\$(?!\$)|"
                r"\\\([^)\n]{1,200}\\\)|\\\[[^\]\n]{1,400}\\\]|"
                r"(?<!\w)[A-Za-z]\s*=\s*[A-Za-z0-9]+"
                r"(?:\s*[\^_*/+\-]\s*[A-Za-z0-9²³⁰¹⁴⁵⁶⁷⁸⁹]+)*"
            ),
        ),
        (
            "url",
            re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>{}\[\]]+"),
        ),
        (
            "email",
            re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        ),
        (
            "doi",
            re.compile(r"(?i)\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b"),
        ),
        (
            "file_path",
            re.compile(
                r"(?<!\w)(?:[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+|"
                r"/(?=[^/\s]*[A-Za-z_~])(?:[^/\s]+/)*[^/\s]+)"
            ),
        ),
        (
            "citation",
            re.compile(
                r"\[(?:\s*\d+(?:\s*[-–]\s*\d+)?\s*,?)+\]|"
                r"\((?:19|20)\d{2}[a-z]?\)"
            ),
        ),
        (
            "number_unit",
            re.compile(
                r"(?<![\w.])(?!(?:19|20)\d{2}s\b)"
                r"[-+]?\d+(?:[.,]\d+)?(?:\s*[×x]\s*10"
                r"(?:\^[-+]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+))?\s*"
                r"(?:%|°?[CF]|kg|g|mg|μg|ug|ng|km|cm|mm|μm|um|nm|"
                r"mL|ml|L|kPa|MPa|Pa|Hz|kHz|MHz|GHz|V|mV|A|mA|"
                r"W|kW|MW|J|kJ|mol|mmol|s|ms|min|h|d)(?!\w)",
                re.IGNORECASE,
            ),
        ),
    )

    def protect(
        self,
        text: str,
        task_id: str,
        segment_id: str,
        protected_terms: list[str] | None = None,
    ) -> PlaceholderResult:
        candidates: list[tuple[int, int, int, str, str]] = []
        for priority, (kind, pattern) in enumerate(self._patterns):
            for match in pattern.finditer(text):
                candidates.append(
                    (match.start(), match.end(), priority, kind, match.group(0))
                )

        for term in sorted(protected_terms or [], key=len, reverse=True):
            cleaned = term.strip()
            if not cleaned:
                continue
            pattern = re.compile(re.escape(cleaned), re.IGNORECASE)
            for match in pattern.finditer(text):
                candidates.append(
                    (match.start(), match.end(), -1, "protected_term", match.group(0))
                )

        selected: list[tuple[int, int, str, str]] = []
        occupied: list[tuple[int, int]] = []
        for start, end, priority, kind, original in sorted(
            candidates, key=lambda item: (item[0], item[2], -(item[1] - item[0]))
        ):
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                continue
            selected.append((start, end, kind, original))
            occupied.append((start, end))

        placeholders: list[Placeholder] = []
        replacements: list[tuple[int, int, str]] = []
        for index, (start, end, kind, original) in enumerate(
            sorted(selected, key=lambda item: item[0])
        ):
            code = sha256(
                f"{task_id}|{segment_id}|{kind}|{index}|{original}".encode("utf-8")
            ).hexdigest()[:10].upper()
            token = f"⟦X{code}⟧"
            placeholders.append(Placeholder(token=token, original=original, kind=kind))
            replacements.append((start, end, token))

        protected = text
        for start, end, token in reversed(replacements):
            protected = protected[:start] + token + protected[end:]
        return PlaceholderResult(protected_text=protected, placeholders=placeholders)

    def restore(self, translated_text: str, placeholders: list[Placeholder]) -> RestoreResult:
        restored = translated_text
        missing: list[str] = []
        duplicated: list[str] = []
        expected_codes = {item.token[1:-1].upper() for item in placeholders}

        for placeholder in placeholders:
            code = placeholder.token[1:-1]
            tolerant = self._tolerant_token_pattern(code)
            matches = list(tolerant.finditer(restored))
            if not matches:
                original_count = restored.count(placeholder.original)
                if original_count == 1:
                    # Manual repair is allowed to insert the protected original
                    # value directly instead of reconstructing the opaque token.
                    continue
                if original_count > 1:
                    duplicated.append(placeholder.token)
                    continue
                missing.append(placeholder.token)
                continue
            if len(matches) > 1:
                duplicated.append(placeholder.token)
                continue
            restored = tolerant.sub(
                lambda _match, original=placeholder.original: original,
                restored,
                count=1,
            )

        unknown: list[str] = []
        for match in BARE_TOKEN_PATTERN.finditer(restored):
            code = match.group(0).upper()
            if code not in expected_codes:
                unknown.append(match.group(0))

        return RestoreResult(
            restored_text=restored,
            missing=sorted(set(missing)),
            duplicated=sorted(set(duplicated)),
            unknown=sorted(set(unknown)),
        )

    def allowed_missing_tokens(
        self,
        result: RestoreResult,
        placeholders: list[Placeholder],
        ignore_number_warnings: bool,
    ) -> list[str]:
        by_token = {placeholder.token: placeholder for placeholder in placeholders}
        return [
            token
            for token in result.missing
            if self.allow_missing_placeholder(
                by_token.get(token),
                ignore_number_warnings,
            )
        ]

    def blocking_missing_tokens(
        self,
        result: RestoreResult,
        placeholders: list[Placeholder],
        ignore_number_warnings: bool,
    ) -> list[str]:
        allowed = set(
            self.allowed_missing_tokens(
                result,
                placeholders,
                ignore_number_warnings,
            )
        )
        return [token for token in result.missing if token not in allowed]

    def output_restore_ok(
        self,
        result: RestoreResult,
        placeholders: list[Placeholder],
        ignore_number_warnings: bool,
    ) -> bool:
        return not (
            self.blocking_missing_tokens(
                result,
                placeholders,
                ignore_number_warnings,
            )
            or result.duplicated
            or result.unknown
        )

    @staticmethod
    def allow_missing_placeholder(
        placeholder: Placeholder | None,
        ignore_number_warnings: bool,
    ) -> bool:
        if placeholder is None:
            return False
        if placeholder.kind == "number_unit" and ignore_number_warnings:
            return True
        return placeholder.kind == "file_path" and not any(
            character.isalpha() for character in placeholder.original
        )

    @staticmethod
    def _tolerant_token_pattern(code: str) -> re.Pattern[str]:
        characters = r"\s*".join(re.escape(character) for character in code)
        return re.compile(
            rf"[\[\]【】〔〕⟦⟧{{}}()（）]?\s*{characters}\s*"
            rf"[\[\]【】〔〕⟦⟧{{}}()（）]?",
            re.IGNORECASE,
        )


def extract_numbers(text: str) -> Counter[str]:
    # Do not use ``\w`` boundaries here: Chinese normally touches numbers
    # directly (例如“第182页”), while English commonly uses suffixes such as
    # “19th”. Signs and range dashes are ignored for a stable source/target
    # comparison; the exact unit-bearing values are protected separately.
    values = re.findall(r"\d+(?:[.,]\d+)?", text)
    return Counter(value.replace(",", ".") for value in values)


def chinese_character_ratio(text: str) -> float:
    meaningful = [character for character in text if character.isalpha()]
    if not meaningful:
        return 0.0
    chinese = sum("\u3400" <= character <= "\u9fff" for character in meaningful)
    return chinese / len(meaningful)
