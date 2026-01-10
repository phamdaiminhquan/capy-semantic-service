from __future__ import annotations

from pyvi import ViTokenizer
import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass


AMOUNT_TOKEN = "<AMOUNT>"
_MONEY_PLACEHOLDER = "money_token_placeholder"


@dataclass(frozen=True)
class NormalizedResult:
    raw_text: str
    normalized_text: str
    amount: float | None

    @property
    def cache_key(self) -> str:
        return sha256_text(self.normalized_text)

    @property
    def token_count(self) -> int:
        return count_tokens(self.normalized_text)


def normalize(raw_text: str) -> NormalizedResult:
    """Single source of truth for Vietnamese text normalization.

    Pipeline (required):
    1) underthesea tokenization (format="text")
    2) Unicode normalization (NFC)
    3) money parsing + replacement with <AMOUNT>

    Notes:
    - We keep a placeholder during tokenization so the tokenizer doesn't split <AMOUNT>.
    - Output aims to match the legacy teacher_preprocess behavior: compound words joined by '_',
      and the canonical <AMOUNT> token appears with single spaces around it.
    """

    if raw_text is None:
        raw_text = ""

    # Clean-ish: keep behavior close to legacy (lower + trim + collapse spaces)
    text = str(raw_text).strip().lower()
    text = re.sub(r"\s+", " ", text)

    # Money parsing + replacement
    amount = _extract_amount(text)
    text = _replace_money(text)

    # Protect token before segmentation
    text = text.replace(AMOUNT_TOKEN, _MONEY_PLACEHOLDER)

    # Segment using underthesea
    text = _word_tokenize(text)

    # Restore amount token and enforce canonical spacing
    text = text.replace(_MONEY_PLACEHOLDER, AMOUNT_TOKEN)
    text = _ensure_amount_token_spacing(text)

    # Unicode normalize Vietnamese (precomposed)
    text = unicodedata.normalize("NFC", text)

    return NormalizedResult(raw_text=str(raw_text), normalized_text=text, amount=amount)


def count_tokens(normalized_text: str) -> int:
    """Count tokens based on the tokenizer output.

    - Tokens are space-separated.
    - Compound words use '_' and remain a single token.
    """

    if not normalized_text:
        return 0
    return len([t for t in normalized_text.strip().split() if t])


def sha256_text(normalized_text: str) -> str:
    return hashlib.sha256((normalized_text or "").encode("utf-8")).hexdigest()


def _word_tokenize(text: str) -> str:
    segmented = ViTokenizer.tokenize(text)
    return re.sub(r"\s+", " ", segmented).strip()


_AMOUNT_PATTERN = re.compile(r"<\s*amount\s*>", re.IGNORECASE)


def _ensure_amount_token_spacing(text: str) -> str:
    text = _AMOUNT_PATTERN.sub(AMOUNT_TOKEN, text)
    text = re.sub(r"\s*" + re.escape(AMOUNT_TOKEN) + r"\s*", f" {AMOUNT_TOKEN} ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_amount(text: str) -> float | None:
    """Parse money amount from Vietnamese text.

    Supports: 50k, 50.000, 50 nghìn/ngàn, 7tr5, 1tr2, 2tr, 2 triệu...
    """

    patterns: list[tuple[str, object]] = [
        (
            r"(\d+(?:\.\d+)?)\s*tr(?:iệu)?(?:\s*(\d+))?",
            lambda m: float(m.group(1)) * 1_000_000
            + (float(m.group(2)) * 100_000 if m.group(2) else 0),
        ),
        (r"(\d+(?:\.\d+)?)\s*k", lambda m: float(m.group(1)) * 1_000),
        (
            r"(\d+(?:\.\d+)?)\s*(?:nghìn|ngàn|ngh)",
            lambda m: float(m.group(1)) * 1_000,
        ),
        (r"(\d{1,3}(?:\.\d{3})+)", lambda m: float(m.group(1).replace(".", ""))),
    ]

    for pattern, converter in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return converter(match)

    return None


def _replace_money(text: str) -> str:
    patterns = [
        r"\d+(?:\.\d+)?\s*tr(?:iệu)?(?:\s*\d+)?",
        r"\d+(?:\.\d+)?\s*k",
        r"\d+(?:\.\d+)?\s*(?:nghìn|ngàn|ngh)",
        r"\d{1,3}(?:\.\d{3})+",
    ]

    for pattern in patterns:
        text = re.sub(pattern, AMOUNT_TOKEN, text, flags=re.IGNORECASE)

    return text
