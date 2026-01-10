import re
from typing import Any

from shared.vi_processor import AMOUNT_TOKEN, count_tokens, normalize, sha256_text


_AMOUNT_PATTERN = re.compile(r"<\s*amount\s*>", re.IGNORECASE)


def normalize_amount_token(text: str) -> str:
    """Normalize variants like '< AMOUNT >' to '<AMOUNT>'."""
    return _AMOUNT_PATTERN.sub(AMOUNT_TOKEN, text)


def ensure_amount_token_spacing(text: str) -> str:
    """Ensure the canonical amount token has single spaces around it."""
    text = normalize_amount_token(text)
    text = re.sub(r"\s*" + re.escape(AMOUNT_TOKEN) + r"\s*", f" {AMOUNT_TOKEN} ", text)
    return re.sub(r"\s+", " ", text).strip()


def teacher_preprocess(raw_text: str) -> dict:
    """Canonical preprocessing used by teacher (DB), training and inference.

    Returns: {raw_text, normalized_text, amount}

    Notes:
    - Delegates segmentation + amount extraction to `preprocess_text`.
    - Applies canonical normalization for amount-token variants/spaces.
    """

    res = normalize(raw_text)
    # Safety: enforce canonical spacing in case downstream code mutates the string.
    normalized_text = ensure_amount_token_spacing(res.normalized_text)
    return {"raw_text": res.raw_text, "normalized_text": normalized_text, "amount": res.amount}


def teacher_cache_key(raw_text: str) -> str:
    """Cache key derived from canonical normalized text (sha256)."""

    return sha256_text(normalize(raw_text).normalized_text)


def teacher_token_count(raw_text: str) -> int:
    """Token count derived from canonical normalized text."""

    return count_tokens(normalize(raw_text).normalized_text)


def tokenize_text(
    text: str,
    *,
    tokenizer: Any,
    truncation: bool = True,
    max_length: int | None = None,
    return_tensors: str | None = None,
) -> dict:
    """Shared tokenizer wrapper.

    - For inference: pass `return_tensors="pt"`.
    - For HF Trainer datasets: keep `return_tensors=None`.
    """

    kwargs: dict[str, Any] = {"truncation": truncation}
    if max_length is not None:
        kwargs["max_length"] = int(max_length)
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors

    return tokenizer(text, **kwargs)
