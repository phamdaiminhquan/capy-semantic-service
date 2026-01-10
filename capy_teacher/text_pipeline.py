import re
from typing import Any

from .preprocess import preprocess_text


AMOUNT_TOKEN = "<AMOUNT>"
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

    processed = preprocess_text(raw_text)
    processed["normalized_text"] = ensure_amount_token_spacing(processed["normalized_text"])
    return processed


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
