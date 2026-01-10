"""Legacy preprocessing API.

This module is kept for backward compatibility, but the single source of truth
for normalization is now `shared.vi_processor.normalize()`.
"""

from shared.vi_processor import normalize


def preprocess_text(raw_text: str) -> dict:
    """Canonical pipeline for both training and inference.

    Returns: {raw_text, normalized_text, amount}
    """

    res = normalize(raw_text)
    return {"raw_text": res.raw_text, "normalized_text": res.normalized_text, "amount": res.amount}

