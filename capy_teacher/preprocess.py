from shared.vi_processor import normalize


def preprocess_text(raw_text: str) -> dict:
    res = normalize(raw_text)
    return {"raw_text": res.raw_text, "normalized_text": res.normalized_text, "amount": res.amount}

