import re
from pyvi import ViTokenizer

MONEY_TOKEN = "<AMOUNT>"

def preprocess_text(raw_text: str) -> dict:
    """
    Pipeline chuẩn cho cả train và inference
    Returns: {raw_text, normalized_text, amount}
    """
    # lowercase và trim
    text = raw_text.strip().lower()
    
    # parse tiền
    amount = _extract_amount(text)
    
    # thay toàn bộ tiền bằng placeholder đơn giản (không có ký tự đặc biệt)
    placeholder = "money_token_placeholder"
    text = _replace_money(text)
    text = text.replace(MONEY_TOKEN, placeholder)
    
    # word segmentation
    text = ViTokenizer.tokenize(text)
    
    # thay placeholder về <AMOUNT> và normalize spaces
    text = text.replace(placeholder, MONEY_TOKEN)
    # đảm bảo <AMOUNT> có 1 space mỗi bên
    text = re.sub(r'\s*' + re.escape(MONEY_TOKEN) + r'\s*', ' ' + MONEY_TOKEN + ' ', text)
    text = text.strip()
    
    return {
        "raw_text": raw_text,
        "normalized_text": text,
        "amount": amount
    }

def _extract_amount(text: str) -> float | None:
    """Parse tiền từ text: 50k, 50.000, 50 nghìn, 7tr5, 1tr2"""
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*tr(?:iệu)?(?:\s*(\d+))?', lambda m: float(m.group(1)) * 1_000_000 + (float(m.group(2)) * 100_000 if m.group(2) else 0)),
        (r'(\d+(?:\.\d+)?)\s*k', lambda m: float(m.group(1)) * 1_000),
        (r'(\d+(?:\.\d+)?)\s*(?:nghìn|ngàn|ngh)', lambda m: float(m.group(1)) * 1_000),
        (r'(\d{1,3}(?:\.\d{3})+)', lambda m: float(m.group(1).replace('.', ''))),
    ]
    
    for pattern, converter in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return converter(match)
    
    return None

def _replace_money(text: str) -> str:
    """Thay tất cả pattern tiền bằng MONEY_TOKEN"""
    patterns = [
        r'\d+(?:\.\d+)?\s*tr(?:iệu)?(?:\s*\d+)?',
        r'\d+(?:\.\d+)?\s*k',
        r'\d+(?:\.\d+)?\s*(?:nghìn|ngàn|ngh)',
        r'\d{1,3}(?:\.\d{3})+',
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, MONEY_TOKEN, text, flags=re.IGNORECASE)
    
    return text

