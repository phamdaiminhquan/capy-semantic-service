import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from capy_teacher.preprocess import preprocess_text

from .utils import LabelMapping, normalize_amount_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict label with fine-tuned PhoBERT text classifier")
    parser.add_argument("--model_dir", default="artifacts/phobert_textcls")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--no_preprocess", dest="preprocess", action="store_false")
    parser.set_defaults(preprocess=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    mapping = LabelMapping.load(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    text = args.text
    if args.preprocess:
        text = preprocess_text(text)["normalized_text"]
    text = normalize_amount_token(text)

    enc = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits[0], dim=-1)

    top_k = min(args.top_k, probs.shape[-1])
    values, indices = torch.topk(probs, k=top_k)

    for score, idx in zip(values.tolist(), indices.tolist()):
        label = mapping.id2label[int(idx)]
        print(f"{label}\t{score:.4f}")


if __name__ == "__main__":
    main()
