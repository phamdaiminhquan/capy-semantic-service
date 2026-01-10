import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .text_pipeline import teacher_preprocess, tokenize_text


DEFAULT_MODEL_DIR = "artifacts/phobert_textcls"
DEFAULT_MAX_LENGTH = 128


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


class TextClassifier:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.id2label = self._load_id2label()

    def _load_id2label(self) -> dict[int, str]:
        # Prefer our training artifact if present; otherwise fall back to model config
        label_map_path = self.model_dir / "label_map.json"
        if label_map_path.exists():
            with label_map_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): str(v) for k, v in data.get("id2label", {}).items()}

        cfg = getattr(self.model, "config", None)
        id2label = getattr(cfg, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            return {int(k): str(v) for k, v in id2label.items()}

        raise ValueError("Could not load id2label mapping")

    def predict(self, raw_text: str, *, top_k: int = 3) -> dict:
        processed = teacher_preprocess(raw_text)

        enc = tokenize_text(
            processed["normalized_text"],
            tokenizer=self.tokenizer,
            return_tensors="pt",
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.softmax(out.logits[0], dim=-1)

        top_k = max(1, min(int(top_k), int(probs.shape[-1])))
        values, indices = torch.topk(probs, k=top_k)

        top = [
            Prediction(label=self.id2label[int(idx)], score=float(score))
            for score, idx in zip(values.tolist(), indices.tolist())
        ]

        return {
            "raw_text": raw_text,
            "normalized_text": processed["normalized_text"],
            "amount": processed["amount"],
            "label": top[0].label,
            "score": top[0].score,
            "top_k": [{"label": p.label, "score": p.score} for p in top],
        }


_classifier: TextClassifier | None = None


def get_text_classifier() -> TextClassifier:
    global _classifier
    if _classifier is not None:
        return _classifier

    model_dir = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
    # `.env` is primarily for docker-compose, so MODEL_DIR may be `/app/...`.
    # When running on the host OS, rewrite it to this repo folder.
    if not os.path.exists("/.dockerenv"):
        model_dir_posix = str(PurePosixPath(str(model_dir)))
        if model_dir_posix.startswith("/app/"):
            repo_root = Path(__file__).resolve().parent.parent
            model_dir = str(repo_root / model_dir_posix.removeprefix("/app/"))
    _classifier = TextClassifier(model_dir=model_dir)
    return _classifier
