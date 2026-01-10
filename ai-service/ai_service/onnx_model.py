from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path
    tokenizer_path: Path
    label_map_path: Path
    max_length: int = 128


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class OnnxTextClassifier:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

        if not cfg.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {cfg.model_path}")
        if not cfg.tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {cfg.tokenizer_path}")
        if not cfg.label_map_path.exists():
            raise FileNotFoundError(f"label_map.json not found: {cfg.label_map_path}")

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(cfg.model_path), providers=providers)

        self.tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=int(cfg.max_length))
        # RoBERTa-style tokenizers usually need padding to max length for ONNX
        self.tokenizer.enable_padding(length=int(cfg.max_length))

        with cfg.label_map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        id2label = data.get("id2label") or {}
        self.id2label = {int(k): str(v) for k, v in id2label.items()}

        if not self.id2label:
            raise ValueError("label_map.json missing id2label")

        self._input_names = {i.name for i in self.session.get_inputs()}

    @staticmethod
    def from_env() -> "OnnxTextClassifier":
        model_path = Path(os.getenv("ONNX_MODEL_PATH", "artifacts/onnx/model-int8.onnx"))
        tokenizer_path = Path(os.getenv("TOKENIZER_JSON_PATH", "artifacts/onnx/tokenizer.json"))
        label_map_path = Path(os.getenv("LABEL_MAP_PATH", "artifacts/phobert_textcls/label_map.json"))
        max_length = int(os.getenv("MAX_LENGTH", "128"))
        return OnnxTextClassifier(ModelConfig(model_path=model_path, tokenizer_path=tokenizer_path, label_map_path=label_map_path, max_length=max_length))

    def predict(self, normalized_text: str, *, top_k: int = 3) -> dict:
        enc = self.tokenizer.encode(normalized_text)

        input_ids = np.asarray([enc.ids], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask], dtype=np.int64)

        ort_inputs: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            ort_inputs["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            ort_inputs["attention_mask"] = attention_mask
        # Some exports include token_type_ids; PhoBERT (RoBERTa) typically doesn't.
        if "token_type_ids" in self._input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, ort_inputs)

        # Commonly first output is logits
        logits = outputs[0]
        if logits.ndim == 2:
            logits = logits[0]

        probs = _softmax(np.asarray(logits))
        top_k = max(1, min(int(top_k), int(probs.shape[-1])))

        idxs = np.argsort(-probs)[:top_k]
        top = [{"label": self.id2label.get(int(i), str(int(i))), "score": float(probs[int(i)])} for i in idxs]

        return {"label": top[0]["label"], "score": float(top[0]["score"]), "top_k": top}
