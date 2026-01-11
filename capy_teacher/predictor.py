import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import numpy as np # <--- Cần thêm numpy để tính toán
import onnxruntime as ort # <--- Thư viện chạy ONNX
from transformers import AutoTokenizer

from .text_pipeline import teacher_preprocess, tokenize_text

DEFAULT_MODEL_DIR = "artifacts/onnx"
DEFAULT_ONNX_FILE = "model_int8.onnx"
DEFAULT_MAX_LENGTH = 128

@dataclass(frozen=True)
class Prediction:
    label: str
    score: float

class TextClassifier:
    def __init__(self, model_dir: str | Path, onnx_filename: str = DEFAULT_ONNX_FILE):
        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / onnx_filename

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        # 1. Load Tokenizer (Vẫn dùng của HuggingFace, lấy từ thư mục gốc)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # 2. Load ONNX Session (Thay cho AutoModel)
        # Load model vào RAM để chạy
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        
        # Lấy tên input đầu vào của model (thường là 'input_ids' và 'attention_mask')
        self.input_names = [i.name for i in self.session.get_inputs()]
        
        self.id2label = self._load_id2label()

    def _load_id2label(self) -> dict[int, str]:
        # (Giữ nguyên logic cũ của bạn để load label map)
        label_map_path = self.model_dir / "label_map.json"
        if label_map_path.exists():
            with label_map_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(k): str(v) for k, v in data.get("id2label", {}).items()}
        # Fallback load từ config.json nếu cần...
        config_path = self.model_dir / "config.json"
        if config_path.exists():
             with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return {int(k): str(v) for k, v in data.get("id2label", {}).items()}
        raise ValueError("Could not load id2label mapping")

    def _softmax(self, x):
        """Hàm tính xác suất thủ công (vì ONNX trả về raw scores)"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, raw_text: str, *, top_k: int = 3) -> dict:
        processed = teacher_preprocess(raw_text)

        # 3. Tokenize nhưng trả về NUMPY (return_tensors="np")
        enc = tokenize_text(
            processed["normalized_text"],
            tokenizer=self.tokenizer,
            return_tensors="np", # <--- QUAN TRỌNG: Đổi pt thành np
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
        )

        # 4. Chuẩn bị input cho ONNX
        # ONNX yêu cầu input dạng dictionary { "tên_cổng": dữ_liệu }
        # input_ids và attention_mask phải là int64
        onnx_inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)
        }
        
        # Nếu model có token_type_ids (PhBERT thường có), cần thêm vào
        if "token_type_ids" in enc and "token_type_ids" in self.input_names:
             onnx_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        # 5. Chạy dự đoán (Run Session)
        # output[0] chính là logits (điểm số thô)
        logits = self.session.run(None, onnx_inputs)[0]

        # 6. Tính Softmax để ra % (Confidence Score)
        probs = self._softmax(logits) # probs shape (1, num_labels)
        
        # Xử lý kết quả (Giống logic cũ nhưng dùng numpy)
        top_k = max(1, min(int(top_k), int(probs.shape[-1])))
        
        # Lấy top k (Dùng numpy argsort)
        # argsort mặc định tăng dần, nên lấy [-top_k:] rồi đảo ngược [::-1]
        top_indices = np.argsort(probs[0])[-top_k:][::-1]
        top_scores = probs[0][top_indices]

        top = [
            Prediction(label=self.id2label[int(idx)], score=float(score))
            for score, idx in zip(top_scores, top_indices)
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

