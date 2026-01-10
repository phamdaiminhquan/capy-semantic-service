import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


from capy_teacher.text_pipeline import AMOUNT_TOKEN as AMOUNT_CANONICAL
from capy_teacher.text_pipeline import normalize_amount_token


def read_labeled_csv(path: str | Path, *, text_col: str = "text", label_col: str = "label") -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        if text_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must have columns '{text_col}' and '{label_col}'. Found: {reader.fieldnames}"
            )

        rows: list[dict] = []
        for row in reader:
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if not text or not label:
                continue
            rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError("CSV has no valid (text,label) rows")

    return rows


def train_val_split(
    items: list[dict],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:val_size]
    train_items = shuffled[val_size:]
    if not train_items:
        raise ValueError("val split too large; train set would be empty")

    return train_items, val_items


@dataclass(frozen=True)
class LabelMapping:
    label2id: dict[str, int]
    id2label: dict[int, str]

    @staticmethod
    def from_labels(labels: Iterable[str]) -> "LabelMapping":
        unique = sorted(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique)}
        id2label = {idx: label for label, idx in label2id.items()}
        return LabelMapping(label2id=label2id, id2label=id2label)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "label_map.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, file, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load(model_dir: str | Path) -> "LabelMapping":
        model_dir = Path(model_dir)
        path = model_dir / "label_map.json"
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        return LabelMapping(label2id=label2id, id2label=id2label)
