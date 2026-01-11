import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from capy_teacher.text_pipeline import ensure_amount_token_spacing, teacher_preprocess, tokenize_text

from .utils import AMOUNT_CANONICAL, LabelMapping, normalize_amount_token, read_labeled_csv, train_val_split


@dataclass
class Example:
    text: str
    label: str


class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: list[Example],
        *,
        tokenizer,
        label2id: dict[str, int],
        max_length: int,
        do_preprocess: bool,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.do_preprocess = do_preprocess

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        text = ex.text

        if self.do_preprocess:
            text = teacher_preprocess(text)["normalized_text"]
        else:
            # CSV exported by teacher already contains normalized_text.
            # Still canonicalize amount-token variants/spaces for safety.
            text = ensure_amount_token_spacing(text)

        enc = tokenize_text(
            text,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = self.label2id[ex.label]
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = float((preds == labels).mean())

    # Macro F1 without pulling sklearn (simple implementation)
    f1s = []
    for cls in sorted(set(labels.tolist())):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)

    f1_macro = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": accuracy, "f1_macro": f1_macro}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT for text classification (CSV: text,label).")
    parser.add_argument("--train_csv", default="dataset/train.csv")
    parser.add_argument("--eval_csv", default=None)
    parser.add_argument("--val_split", type=float, default=0.1, help="Used if --eval_csv is not provided")
    parser.add_argument("--model_name", default="vinai/phobert-base")
    parser.add_argument("--output_dir", default="artifacts/phobert_textcls")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply teacher preprocessing before tokenization (use if CSV contains raw text, not teacher-exported normalized_text)",
    )
    parser.add_argument("--no_preprocess", dest="preprocess", action="store_false")
    # Default assumes dataset/train.csv exported from the teacher DB (already normalized).
    parser.set_defaults(preprocess=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_labeled_csv(args.train_csv)
    if args.eval_csv:
        eval_rows = read_labeled_csv(args.eval_csv)
    else:
        train_rows, eval_rows = train_val_split(train_rows, val_ratio=args.val_split, seed=args.seed)

    train_examples = [Example(text=r["text"], label=r["label"]) for r in train_rows]
    eval_examples = [Example(text=r["text"], label=r["label"]) for r in eval_rows]

    mapping = LabelMapping.from_labels([ex.label for ex in train_examples])
    mapping.save(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure our canonical token is treated as a single special token.
    tokenizer.add_special_tokens({"additional_special_tokens": [AMOUNT_CANONICAL]})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(mapping.label2id),
        id2label=mapping.id2label,
        label2id=mapping.label2id,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_ds = TextClassificationDataset(
        train_examples,
        tokenizer=tokenizer,
        label2id=mapping.label2id,
        max_length=args.max_length,
        do_preprocess=args.preprocess,
    )
    eval_ds = TextClassificationDataset(
        eval_examples,
        tokenizer=tokenizer,
        label2id=mapping.label2id,
        max_length=args.max_length,
        do_preprocess=args.preprocess,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
