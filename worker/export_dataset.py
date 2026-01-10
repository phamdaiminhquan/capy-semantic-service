from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sqlalchemy.orm import Session

from capy_teacher.database import SessionLocal
from capy_teacher.models import TrainingExample
from shared.vi_processor import normalize


def export_dataset(
    db: Session,
    *,
    output_path: str,
    fmt: str = "csv",
    only_active: bool = True,
) -> dict:
    rows = db.query(TrainingExample)
    if only_active:
        rows = rows.filter(TrainingExample.is_active.is_(True))

    examples = rows.order_by(TrainingExample.id.asc()).all()
    if not examples:
        raise ValueError("No examples found to export")

    fmt = fmt.lower().strip()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("fmt must be 'csv' or 'parquet'")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    label_counts: dict[str, int] = {}

    for ex in examples:
        res = normalize(ex.raw_text)
        records.append({"text": res.normalized_text, "label": ex.label})
        label_counts[ex.label] = label_counts.get(ex.label, 0) + 1

    if fmt == "csv":
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(records)
    else:
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Parquet export requires pandas + pyarrow") from e
        df = pd.DataFrame.from_records(records)
        df.to_parquet(out, index=False)

    return {"total": len(records), "output_path": str(out), "label_counts": label_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset from Postgres using canonical vi_processor.normalize")
    parser.add_argument("--output", default="dataset/train.csv")
    parser.add_argument("--format", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = export_dataset(
            db,
            output_path=args.output,
            fmt=args.format,
            only_active=not args.include_inactive,
        )
        print(f"✓ Exported {result['total']} rows to {result['output_path']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
