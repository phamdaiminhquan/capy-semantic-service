"""Recompute normalized_text (and amount) from raw_text for all examples.

Use this when the teacher preprocessing logic changes and you need to migrate
existing DB rows to match the new canonical pipeline.

Examples:
  python scripts/migrate_normalized_text.py --only-active
  python scripts/migrate_normalized_text.py --dry-run --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running via: `python scripts/migrate_normalized_text.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from capy_teacher.database import SessionLocal
from capy_teacher.models import TrainingExample
from capy_teacher.text_pipeline import teacher_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate training_examples.normalized_text based on raw_text")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print changes without writing to DB")
    parser.add_argument(
        "--print-rows",
        action="store_true",
        help="Print a preview line for each processed row (useful with --dry-run)",
    )
    parser.add_argument("--only-active", action="store_true", help="Only migrate rows where is_active=true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows processed")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        query = db.query(TrainingExample).order_by(TrainingExample.id.asc())
        if args.only_active:
            query = query.filter(TrainingExample.is_active.is_(True))
        if args.limit is not None:
            query = query.limit(int(args.limit))

        rows = query.all()
        changed = 0

        for ex in rows:
            processed = teacher_preprocess(ex.raw_text)
            new_norm = processed["normalized_text"]
            new_amount = processed["amount"]

            if ex.normalized_text != new_norm or ex.amount != new_amount:
                changed += 1
                if args.dry_run:
                    print(f"#{ex.id}: '{ex.normalized_text}' -> '{new_norm}' | amount {ex.amount} -> {new_amount}")
                else:
                    ex.normalized_text = new_norm
                    ex.amount = new_amount
            elif args.dry_run and args.print_rows:
                print(f"#{ex.id}: OK '{ex.normalized_text}' | amount {ex.amount}")

        if args.dry_run:
            print(f"Dry-run done. Would change {changed}/{len(rows)} rows.")
            return

        db.commit()
        print(f"Migration done. Changed {changed}/{len(rows)} rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
