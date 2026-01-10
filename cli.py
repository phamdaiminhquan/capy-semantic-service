#!/usr/bin/env python
import argparse
from capy_teacher.database import SessionLocal, init_db
from capy_teacher.crud import DatasetManager
from capy_teacher.export import export_train_csv

def main():
    parser = argparse.ArgumentParser(description="Capy Teacher - Dataset Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    # Init DB
    init_parser = subparsers.add_parser("init", help="Init database")
    
    # Create example
    create_parser = subparsers.add_parser("create", help="Create example")
    create_parser.add_argument("text", help="Raw text")
    create_parser.add_argument("label", help="Label")
    
    # List examples
    list_parser = subparsers.add_parser("list", help="List examples")
    list_parser.add_argument("--label", help="Filter by label")
    list_parser.add_argument("--limit", type=int, default=20, help="Limit")
    
    # Update example
    update_parser = subparsers.add_parser("update", help="Update example")
    update_parser.add_argument("id", type=int, help="Example ID")
    update_parser.add_argument("--label", help="New label")
    update_parser.add_argument("--deactivate", action="store_true", help="Soft delete")
    
    # Stats
    stats_parser = subparsers.add_parser("stats", help="Label statistics")
    
    # Export
    export_parser = subparsers.add_parser("export", help="Export train.csv")
    export_parser.add_argument("--output", default="train.csv", help="Output path")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
        print("✓ Database initialized")
        return
    
    db = SessionLocal()
    manager = DatasetManager(db)
    
    try:
        if args.command == "create":
            ex = manager.create_example(args.text, args.label)
            print(f"✓ Created example #{ex.id}")
            print(f"  Raw: {ex.raw_text}")
            print(f"  Normalized: {ex.normalized_text}")
            print(f"  Label: {ex.label}")
            print(f"  Amount: {ex.amount}")
        
        elif args.command == "list":
            examples = manager.list_examples(label=args.label, limit=args.limit)
            print(f"Found {len(examples)} examples:\n")
            for ex in examples:
                print(f"#{ex.id} [{ex.label}] {ex.normalized_text}")
                if ex.amount:
                    print(f"      Amount: {ex.amount:,.0f}")
        
        elif args.command == "update":
            is_active = None if not args.deactivate else False
            ex = manager.update_example(args.id, label=args.label, is_active=is_active)
            print(f"✓ Updated example #{ex.id}")
        
        elif args.command == "stats":
            stats = manager.get_label_stats()
            total = sum(stats.values())
            print(f"Total active examples: {total}\n")
            for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                print(f"{label:20} {count:4}")
        
        elif args.command == "export":
            result = export_train_csv(db, args.output)
            print(f"✓ Exported {result['total']} examples to {result['output_path']}")
            print("\nLabel distribution:")
            for label, count in sorted(result['label_counts'].items()):
                print(f"  {label:20} {count:4}")
        
        else:
            parser.print_help()
    
    finally:
        db.close()

if __name__ == "__main__":
    main()

