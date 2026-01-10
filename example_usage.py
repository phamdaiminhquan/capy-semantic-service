from capy_teacher.database import SessionLocal, init_db
from capy_teacher.crud import DatasetManager
from capy_teacher.export import export_train_csv
from capy_teacher.preprocess import preprocess_text

# Test preprocess pipeline
print("=== Test Preprocess Pipeline ===")
test_cases = [
    "bánh mì 50k",
    "15k bánh mì",
    "lên đời điện thoại iphone 16 promax 7tr5",
    "đổi điện thoại iphone 6s 5tr",
    "được hoàn tiền grab 40k",
]

for text in test_cases:
    result = preprocess_text(text)
    print(f"\nRaw: {result['raw_text']}")
    print(f"Normalized: {result['normalized_text']}")
    print(f"Amount: {result['amount']}")

# Init DB và thêm examples
print("\n\n=== Database Operations ===")
init_db()
db = SessionLocal()
manager = DatasetManager(db)

# Tạo examples
examples_to_add = [
    ("bánh mì 50k", "chi_an_uong"),
    ("15k bánh mì", "chi_an_uong"),
    ("lên đời điện thoại iphone 16 promax 7tr5", "chi_mua_sam"),
    ("đổi điện thoại iphone 6s 5tr", "chi_mua_sam"),
    ("được hoàn tiền grab 40k", "thu_khac"),
    ("lương tháng 1", "thu_luong"),
    ("cà phê 25k", "chi_an_uong"),
    ("grab 35k", "chi_di_chuyen"),
]

print("\nCreating examples...")
for text, label in examples_to_add:
    ex = manager.create_example(text, label)
    print(f"✓ #{ex.id} [{label}] {ex.normalized_text}")

# Stats
print("\n\n=== Label Statistics ===")
stats = manager.get_label_stats()
for label, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"{label:20} {count:4}")

# Export
print("\n\n=== Export train.csv ===")
result = export_train_csv(db, "dataset/train.csv")
print(f"✓ Exported {result['total']} examples to {result['output_path']}")

db.close()
print("\n✓ Done!")

