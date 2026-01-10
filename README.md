# Capy Teacher - Dataset Manager cho PhoBERT

Module nội bộ để thu thập và quản lý dữ liệu huấn luyện cho model NLP (PhoBERT) trong app quản lý tài chính cá nhân.

## Nguyên tắc quan trọng

- Train và inference dùng **cùng 1 pipeline preprocess**
- Không gạch tay, không sửa text sau khi tokenizer
- Pipeline: lowercase → word segmentation → parse tiền → replace `<AMOUNT>`

## Cấu trúc project

```
phobert-docker/
├── capy_teacher/           # Module core
│   ├── preprocess.py       # Pipeline chuẩn (shared)
│   ├── models.py           # SQLAlchemy models
│   ├── crud.py             # CRUD operations
│   ├── database.py         # DB connection
│   └── export.py           # Export train.csv
├── api.py                  # REST API server (FastAPI)
├── cli.py                  # CLI tool
├── example_usage.py        # Example script
├── docker-compose.yml      # PostgreSQL + app
└── dataset/
    └── train.csv           # Export output
```

## Quick Start

### 1. Start PostgreSQL

```bash
docker-compose up -d postgres
```

### 2. Init database

```bash
python cli.py init
```

### 3. Tạo examples

```bash
python cli.py create "bánh mì 50k" chi_an_uong
python cli.py create "grab 35k" chi_di_chuyen
python cli.py create "lương tháng 1" thu_luong
```

### 4. List examples

```bash
python cli.py list
python cli.py list --label chi_an_uong --limit 10
```

### 5. Update example

```bash
python cli.py update 1 --label chi_mua_sam
python cli.py update 2 --deactivate
```

### 6. Stats

```bash
python cli.py stats
```

### 7. Export train.csv

```bash
python cli.py export --output dataset/train.csv
```

## Labels

- `chi_an_uong`, `chi_di_chuyen`, `chi_sinh_hoat`, `chi_mua_sam`
- `chi_giai_tri`, `chi_phat_trien`, `chi_xa_hoi`, `chi_khac`
- `thu_luong`, `thu_thuong`, `thu_dau_tu`, `thu_khac`

## Preprocess Pipeline

```python
from capy_teacher import preprocess_text

result = preprocess_text("bánh mì 50k")
# {
#   'raw_text': 'bánh mì 50k',
#   'normalized_text': 'bánh_mì <AMOUNT>',
#   'amount': 50000.0
# }
```

## Python API

```python
from capy_teacher.database import SessionLocal, init_db
from capy_teacher.crud import DatasetManager
from capy_teacher.export import export_train_csv

init_db()
db = SessionLocal()
manager = DatasetManager(db)

# Create
ex = manager.create_example("bánh mì 50k", "chi_an_uong")

# List
examples = manager.list_examples(label="chi_an_uong", limit=10)

# Update
manager.update_example(1, label="chi_mua_sam")

# Soft delete
manager.soft_delete(2)

# Stats
stats = manager.get_label_stats()

# Export
result = export_train_csv(db, "train.csv")

db.close()
```

## REST API

### Start API Server

```bash
docker-compose up -d
```

API sẽ chạy tại: http://localhost:8000

### Tạo example qua API (TỰ ĐỘNG PREPROCESS)

```bash
curl -X POST "http://localhost:8000/examples" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "bánh mì 50k",
    "label": "chi_an_uong"
  }'
```

Response:
```json
{
  "id": 1,
  "raw_text": "bánh mì 50k",
  "normalized_text": "bánh_mì <AMOUNT>",
  "label": "chi_an_uong",
  "amount": 50000.0,
  ...
}
```

**Khi gửi `raw_text`, hệ thống tự động:**
- Lowercase và trim
- Parse tiền (50k → 50000.0)
- Word segmentation (bánh mì → bánh_mì)
- Replace tiền bằng `<AMOUNT>` token

Xem chi tiết tại [API_USAGE.md](API_USAGE.md)

### API Documentation

Truy cập http://localhost:8000/docs để xem Swagger UI

## Docker Compose

```bash
docker-compose up -d
docker-compose exec app python cli.py stats
docker-compose down
```

## Ví dụ đầy đủ

```bash
python example_usage.py
```

## Fine-tune PhoBERT (text classification)

Dataset export mặc định nằm ở `dataset/train.csv` với format: `text,label`.

### 1) Cài dependencies cho training

```bash
python -m pip install -r requirements-train.txt
```

### 2) Train

```bash
python -m training.train_textcls --train_csv dataset/train.csv --output_dir artifacts/phobert_textcls
```

Note: `dataset/train.csv` exported by the teacher already contains `normalized_text`, so training defaults to `--no_preprocess`.
If you train from a raw-text CSV, pass `--preprocess`.

Mặc định script sẽ:
- Tự split validation 10% (hoặc truyền `--eval_csv`)
- Dùng chung preprocess pipeline `capy_teacher.preprocess_text`
- Normalize token tiền về dạng `<AMOUNT>` (kể cả dữ liệu có dạng `< AMOUNT >`)

### 3) Predict thử

```bash
python -m training.predict_textcls --model_dir artifacts/phobert_textcls --text "ăn sáng bánh mì 50k"
```

