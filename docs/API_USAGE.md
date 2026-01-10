# Capy Teacher API Usage

## Endpoints

### 1. POST /examples - Tạo example mới
**Tự động preprocess raw_text**

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
  "is_active": true,
  "created_at": "2026-01-03T13:00:00",
  "updated_at": "2026-01-03T13:00:00"
}
```

### 1b. POST /examples/batch - Tạo nhiều examples cùng label

```bash
curl -X POST "http://localhost:8000/examples/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "label": "chi_an_uong",
    "raw_texts": [
      "bánh mì 50k",
      "ăn trưa bún bò 70k",
      "cà phê 35k"
    ]
  }'
```

PowerShell (UTF-8) ví dụ:

```powershell
$payload = @{ label = "chi_an_uong"; raw_texts = @("bánh mì 50k", "ăn trưa bún bò 70k") } | ConvertTo-Json
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/examples/batch" -ContentType "application/json; charset=utf-8" -Body $bytes
```

### 2. GET /examples - List examples

```bash
# List tất cả
curl "http://localhost:8000/examples"

# Filter theo label
curl "http://localhost:8000/examples?label=chi_an_uong"

# Pagination
curl "http://localhost:8000/examples?limit=10&offset=0"
```

### 3. GET /examples/{id} - Lấy example theo ID

```bash
curl "http://localhost:8000/examples/1"
```

### 4. PATCH /examples/{id} - Cập nhật example

**Chỉ cho sửa label và is_active**

```bash
curl -X PATCH "http://localhost:8000/examples/1" \
  -H "Content-Type: application/json" \
  -d '{
    "label": "chi_mua_sam"
  }'
```

### 5. DELETE /examples/{id} - Soft delete

```bash
curl -X DELETE "http://localhost:8000/examples/1"
```

### 6. GET /stats - Thống kê

```bash
curl "http://localhost:8000/stats"
```

Response:
```json
{
  "total": 10,
  "label_counts": {
    "chi_an_uong": 3,
    "chi_mua_sam": 2,
    ...
  }
}
```

### 7. POST /preprocess - Preprocess text (utility)

```bash
curl -X POST "http://localhost:8000/preprocess?raw_text=bánh mì 50k"
```

Response:
```json
{
  "raw_text": "bánh mì 50k",
  "normalized_text": "bánh_mì <AMOUNT>",
  "amount": 50000.0
}
```

### 8. POST /export - Export train.csv

```bash
curl -X POST "http://localhost:8000/export?output_path=train.csv"
```

### 9. GET /labels - Danh sách labels

```bash
curl "http://localhost:8000/labels"
```

### 10. POST /predict - Dự đoán label từ raw_text

API sẽ tự chạy preprocess pipeline (lowercase → parse tiền → word segmentation → `<AMOUNT>`), rồi gọi model fine-tuned trong `artifacts/phobert_textcls/`.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "ăn sáng bánh mì 50k",
    "top_k": 3
  }'
```

Response (ví dụ):
```json
{
  "raw_text": "ăn sáng bánh mì 50k",
  "normalized_text": "ăn_sáng bánh_mì <AMOUNT>",
  "amount": 50000.0,
  "label": "chi_an_uong",
  "score": 0.93,
  "top_k": [
    {"label": "chi_an_uong", "score": 0.93},
    {"label": "chi_sinh_hoat", "score": 0.04},
    {"label": "chi_khac", "score": 0.02}
  ]
}
```

PowerShell (UTF-8) ví dụ:

```powershell
$payload = @{ raw_text = "ăn sáng bánh mì 50k"; top_k = 3 } | ConvertTo-Json
$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json; charset=utf-8" -Body $bytes
```

## Python Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Tạo example - TỰ ĐỘNG PREPROCESS
response = requests.post(f"{BASE_URL}/examples", json={
    "raw_text": "lên đời điện thoại iphone 16 promax 7tr5",
    "label": "chi_mua_sam"
})
print(response.json())
# {
#   "normalized_text": "lên_đời điện_thoại iphone 16 promax <AMOUNT>",
#   "amount": 7500000.0,
#   ...
# }

# List examples
response = requests.get(f"{BASE_URL}/examples?label=chi_an_uong")
examples = response.json()

# Preprocess text
response = requests.post(f"{BASE_URL}/preprocess", params={"raw_text": "grab 35k"})
preprocessed = response.json()
print(preprocessed["normalized_text"])  # "grab <AMOUNT>"
```

