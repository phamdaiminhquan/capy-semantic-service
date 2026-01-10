# Backend Integration (Existing NestJS)

Repo này **không** chứa NestJS backend. Backend hiện tại chỉ cần gọi sang `capy-semantic-service` để:
- Chuẩn hoá text + đếm token **đúng theo underthesea** (không dùng whitespace count)
- Lấy `cache_key = sha256(normalized_text)` để check Redis
- (Tuỳ tier) gọi ONNX local qua ai-service

## 1) AI Service endpoints

Service: `ai-service` (FastAPI)

### POST /analyze-tokens
Request:
```json
{ "raw_text": "ăn sáng bánh mì 50k" }
```
Response:
```json
{
  "raw_text": "ăn sáng bánh mì 50k",
  "normalized_text": "ăn_sáng bánh_mì <AMOUNT>",
  "token_count": 3,
  "cache_key": "<sha256>",
  "amount": 50000.0
}
```

Backend dùng:
- `token_count` để routing tier (<=3 keyword, 4-10 local onnx, >10 gemini)
- `cache_key` để get/set Redis (key dựa trên normalized_text)

### POST /predict
Request:
```json
{ "raw_text": "ăn sáng bánh mì 50k", "top_k": 3 }
```
Response:
```json
{
  "raw_text": "...",
  "normalized_text": "...",
  "token_count": 3,
  "cache_key": "...",
  "amount": 50000.0,
  "label": "chi_an_uong",
  "score": 0.97,
  "top_k": [{"label":"chi_an_uong","score":0.97}]
}
```

## 2) Env vars (ai-service)

- `ONNX_MODEL_PATH` (default `artifacts/onnx/model-int8.onnx`)
- `TOKENIZER_JSON_PATH` (default `artifacts/onnx/tokenizer.json`)
- `LABEL_MAP_PATH` (default `artifacts/phobert_textcls/label_map.json`)
- `MAX_LENGTH` (default `128`)
- `AI_WORKERS` (default `1`, survival mode <=2)

## 3) Redis/Postgres

Redis + Postgres đã chạy ở hệ thống backend của bạn.
Repo này chỉ đảm bảo **pipeline normalize giống hệt** giữa export/inference/cache-key.

Khuyến nghị backend lưu log/gold dataset vào Postgres (bên backend), với các field:
- `source, routing_layer, model_version, created_at`
- `raw_text, normalized_text, cache_key, token_count, amount, label, confidence, rationale?`

SQL schema mẫu: [db/init.sql](db/init.sql)

## 4) Local dev (optional)

Nếu muốn dựng nhanh Postgres/Redis local để test (không bắt buộc):
- `docker compose --profile local-deps up -d postgres redis`
- `docker compose up -d ai-service`
