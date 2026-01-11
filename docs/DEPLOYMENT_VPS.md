# Production deployment (4GB VPS)

This repo contains both training + inference. For a small VPS, deploy **inference-only**.

## Git strategy (keep repo lightweight)

- Heavy folders are ignored by default via `.gitignore`:
  - `dataset/`, `training/`, `worker/`, and most of `artifacts/`
- Inference artifacts are whitelisted:
  - `artifacts/onnx/` (ONNX model + tokenizer files)

### Git LFS (recommended)

`artifacts/onnx/model_int8.onnx` is >100MB, so GitHub will reject it without LFS.

1) Install Git LFS on your machine and VPS

2) Track ONNX files:

```bash
git lfs install
git lfs track "*.onnx"
```

This repo already includes `.gitattributes` for `*.onnx`.

> If you prefer **not** to use Git LFS: remove `artifacts/onnx/` from Git and copy it to the VPS via SCP/rsync, then set `MODEL_DIR=/app/artifacts/onnx`.

## Docker strategy (inference-only)

Use:
- `Dockerfile.deploy`
- `docker-compose.deploy.yml`

This copies only:
- `capy_teacher/`, `shared/`, `scripts/`
- `api.py`, `main.py`
- `artifacts/onnx/`

It does NOT copy: `training/`, `dataset/`, `worker/`, `artifacts/phobert_textcls/`.

## Build & run

```bash
docker compose -f docker-compose.deploy.yml up -d --build
```

API will be available at:
- `http://<VPS_IP>:8000/docs`

## Notes

- `requirements-api.txt` is inference-only: **no torch**.
- The `/export` endpoint is disabled in deploy builds (returns HTTP 501) because `worker/` is excluded.
