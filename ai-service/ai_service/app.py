from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.vi_processor import normalize

from .onnx_model import OnnxTextClassifier


app = FastAPI(title="capy-ai-service", version="1.0.0")


class AnalyzeTokensRequest(BaseModel):
    raw_text: str = Field(..., min_length=1)


class AnalyzeTokensResponse(BaseModel):
    raw_text: str
    normalized_text: str
    token_count: int
    cache_key: str
    amount: float | None


class PredictRequest(BaseModel):
    raw_text: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class PredictResponse(BaseModel):
    raw_text: str
    normalized_text: str
    token_count: int
    cache_key: str
    amount: float | None
    label: str
    score: float
    top_k: list[dict]


@lru_cache(maxsize=1)
def _get_model() -> OnnxTextClassifier:
    return OnnxTextClassifier.from_env()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-tokens", response_model=AnalyzeTokensResponse)
def analyze_tokens(req: AnalyzeTokensRequest):
    res = normalize(req.raw_text)
    return AnalyzeTokensResponse(
        raw_text=res.raw_text,
        normalized_text=res.normalized_text,
        token_count=res.token_count,
        cache_key=res.cache_key,
        amount=res.amount,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = _get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {e}")

    res = normalize(req.raw_text)

    try:
        pred = model.predict(res.normalized_text, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        raw_text=res.raw_text,
        normalized_text=res.normalized_text,
        token_count=res.token_count,
        cache_key=res.cache_key,
        amount=res.amount,
        label=pred["label"],
        score=float(pred["score"]),
        top_k=pred["top_k"],
    )
