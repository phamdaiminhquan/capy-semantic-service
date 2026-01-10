from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from capy_teacher.database import get_db, init_db
from capy_teacher.crud import DatasetManager
from capy_teacher.export import export_train_csv
from capy_teacher.text_pipeline import teacher_preprocess
from capy_teacher.predictor import get_text_classifier
from capy_teacher.models import VALID_LABELS
import os

app = FastAPI(title="Capy Teacher API", version="1.0.0")

# Mở CORS cho tất cả các nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    # Load model once at startup (warm-up). If missing, fail fast.
    get_text_classifier()

# Request/Response models
class CreateExampleRequest(BaseModel):
    raw_text: str
    label: str

class UpdateExampleRequest(BaseModel):
    label: Optional[str] = None
    is_active: Optional[bool] = None

class ExampleResponse(BaseModel):
    id: int
    raw_text: str
    normalized_text: str
    label: str
    amount: Optional[float]
    is_active: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class PreprocessResponse(BaseModel):
    raw_text: str
    normalized_text: str
    amount: Optional[float]

class StatsResponse(BaseModel):
    total: int
    label_counts: dict

class PredictRequest(BaseModel):
    raw_text: str
    top_k: int = 3

class PredictResponse(BaseModel):
    raw_text: str
    normalized_text: str
    amount: Optional[float]
    label: str
    score: float
    top_k: List[dict]

@app.post("/examples", response_model=ExampleResponse)
def create_example(
    request: CreateExampleRequest,
    db: Session = Depends(get_db)
):
    """Tạo example mới. Tự động preprocess raw_text."""
    manager = DatasetManager(db)
    try:
        example = manager.create_example(request.raw_text, request.label)
        return ExampleResponse(
            id=example.id,
            raw_text=example.raw_text,
            normalized_text=example.normalized_text,
            label=example.label,
            amount=example.amount,
            is_active=example.is_active,
            created_at=example.created_at.isoformat(),
            updated_at=example.updated_at.isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/examples", response_model=List[ExampleResponse])
def list_examples(
    label: Optional[str] = Query(None),
    is_active: bool = Query(True),
    limit: int = Query(5000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List examples với filter."""
    manager = DatasetManager(db)
    examples = manager.list_examples(label=label, is_active=is_active, limit=limit, offset=offset)
    return [
        ExampleResponse(
            id=ex.id,
            raw_text=ex.raw_text,
            normalized_text=ex.normalized_text,
            label=ex.label,
            amount=ex.amount,
            is_active=ex.is_active,
            created_at=ex.created_at.isoformat(),
            updated_at=ex.updated_at.isoformat()
        )
        for ex in examples
    ]

@app.get("/examples/{example_id}", response_model=ExampleResponse)
def get_example(example_id: int, db: Session = Depends(get_db)):
    """Lấy example theo ID."""
    manager = DatasetManager(db)
    example = manager.get_example_by_id(example_id)
    if not example:
        raise HTTPException(status_code=404, detail=f"Example {example_id} not found")
    return ExampleResponse(
        id=example.id,
        raw_text=example.raw_text,
        normalized_text=example.normalized_text,
        label=example.label,
        amount=example.amount,
        is_active=example.is_active,
        created_at=example.created_at.isoformat(),
        updated_at=example.updated_at.isoformat()
    )

@app.patch("/examples/{example_id}", response_model=ExampleResponse)
def update_example(
    example_id: int,
    request: UpdateExampleRequest,
    db: Session = Depends(get_db)
):
    """Cập nhật example. Chỉ cho sửa label và is_active."""
    manager = DatasetManager(db)
    try:
        example = manager.update_example(
            example_id,
            label=request.label,
            is_active=request.is_active
        )
        return ExampleResponse(
            id=example.id,
            raw_text=example.raw_text,
            normalized_text=example.normalized_text,
            label=example.label,
            amount=example.amount,
            is_active=example.is_active,
            created_at=example.created_at.isoformat(),
            updated_at=example.updated_at.isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/examples/{example_id}")
def delete_example(example_id: int, db: Session = Depends(get_db)):
    """Soft delete example."""
    manager = DatasetManager(db)
    try:
        manager.soft_delete(example_id)
        return {"message": f"Example {example_id} deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Thống kê số lượng mẫu mỗi label."""
    manager = DatasetManager(db)
    stats = manager.get_label_stats()
    total = sum(stats.values())
    return StatsResponse(total=total, label_counts=stats)

@app.post("/export")
def export_dataset(output_path: str = Query("train.csv"), db: Session = Depends(get_db)):
    """Export train.csv."""
    try:
        result = export_train_csv(db, output_path)
        return {
            "message": "Export successful",
            "output_path": result["output_path"],
            "total": result["total"],
            "label_counts": result["label_counts"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess_endpoint(raw_text: str):
    """Preprocess text (utility endpoint)."""
    result = teacher_preprocess(raw_text)
    return PreprocessResponse(
        raw_text=result["raw_text"],
        normalized_text=result["normalized_text"],
        amount=result["amount"]
    )

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """Predict label from raw_text using fine-tuned PhoBERT (includes preprocess pipeline)."""
    try:
        clf = get_text_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    result = clf.predict(request.raw_text, top_k=request.top_k)
    return PredictResponse(**result)

@app.get("/labels")
def get_labels():
    """Danh sách labels hợp lệ."""
    return {"labels": VALID_LABELS}

@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "message": "Capy Teacher API"}

