import re

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import TrainingExample
from .text_pipeline import teacher_preprocess
from typing import List, Optional


_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _validate_label(label: str) -> str:
    if label is None:
        raise ValueError("label is required")
    label = str(label).strip()
    if not label:
        raise ValueError("label must not be empty")
    if not _LABEL_RE.match(label):
        raise ValueError(
            "Invalid label format. Use letters/digits and characters '_.-' only (max 64), no spaces."
        )
    return label

class DatasetManager:
    def __init__(self, db: Session):
        self.db = db
    
    def create_example(self, raw_text: str, label: str, *, commit: bool = True) -> TrainingExample:
        """Tạo example mới, tự động preprocess.

        Args:
            commit: Nếu False, chỉ add vào session (caller tự commit).
        """
        label = _validate_label(label)
        
        # Tự động preprocess (canonicalized)
        preprocessed = teacher_preprocess(raw_text)
        
        example = TrainingExample(
            raw_text=preprocessed["raw_text"],
            normalized_text=preprocessed["normalized_text"],
            label=label,
            amount=preprocessed["amount"]
        )
        
        self.db.add(example)
        if commit:
            self.db.commit()
            self.db.refresh(example)
        return example
    
    def list_examples(
        self, 
        label: Optional[str] = None,
        is_active: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[TrainingExample]:
        """List examples với filter"""
        query = self.db.query(TrainingExample).filter(
            TrainingExample.is_active == is_active
        )
        
        if label:
            query = query.filter(TrainingExample.label == label)
        
        return query.offset(offset).limit(limit).all()
    
    def update_example(self, example_id: int, label: Optional[str] = None, is_active: Optional[bool] = None):
        """Chỉ cho sửa label và is_active"""
        example = self.db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
        if not example:
            raise ValueError(f"Example {example_id} not found")
        
        if label is not None:
            example.label = _validate_label(label)
        
        if is_active is not None:
            example.is_active = is_active
        
        self.db.commit()
        self.db.refresh(example)
        return example
    
    def soft_delete(self, example_id: int):
        """Soft delete bằng is_active"""
        return self.update_example(example_id, is_active=False)
    
    def get_example_by_id(self, example_id: int) -> Optional[TrainingExample]:
        """Lấy example theo ID"""
        return self.db.query(TrainingExample).filter(TrainingExample.id == example_id).first()
    
    def get_label_stats(self) -> dict:
        """Kiểm tra số lượng mẫu mỗi label (dynamic from DB)."""
        rows = (
            self.db.query(TrainingExample.label, func.count(TrainingExample.id))
            .filter(TrainingExample.is_active == True)
            .group_by(TrainingExample.label)
            .all()
        )
        return {label: int(count) for label, count in rows}

