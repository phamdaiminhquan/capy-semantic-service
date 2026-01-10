from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime
from .database import Base

VALID_LABELS = [
    "chi_an_uong",
    "chi_di_chuyen",
    "chi_sinh_hoat",
    "chi_mua_sam",
    "chi_giai_tri",
    "chi_phat_trien",
    "chi_xa_hoi",
    "chi_khac",
    "thu_luong",
    "thu_thuong",
    "thu_dau_tu",
    "thu_khac",
]

class TrainingExample(Base):
    __tablename__ = "training_examples"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_text = Column(String, nullable=False)
    normalized_text = Column(String, nullable=False)
    label = Column(String, nullable=False)
    amount = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

