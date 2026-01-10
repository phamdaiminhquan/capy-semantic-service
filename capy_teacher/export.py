import csv
from sqlalchemy.orm import Session
from .models import TrainingExample

def export_train_csv(db: Session, output_path: str = "train.csv") -> dict:
    """
    Export train.csv
    Chỉ export normalized_text và label
    Format: text,label
    """
    examples = db.query(TrainingExample).filter(
        TrainingExample.is_active == True
    ).all()
    
    if not examples:
        raise ValueError("No active examples to export")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        
        for ex in examples:
            writer.writerow([ex.normalized_text, ex.label])
    
    # Stats
    label_counts = {}
    for ex in examples:
        label_counts[ex.label] = label_counts.get(ex.label, 0) + 1
    
    return {
        "total": len(examples),
        "output_path": output_path,
        "label_counts": label_counts
    }

