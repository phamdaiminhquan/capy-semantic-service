
import csv
from pathlib import Path

def validate(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2): # Start 2 specific to header at 1
            text = row.get("text", "")
            label = row.get("label", "")
            
            # Heuristic: label shouldn't have spaces or <AMOUNT> usually
            if "<AMOUNT>" in label or " " in label:
                print(f"Line {i}: Possible malformed row.")
                print(f"  Text: {text}")
                print(f"  Label: {label}")
                
            if len(row) > 2:
                 print(f"Line {i}: Extra columns found: {row}")

validate("dataset/train.csv")
