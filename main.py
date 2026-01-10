from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

raw_text = "ăn sáng bánh mì 50k"

# Word segmentation
text = ViTokenizer.tokenize(raw_text)
print("Segmented:", text)

# Load PhoBERT
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.last_hidden_state.mean(dim=1)
print("Embedding shape:", embedding.shape)
