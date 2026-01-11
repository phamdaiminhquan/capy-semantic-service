# Install dependencies
pip install -r requirements.txt

# Run the app (development mode)
uvicorn app:app --reload

# training (bắt đầu fine-tune)
python -m training.train_textcls --train_csv dataset/train.csv --output_dir artifacts/ phobert_textcls

# predict
python -m training.predict_textcls --model_dir artifacts/phobert_textcls --text
# ví dụ
python -m training.predict_textcls --model_dir artifacts/phobert_textcls --text "đi ăn bún chả 35k"

# Run the API with the fine-tuned model
$env:REQUIRE_MODEL="1"; $env:MODEL_DIR="artifacts/phobert_textcls"; python -m uvicorn api:app --reload --port 8000
$env:MODEL_DIR="artifacts/onnx"; $env:REQUIRE_MODEL="1"; python -m uvicorn api:app --reload --port 8000