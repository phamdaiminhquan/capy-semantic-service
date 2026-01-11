import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Import preprocess from project
import sys
import os
sys.path.append(os.getcwd())
from capy_teacher.text_pipeline import AMOUNT_TOKEN

def export_onnx(model_dir: str, output_dir: str, quantize: bool = False):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Dummy input for export tracing
    dummy_text = "tôi muốn mua một chiếc bánh mì"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    onnx_path = output_dir / "model.onnx"
    
    print(f"Exporting to {onnx_path}...")
    
    # Dynamic axes allow variable sequence lengths
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print("✅ ONNX model exported.")

    # Save tokenizer config
    tokenizer.save_pretrained(output_dir)
    # Also copy label_map
    import shutil
    if (model_dir / "label_map.json").exists():
        shutil.copy(model_dir / "label_map.json", output_dir / "label_map.json")

    # Quantization (Optional)
    if quantize:
        print("Quantizing model to INT8...")
        quantized_path = output_dir / "model_int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QUInt8
        )
        print(f"✅ Quantized model saved to {quantized_path} (Smaller & Faster)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="artifacts/phobert_textcls")
    parser.add_argument("--output_dir", default="artifacts/onnx")
    parser.add_argument("--quantize", action="store_true", help="Quantize model to INT8")
    args = parser.parse_args()
    
    export_onnx(args.model_dir, args.output_dir, args.quantize)
