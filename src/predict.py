# src/predict.py
from __future__ import annotations
import sys, json
from joblib import load
from .config import TARGET_ID

MODEL_PATH = "models/textclf_svc.joblib"
META_PATH  = "models/meta.json"

def predict_text(text: str) -> str:
    model = load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    id_to_name = meta["id_to_name"]
    y = model.predict([text])[0]
    return id_to_name[str(y)] if isinstance(id_to_name, dict) and str(y) in id_to_name else id_to_name[y]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict \"your complaint narrative here\"")
        sys.exit(1)
    text = sys.argv[1]
    label = predict_text(text)
    print(label)
