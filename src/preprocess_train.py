# src/preprocess_train.py
from __future__ import annotations
import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import dump

from .config import OUTPUT_PARQUET, TARGET_ID

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "textclf_svc.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1) Load the sampled data
    df = pd.read_parquet(OUTPUT_PARQUET)
    # Expect columns: text, label, target
    X = df["text"].astype(str).values
    y = df["target"].values

    # 2) Train/valid split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=13, stratify=y
    )

    # 3) Build a fast pipeline
    pipe = Pipeline([
        ("hash", HashingVectorizer(
            n_features=2**20,      # ~1M features, fast & memory-safe
            alternate_sign=False,  # makes features non-negative (more stable)
            stop_words="english",
            ngram_range=(1, 2),    # unigrams + bigrams helps accuracy
            lowercase=True,
        )),
        ("clf", LinearSVC(C=1.0, random_state=13))
    ])

    # 4) Train
    pipe.fit(X_tr, y_tr)

    # 5) Validate
    y_pr = pipe.predict(X_va)
    acc = accuracy_score(y_va, y_pr)
    f1m = f1_score(y_va, y_pr, average="macro")
    print(f"Validation accuracy: {acc:.4f} | macro-F1: {f1m:.4f}")
    print("\nClassification report:\n",
          classification_report(y_va, y_pr, digits=4))

    # 6) Persist model + metadata
    dump(pipe, MODEL_PATH)
    id_to_name = {v: k for k, v in TARGET_ID.items()}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"id_to_name": id_to_name}, f, indent=2)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved meta:  {META_PATH}")

if __name__ == "__main__":
    main()
