# src/evaluate.py
from __future__ import annotations
import os, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
from .config import OUTPUT_PARQUET, TARGET_ID

OUT_DIR = "outputs"
MODEL_PATH = "models/textclf_svc.joblib"
META_PATH  = "models/meta.json"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(OUTPUT_PARQUET)
    X = df["text"].astype(str).values
    y = df["target"].values

    # same split recipe as training for a fast report snapshot
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=13, stratify=y
    )

    model = load(MODEL_PATH)
    y_pr = model.predict(X_va)

    # report
    id_to_name = {v: k for k, v in TARGET_ID.items()}
    target_names = [id_to_name[i] for i in sorted(id_to_name)]
    report = classification_report(y_va, y_pr, target_names=target_names, digits=4)

    rep_path = os.path.join(OUT_DIR, "classification_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved report -> {rep_path}")

    # confusion matrix plot
    cm = confusion_matrix(y_va, y_pr, labels=sorted(id_to_name))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Validation)")
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(range(len(target_names)))
    ax.set_yticklabels(range(len(target_names)))
    plt.xlabel("Predicted (IDs)")
    plt.ylabel("True (IDs)")

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix -> {fig_path}")

if __name__ == "__main__":
    main()
