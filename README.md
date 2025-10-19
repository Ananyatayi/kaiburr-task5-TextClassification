# 🧠 Task 5 — Text Classification on Consumer Complaint Dataset

This project performs **multi-class text classification** on the [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database) to categorize complaints into four classes:

| ID | Category |
|----|-----------|
| 0  | Credit reporting / repair / other |
| 1  | Debt collection |
| 2  | Consumer Loan |
| 3  | Mortgage |

---


---

## ⚙️ Setup

python -m venv .venv
.venv\Scripts\activate       # (Windows)
pip install -r requirements.txt

Steps to Run
1️⃣ Exploratory Data Analysis & Sampling
python -m src.eda_sample


Cleans and downsamples the large CSV.

Saves → data/processed_sample.parquet.

2️⃣ Model Training
python -m src.preprocess_train


Uses HashingVectorizer + LinearSVC.

Prints accuracy & F1, saves model in /models.

3️⃣ Evaluation
python -m src.evaluate


Generates:

classification_report.txt

confusion_matrix.png

4️⃣ Prediction
python -m src.predict "My mortgage servicer mishandled escrow payments"


→ Predicted Category: Mortgage



🏁 Output Highlights

Validation Accuracy: ≈ 90–94 %

Macro F1: ≈ 0.89–0.93

Confusion Matrix: saved in /outputs/confusion_matrix.png

🧾 Summary

A lightweight and reproducible text-classification pipeline:

Efficient sampling for large data.

Fast feature extraction (Hashing Vectorizer).

Reliable SVM baseline with evaluation and predict utility.
