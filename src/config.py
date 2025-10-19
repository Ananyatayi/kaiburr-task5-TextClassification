RAW_CSV_PATH = r"D:\Datasets\complaints.csv"   # big dataset path

# candidate columns (auto-detect)
TEXT_COLS = [
    "Consumer complaint narrative",
    "consumer_complaint_narrative",
    "complaint_what_happened",
]
LABEL_COLS = ["Product", "product"]

# mapping to 4 target classes
MAP_CANONICAL = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting, repair, or other",
    "Credit reporting, credit repair, or other personal consumer reports":
        "Credit reporting, repair, or other",
    "Debt collection": "Debt collection",
    "Consumer Loan": "Consumer Loan",
    "Consumer loan": "Consumer Loan",
    "Mortgage": "Mortgage",
}
TARGET_ID = {
    "Credit reporting, repair, or other": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3,
}

CHUNK_SIZE = 200_000
SAMPLE_PER_CLASS = 8000       # ~32 k rows total
OUTPUT_PARQUET = "data/processed_sample.parquet"
