import pandas as pd, os
from collections import defaultdict
from .config import *

def detect(df):
    t = next((c for c in TEXT_COLS if c in df.columns), None)
    l = next((c for c in LABEL_COLS if c in df.columns), None)
    return t, l

def normalize(label):
    if not isinstance(label, str): return None
    lbl = label.strip()
    if lbl in MAP_CANONICAL: return MAP_CANONICAL[lbl]
    low = lbl.lower()
    if "credit report" in low or "credit repair" in low: return "Credit reporting, repair, or other"
    if "debt" in low: return "Debt collection"
    if "loan" in low: return "Consumer Loan"
    if "mortgage" in low: return "Mortgage"
    return None

def main():
    os.makedirs("data", exist_ok=True)
    buckets = defaultdict(list)
    txt_col = lbl_col = None

    for chunk in pd.read_csv(RAW_CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        if txt_col is None or lbl_col is None:
            txt_col, lbl_col = detect(chunk)
            if not txt_col or not lbl_col:
                raise ValueError(f"Missing required columns. Got {chunk.columns[:10]}")
        sub = chunk[[txt_col, lbl_col]].rename(columns={txt_col:"text", lbl_col:"label"})
        sub = sub.dropna()
        sub["label"] = sub["label"].apply(normalize)
        sub = sub.dropna()
        for k,g in sub.groupby("label"):
            if k not in TARGET_ID: continue
            need = SAMPLE_PER_CLASS - sum(len(x) for x in buckets[k])
            if need>0: buckets[k].append(g.sample(min(len(g), need)))
        if all(sum(len(x) for x in v)>=SAMPLE_PER_CLASS for v in buckets.values() if v): break

    frames=[]
    for k,v in buckets.items():
        if not v: continue
        df=pd.concat(v).sample(SAMPLE_PER_CLASS, replace=False, random_state=13)
        df["target"]=TARGET_ID[k]
        frames.append(df)

    out=pd.concat(frames).sample(frac=1.0, random_state=13).reset_index(drop=True)
    out.to_parquet(OUTPUT_PARQUET, index=False)
    print(out["label"].value_counts(), "\nSaved:", OUTPUT_PARQUET)

if __name__=="__main__":
    main()
