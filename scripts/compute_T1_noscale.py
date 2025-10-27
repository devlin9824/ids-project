#!/usr/bin/env python3
import pandas as pd, numpy as np, joblib, json

MODEL = "models/rf_binary.joblib"
VAL   = "data/splits/val.csv"
m = joblib.load(MODEL)
clf = m['model']; num_cols = m['num_cols']; label_col = m['label_col']

# lấy benign trên val
usecols = num_cols + [label_col]
val = pd.read_csv(VAL, usecols=usecols)
benign = val[val[label_col].str.lower()=='benign']
Xb = benign[num_cols].fillna(0).astype('float32')

probs = clf.predict_proba(Xb)[:,1]
T1 = float(np.percentile(probs, 90))   # 90th percentile
print("T1 =", T1)

with open("models/thresholds.json","w") as f:
    json.dump({"T1": T1}, f)
print("Saved models/thresholds.json")
