#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, numpy as np, pandas as pd, torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch.nn as nn

LABEL_ALIASES = ["__binary_label__", "Label", "label", "y", "target"]

def _coerce_labels(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.lower()
    y=[]
    for v in s:
        if v in {"0","benign","normal","legit","legitimate","ok"}: y.append(0)
        elif v in {"1","attack","anomaly","anomalous","malicious","bad"}: y.append(1)
        else:
            try: y.append(1 if float(v)!=0.0 else 0)
            except: y.append(1)
    return np.array(y, dtype=int)

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, max(16, d//2)), nn.ReLU(),
            nn.Linear(max(16, d//2), max(8, d//4)), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(max(8, d//4), max(16, d//2)), nn.ReLU(),
            nn.Linear(max(16, d//2), d),
        )
    def forward(self, x): return self.dec(self.enc(x))

def load_cols():
    return json.load(open("models/flow/selected_cols.json"))

def load_T2():
    th = json.load(open("models/flow/thresholds.json"))
    return float(th.get("T2", 0.0))

def load_ae(path, d):
    state = torch.load(path, map_location="cpu")
    mapped={}
    for k,v in state.items():
        nk=k
        if nk.startswith("encoder."): nk="enc."+nk[len("encoder."):]
        if nk.startswith("decoder."): nk="dec."+nk[len("decoder."):]
        mapped[nk]=v
    ae = AE(d); ae.load_state_dict(mapped, strict=False); ae.eval()
    return ae

def main():
    cols = load_cols()
    T2 = load_T2()
    df = pd.read_csv("data/test_flow.csv", low_memory=False)
    label_col = next((c for c in LABEL_ALIASES if c in df.columns), None)
    if label_col is None: raise SystemExit("test_flow.csv thiếu cột nhãn")
    y_true = _coerce_labels(df[label_col])
    X = df[cols].values.astype(np.float32)

    ae = load_ae("models/flow/ae_normal.pth", d=X.shape[1])
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        recon = ae(xt).numpy()
    mse = ((X - recon)**2).mean(axis=1)
    y_pred = (mse > T2).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    print(json.dumps({"T2": T2, "precision": float(pr), "recall": float(rc), "f1": float(f1), "cm": cm}, indent=2))

if __name__ == "__main__":
    main()
