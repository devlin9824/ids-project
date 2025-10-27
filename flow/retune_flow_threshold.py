#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retune ngưỡng T1 cho RandomForest flow theo yêu cầu min_precision.
- Dùng selected_cols.json nếu có (đúng cột/đúng thứ tự)
- Nếu không có: fallback cắt đúng n_features_in_ cột đầu (chỉ để chạy tạm)
- Cập nhật models/flow/thresholds.json với T1 mới
"""

import os, json, argparse, numpy as np, pandas as pd
from joblib import load
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

LABEL_ALIASES = ["__binary_label__", "Label", "label", "y", "target"]

def metrics(y_true, y_pred):
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1), "cm": cm}

def coerce_labels(series):
    return series.astype(int).values if series.dtype != object else \
        series.astype(str).str.lower().map(lambda v: 1 if v in ['1','attack','anomaly','malicious','bad'] else 0).values

def load_X_y(csv_path, sel_cols=None, need=None):
    df = pd.read_csv(csv_path, low_memory=False)
    # label
    label_col = None
    for c in LABEL_ALIASES:
        if c in df.columns: label_col = c; break
    if label_col is None:
        raise SystemExit("CSV không có cột nhãn (Label/label/y/target/__binary_label__).")
    y = coerce_labels(df[label_col])

    # numeric features (loại nhãn)
    df_num = df.select_dtypes(include=[np.number]).copy()
    if label_col in df_num.columns:
        df_num = df_num.drop(columns=[label_col])

    if sel_cols is not None:
        miss = [c for c in sel_cols if c not in df_num.columns]
        if miss:
            raise SystemExit(f"Thiếu cột đã train: {miss[:10]} ...")
        X = df_num[sel_cols].values
    else:
        if need is None: need = 30
        if df_num.shape[1] < need:
            raise SystemExit(f"Input có {df_num.shape[1]} cột numeric, cần {need}.")
        X = df_num.iloc[:, :need].values
    return X, y

def main(a):
    flow_dir = a.model_dir
    rf = load(os.path.join(flow_dir, "rf_binary.joblib"))

    sel_cols = None
    sel_path = os.path.join(flow_dir, "selected_cols.json")
    if os.path.exists(sel_path):
        with open(sel_path,"r") as f:
            sel_cols = json.load(f)
    need = int(getattr(rf, "n_features_in_", 30))

    X, y = load_X_y(a.csv, sel_cols, need)
    p = rf.predict_proba(X)[:,1]

    Ts = np.linspace(0.01, 0.99, 197)
    best_ok = None; best_all = None
    for T in Ts:
        yhat = (p>=T).astype(int)
        m = metrics(y, yhat)
        stat = {"T1": float(T), **m}
        if m["precision"] >= a.min_precision:
            if (best_ok is None) or (m["recall"] > best_ok["recall"] or (m["recall"]==best_ok["recall"] and m["f1"] > best_ok["f1"])):
                best_ok = stat
        if (best_all is None) or (m["f1"] > best_all["f1"]):
            best_all = stat

    chosen = best_ok if best_ok is not None else best_all
    th_path = os.path.join(flow_dir, "thresholds.json")
    try:
        with open(th_path, "r") as f:
            th = json.load(f)
    except Exception:
        th = {}
    th["T1"] = float(chosen["T1"])
    with open(th_path, "w") as f:
        json.dump(th, f, indent=2)

    print(json.dumps({"chosen": chosen, "saved_to": th_path}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--model_dir", type=str, default="./models/flow")
    ap.add_argument("--min_precision", type=float, default=0.99)
    args = ap.parse_args()
    main(args)
