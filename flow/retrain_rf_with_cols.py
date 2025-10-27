#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain RF trên các file train/val/test đã chuẩn hóa (train_flow.csv/val_flow.csv/test_flow.csv),
chọn TOP-K cột bằng mutual information, lưu:
- rf_binary.joblib
- selected_cols.json (danh sách 30 cột đúng thứ tự)
- thresholds.json (T1 sơ bộ lấy theo best F1 trên val)
"""

import os, json, argparse, numpy as np, pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

LABEL_ALIASES = ["__binary_label__", "Label", "label", "y", "target"]

def load_split(data_dir):
    paths = {k: os.path.join(data_dir, f"{k}_flow.csv") for k in ["train","val","test"]}
    for k,p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"Missing {k} file: {p}")
    def _read(p):
        df = pd.read_csv(p, low_memory=False)
        label_col = None
        for c in LABEL_ALIASES:
            if c in df.columns:
                label_col = c; break
        if label_col is None:
            raise SystemExit(f"File {p} không có cột nhãn hợp lệ.")
        y = df[label_col].astype(int).values
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        return X, y
    Xtr, ytr = _read(paths["train"])
    Xva, yva = _read(paths["val"])
    Xte, yte = _read(paths["test"])
    return (Xtr,ytr), (Xva,yva), (Xte,yte)

def select_topk_by_mi(X: pd.DataFrame, y: np.ndarray, k: int):
    mi = mutual_info_classif(X.values, y, discrete_features=False, random_state=1337)
    idx = np.argsort(mi)[::-1][:k]
    cols = list(X.columns[idx])
    return cols, mi[idx].tolist()

def metrics(y_true, y_pred):
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    return {"precision": float(pr), "recall": float(rc), "f1": float(f1), "cm": cm}

def best_f1_threshold(p, y):
    best=(0.5, {"precision":0,"recall":0,"f1":0})
    for T in np.linspace(0.05, 0.95, 181):
        yhat=(p>=T).astype(int)
        m=metrics(y, yhat)
        if m["f1"]>best[1]["f1"]:
            best=(T,m)
    return best

def main(args):
    (Xtr,ytr), (Xva,yva), (Xte,yte) = load_split(args.data_dir)

    # chọn top-k từ train
    cols, _ = select_topk_by_mi(Xtr, ytr, args.topk)
    Xtr_k, Xva_k, Xte_k = Xtr[cols], Xva[cols], Xte[cols]

    # train RF
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        n_jobs=1,
        class_weight="balanced_subsample",
        random_state=args.random_state
    )
    rf.fit(Xtr_k, ytr)

    # đánh giá & tìm T1 sơ bộ
    p_va = rf.predict_proba(Xva_k)[:,1]
    p_te = rf.predict_proba(Xte_k)[:,1]
    T1, m_va = best_f1_threshold(p_va, yva)
    yhat_te = (p_te>=T1).astype(int)
    m_te = metrics(yte, yhat_te)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    dump(rf, os.path.join(out_dir, "rf_binary.joblib"))
    with open(os.path.join(out_dir, "selected_cols.json"), "w") as f:
        json.dump(cols, f, indent=2)
    with open(os.path.join(out_dir, "thresholds.json"), "w") as f:
        json.dump({"T1": float(T1)}, f, indent=2)

    print(json.dumps({
        "saved_to": out_dir,
        "topk": cols,
        "val_at_T1": {"T1": float(T1), **m_va},
        "test_at_T1": m_te
    }, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./models/flow")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--random_state", type=int, default=1337)
    args = ap.parse_args()
    main(args)
