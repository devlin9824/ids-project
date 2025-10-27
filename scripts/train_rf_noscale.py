#!/usr/bin/env python3
import os, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

TRAIN = "data/splits/train.csv"
VAL   = "data/splits/val.csv"
OUT   = "models/rf_binary.joblib"
os.makedirs("models", exist_ok=True)

# detect label col + numeric cols từ sample
sample = pd.read_csv(TRAIN, nrows=2000)
label_col = next((c for c in sample.columns if 'binary' in c.lower() or c.lower()=='label'), None)
assert label_col, "Không tìm được cột label"
num_cols = sample.select_dtypes(include=['number']).columns.tolist()

def load_xy(path):
    df = pd.read_csv(path, usecols=num_cols+[label_col])
    y = df[label_col].apply(lambda x: 1 if str(x).strip().lower()=='attack' else 0).astype('int8')
    X = df[num_cols].fillna(0).astype('float32')
    return X, y

print("Loading train...")
X_train, y_train = load_xy(TRAIN)
print("Train shape:", X_train.shape, "attack ratio:", y_train.mean())

print("Loading val...")
X_val, y_val = load_xy(VAL)
print("Val shape:", X_val.shape, "attack ratio:", y_val.mean())

# RF không cần scale -> tiết kiệm RAM
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=25,
    n_jobs=1,
    class_weight=None,   # nếu bạn giữ imbalance, có thể set 'balanced'
    random_state=42
)

print("Training RF...")
clf.fit(X_train, y_train)

pred = clf.predict(X_val)
print("Validation report:\n", classification_report(y_val, pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_val, pred))

joblib.dump({'model': clf, 'num_cols': num_cols, 'label_col': label_col}, OUT, compress=3)
print("Saved:", OUT)
