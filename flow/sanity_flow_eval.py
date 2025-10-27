import json, pandas as pd
from joblib import load
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

rf = load("models/flow/rf_binary.joblib")
cols = json.load(open("models/flow/selected_cols.json"))
df = pd.read_csv("data/test_flow.csv", low_memory=False)

# bắt cột nhãn
label_col = next(c for c in ["__binary_label__","Label","label","y","target"] if c in df.columns)
X = df[cols].values  # dùng đúng thứ tự cols
y = (df[label_col].astype(str).str.lower().isin(["1","attack","anomaly","malicious","bad"])).astype(int).values

# đọc T1
T1 = json.load(open("models/flow/thresholds.json")).get("T1", 0.5)
p = rf.predict_proba(X)[:,1]
yhat = (p >= T1).astype(int)

pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
cm = confusion_matrix(y, yhat, labels=[0,1]).tolist()
print({"T1":T1, "precision":pr, "recall":rc, "f1":f1, "cm":cm})
