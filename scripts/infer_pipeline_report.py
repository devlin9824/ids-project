#!/usr/bin/env python3
import json, joblib, numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

META = joblib.load("models/rf_binary.joblib")
clf = META['model']; num_cols = META['num_cols']; label_col = META['label_col']

with open("models/thresholds.json") as f:
    th = json.load(f)
T1, T2 = th['T1'], th['T2']
sc = np.load("models/ae_scaler.npy", allow_pickle=True).item()
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32))
        self.dec = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,d), nn.Sigmoid())
    def forward(self,x): return self.dec(self.enc(x))

dim = len(num_cols)
ae = AE(dim)
ae.load_state_dict(torch.load("models/ae_model.pth", map_location='cpu'))
ae.eval(); torch.set_num_threads(1)

TEST = "data/splits/test.csv"
y_true_all=[]; y_pred_all=[]

for ch in pd.read_csv(TEST, usecols=num_cols+[label_col], chunksize=20000):
    y_true = ch[label_col].apply(lambda x: 1 if str(x).strip().lower()=='attack' else 0).astype('int8').values
    X = ch[num_cols].fillna(0).astype('float32').values

    # RF
    probs = clf.predict_proba(X)[:,1]
    pred = (probs > T1).astype('int8')

    # AE recheck for those predicted attack
    idx = np.where(pred==1)[0]
    if len(idx)>0:
        X_sel = X[idx]
        Xs = (X_sel - np.array(sc['min'])) / np.array(sc['scale'])
        Xt = torch.from_numpy(Xs).float()
        with torch.no_grad():
            rec = ae(Xt)
            mse = ((rec - Xt)**2).mean(dim=1).numpy()
        # reclassify to benign if mse <= T2
        pred[idx[mse<=T2]] = 0

    y_true_all.append(y_true)
    y_pred_all.append(pred)

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print("Report:\n", classification_report(y_true_all, y_pred_all, digits=4))
print("Confusion:\n", confusion_matrix(y_true_all, y_pred_all))
