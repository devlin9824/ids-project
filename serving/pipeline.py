# serving/pipeline.py
import json, joblib, numpy as np, torch, torch.nn as nn

class AENet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32))
        self.dec = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,d), nn.Sigmoid())
    def forward(self,x): return self.dec(self.enc(x))

class RFAEPipeline:
    def __init__(self,
                 rf_meta_path="models/rf_binary.joblib",
                 ae_model_path="models/ae_model.pth",
                 ae_scaler_path="models/ae_scaler.npy",
                 thresholds_path="models/thresholds.json"):
        self.meta = joblib.load(rf_meta_path)
        self.clf = self.meta['model']
        self.num_cols = self.meta['num_cols']
        with open(thresholds_path) as f:
            th = json.load(f)
        self.T1 = float(th['T1'])
        self.T2 = float(th['T2'])
        sc = np.load(ae_scaler_path, allow_pickle=True).item()
        self.MIN = np.array(sc['min'], dtype=np.float32)
        self.SCALE = np.array(sc['scale'], dtype=np.float32)
        self.dim = len(self.num_cols)
        self.ae = AENet(self.dim)
        self.ae.load_state_dict(torch.load(ae_model_path, map_location='cpu'))
        self.ae.eval()
        torch.set_num_threads(1)

    def _scale_ae(self, X):
        Xs = (X.astype(np.float32) - self.MIN) / self.SCALE
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(Xs, 0.0, 1.0).astype(np.float32)

    def predict_batch(self, X_np):
        """X_np: ndarray shape [N, dim] (numeric features in the same order as self.num_cols)"""
        probs = self.clf.predict_proba(X_np)[:,1]
        pred = (probs > self.T1).astype(np.int8)
        idx = np.where(pred==1)[0]
        if len(idx) > 0:
            X_sel_scaled = self._scale_ae(X_np[idx])
            Xt = torch.from_numpy(X_sel_scaled)
            with torch.no_grad():
                rec = self.ae(Xt)
                mse = ((rec - Xt)**2).mean(dim=1).numpy()
            # reclassify to benign if mse <= T2
            pred[idx[mse <= self.T2]] = 0
        return probs, pred
