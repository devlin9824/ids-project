# serving/app.py
import os, json, warnings
from flask import Flask, request, jsonify

# ====== ENV & THREADS ======
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# ====== RF–AE Pipeline ======
import joblib, numpy as np, torch, torch.nn as nn

class AENet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,32)
        )
        self.dec = nn.Sequential(
            nn.Linear(32,64), nn.ReLU(),
            nn.Linear(64,128), nn.ReLU(),
            nn.Linear(128,d), nn.Sigmoid()
        )
    def forward(self,x): return self.dec(self.enc(x))

class RFAEPipeline:
    def __init__(self,
                 rf_meta_path="models/rf_binary.joblib",
                 ae_model_path="models/ae_model.pth",
                 ae_scaler_path="models/ae_scaler.npy",
                 thresholds_path="models/thresholds.json"):
        meta = joblib.load(rf_meta_path)
        self.clf = meta['model']
        self.num_cols = meta['num_cols']
        self.label_col = meta.get('label_col', 'binary_label')
        self.dim = len(self.num_cols)

        with open(thresholds_path) as f:
            th = json.load(f)
        self.T1 = float(th["T1"])
        self.T2 = float(th["T2"])

        sc = np.load(ae_scaler_path, allow_pickle=True).item()
        self.MIN = np.array(sc['min'], dtype=np.float32)
        self.SCALE = np.array(sc['scale'], dtype=np.float32)

        self.ae = AENet(self.dim)
        self.ae.load_state_dict(torch.load(ae_model_path, map_location='cpu'))
        self.ae.eval()
        torch.set_num_threads(1)

    def _scale_for_ae(self, X):
        Xs = (X.astype(np.float32) - self.MIN) / self.SCALE
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(Xs, 0.0, 1.0).astype(np.float32)

    def predict_flows(self, X_np):
        """X_np: ndarray [N, dim], numeric features same order as self.num_cols"""
        probs = self.clf.predict_proba(X_np)[:, 1]
        pred = (probs > self.T1).astype(np.int8)

        idx = np.where(pred == 1)[0]
        if len(idx) > 0:
            X_sel = self._scale_for_ae(X_np[idx])
            Xt = torch.from_numpy(X_sel)
            with torch.no_grad():
                rec = self.ae(Xt)
                mse = ((rec - Xt)**2).mean(dim=1).cpu().numpy()
            pred[idx[mse <= self.T2]] = 0
        return probs, pred

# ====== BERT Payload Classifier ======
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BertPayload:
    def __init__(self, model_dir="bert", max_len=256, thresholds_file="models/thresholds_payload.json"):
        self.device = torch.device("cpu")
        self.max_len = max_len
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        torch.set_num_threads(1)

        # threshold Tb (optional)
        self.Tb = 0.5
        try:
            with open(thresholds_file) as f:
                self.Tb = float(json.load(f).get("Tb", 0.5))
        except Exception:
            pass

    def predict_texts(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().tolist()  # avoid numpy dependency
        preds = [int(p.index(max(p))) for p in probs]
        return probs, preds

# ====== App Bootstrap ======
app = Flask(__name__)

# Load models once
MODEL_DIR = os.environ.get("MODEL_DIR", "bert")  # "bert" or "models/payload_bert"
bert = BertPayload(MODEL_DIR)

rf_ae = RFAEPipeline(
    rf_meta_path="models/rf_binary.joblib",
    ae_model_path="models/ae_model.pth",
    ae_scaler_path="models/ae_scaler.npy",
    thresholds_path="models/thresholds.json"
)

# ====== Routes ======
@app.get("/health")
def health():
    return {
        "ok": True,
        "bert_model_dir": bert.model_dir,
        "bert_Tb": bert.Tb,
        "flow_dim": rf_ae.dim,
        "T1": rf_ae.T1,
        "T2": rf_ae.T2
    }

@app.post("/infer/payload")
def infer_payload():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "no texts"}), 400
    probs, preds = bert.predict_texts(texts)
    out = [{"p_attack": float(p[1]), "pred": int(pred)} for p, pred in zip(probs, preds)]
    return jsonify({"results": out})

@app.post("/infer/flows")
def infer_flows():
    """
    Body:
    {
      "rows": [
        {"features": [v1..v_dim], "meta": {...}},
        ...
      ]
    }
    """
    data = request.get_json(force=True)
    rows = data.get("rows", [])
    if not rows:
        return jsonify({"error": "no rows"}), 400
    import numpy as np
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    probs, pred = rf_ae.predict_flows(X)
    out = []
    for i, r in enumerate(rows):
        out.append({
            "pred": int(pred[i]),             # 0 benign / 1 attack
            "p_attack": float(probs[i]),
            "meta": r.get("meta", {})
        })
    return jsonify({"results": out})

# (Tùy chọn) endpoint hợp nhất: nhận cả payload_texts và flow features trong 1 call
@app.post("/infer/fuse")
def infer_fuse():
    """
    Body:
    {
      "payload_texts": ["GET /..."],
      "flow_rows": [{"features":[...], "meta": {...}}, ...]
    }
    """
    data = request.get_json(force=True)
    res = {}
    if "payload_texts" in data:
        probs, preds = bert.predict_texts(data["payload_texts"])
        res["payload"] = [{"p_attack": float(p[1]), "pred": int(pred)} for p, pred in zip(probs, preds)]
    if "flow_rows" in data:
        import numpy as np
        X = np.array([r["features"] for r in data["flow_rows"]], dtype=np.float32)
        probs, pred = rf_ae.predict_flows(X)
        res["flows"] = [{"pred": int(pred[i]), "p_attack": float(probs[i]), "meta": data["flow_rows"][i].get("meta", {})} for i in range(len(data["flow_rows"]))]
    return jsonify(res or {"error":"no content"})

if __name__ == "__main__":
    # Chạy 1 lệnh cho tất cả
    app.run(host="0.0.0.0", port=8010, threaded=True)
