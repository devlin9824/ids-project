#!/usr/bin/env python3
# Train Autoencoder (benign-only) RAM-friendly + compute T2
import os, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib

torch.set_num_threads(1)

TRAIN = "data/splits/train.csv"
VAL   = "data/splits/val.csv"
MODEL_PATH   = "models/ae_model.pth"
SCALER_PATH  = "models/ae_scaler.npy"
THRESH_PATH  = "models/thresholds.json"
RF_META_PATH = "models/rf_binary.joblib"

os.makedirs("models", exist_ok=True)

# ---- Load RF metadata (numeric columns + label col) ----
rfmeta = joblib.load(RF_META_PATH)
num_cols  = rfmeta["num_cols"]
label_col = rfmeta["label_col"]
dim = len(num_cols)

EPS = 1e-8  # to avoid division by zero

def build_minmax_scaler(sample_n: int = 20000):
    """Fit MinMaxScaler on benign sample; save min/scale as float32, clamp scale==0 to EPS."""
    it = pd.read_csv(TRAIN, usecols=num_cols+[label_col], chunksize=50000)
    frames = []
    got = 0
    for ch in it:
        b = ch[ch[label_col].str.lower() == "benign"]
        if len(b) == 0:
            continue
        frames.append(b[num_cols])
        got += len(b)
        if got >= sample_n:
            break
    if not frames:
        raise RuntimeError("No benign rows found to build MinMaxScaler.")
    df = pd.concat(frames)[:sample_n].fillna(0).astype("float32")

    sc = MinMaxScaler()
    sc.fit(df)

    min_arr   = sc.min_.astype(np.float32)
    scale_arr = sc.scale_.astype(np.float32)
    # fix constant features (scale==0)
    scale_arr[~np.isfinite(scale_arr)] = EPS
    scale_arr[scale_arr == 0.0] = EPS

    np.save(SCALER_PATH, {"min": min_arr.tolist(), "scale": scale_arr.tolist()})
    print(f"[Scaler] Built from {len(df)} benign samples → {SCALER_PATH}")
    return min_arr, scale_arr

MIN_ARR, SCALE_ARR = build_minmax_scaler(20000)

class AE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.dec = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, d), nn.Sigmoid()
        )
    def forward(self, x): return self.dec(self.enc(x))

def safe_scale_float32(X_np: np.ndarray) -> np.ndarray:
    """MinMax scale → float32, replace NaN/Inf, clip to [0, 1] where possible."""
    Xs = (X_np - MIN_ARR) / SCALE_ARR
    # replace non-finite
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=1.0, neginf=0.0)
    # clamp (since MinMax → ideally [0,1], nhưng dữ liệu thực có thể out-of-range)
    Xs = np.clip(Xs, 0.0, 1.0).astype(np.float32)
    return Xs

def gen_benign_batches(csv_path: str, batch_size: int = 32):
    for ch in pd.read_csv(csv_path, usecols=num_cols+[label_col], chunksize=50000):
        b = ch[ch[label_col].str.lower() == "benign"]
        if len(b) == 0: continue
        X = b[num_cols].fillna(0).astype("float32").values
        Xs = safe_scale_float32(X)
        for i in range(0, len(Xs), batch_size):
            yield torch.from_numpy(Xs[i:i+batch_size])

def train_ae(epochs: int = 20, batch_size: int = 32, lr: float = 1e-3):
    model = AE(dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    saved_once = False

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        nb = 0
        bad_batches = 0

        for bt in gen_benign_batches(TRAIN, batch_size):
            bt = bt.float()
            opt.zero_grad()
            rec = model(bt)
            loss = loss_fn(rec, bt)
            val = loss.item()
            if not np.isfinite(val):
                bad_batches += 1
                continue
            loss.backward()
            opt.step()
            total += val
            nb += 1

        if nb == 0:
            print(f"[AE] Epoch {ep}: no valid batches (bad_batches={bad_batches}).")
            continue

        avg = total / nb
        print(f"[AE] Epoch {ep:02d}/{epochs} avg_loss={avg:.6f} batches={nb} (skipped={bad_batches})")

        if np.isfinite(avg) and avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), MODEL_PATH)
            saved_once = True
            print(f"[AE] Saved best → {MODEL_PATH}")

    # ensure at least one checkpoint exists
    if not saved_once:
        torch.save(model.state_dict(), MODEL_PATH)
        print("[AE] No finite improvement, saved last model →", MODEL_PATH)

    return MODEL_PATH

def compute_T2(model_path: str, batch_size: int = 32) -> float:
    model = AE(dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    errs = []
    with torch.no_grad():
        for bt in gen_benign_batches(VAL, batch_size):
            bt = bt.float()
            rec = model(bt)
            mse = ((rec - bt)**2).mean(dim=1).numpy()
            errs.extend(mse.tolist())

    if not errs:
        raise RuntimeError("No benign batches on VAL to compute T2.")
    errs = np.array(errs, dtype=np.float32)
    errs = np.nan_to_num(errs, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
    T2 = float(np.percentile(errs, 75))
    print(f"[AE] T2 (75th percentile of benign MSE on VAL) = {T2}")
    return T2

def update_thresholds(T2_value: float):
    th = {}
    if os.path.exists(THRESH_PATH):
        try:
            with open(THRESH_PATH, "r") as f:
                th = json.load(f)
        except Exception:
            th = {}
    th["T2"] = float(T2_value)
    with open(THRESH_PATH, "w") as f:
        json.dump(th, f)
    print(f"[AE] Updated {THRESH_PATH} with T2={T2_value}")

if __name__ == "__main__":
    EPOCHS     = int(os.environ.get("AE_EPOCHS", "20"))
    BATCH_SIZE = int(os.environ.get("AE_BATCH", "32"))
    LR         = float(os.environ.get("AE_LR", "1e-3"))

    print(f"[AE] Training: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, dim={dim}")
    best_path = train_ae(EPOCHS, BATCH_SIZE, LR)

    print("[AE] Computing T2 ...")
    T2 = compute_T2(best_path, BATCH_SIZE)
    update_thresholds(T2)
    print("[AE] Done.")
