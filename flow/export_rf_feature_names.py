#!/usr/bin/env python3
import os, json
from joblib import load

FLOW_DIR = os.path.dirname(__file__)
rf_path = os.path.join(FLOW_DIR, "rf_binary.joblib")
out_path = os.path.join(FLOW_DIR, "selected_cols.json")

rf = load(rf_path)
names = getattr(rf, "feature_names_in_", None)
if names is None:
    raise SystemExit("Model RF không chứa feature_names_in_. Cần cung cấp danh sách cột TOP-30 (xem phần B.3).")

with open(out_path, "w") as f:
    json.dump(list(names), f, indent=2)
print(f"[OK] wrote {out_path} with {len(names)} columns.")
