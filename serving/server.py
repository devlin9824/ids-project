# serving/server.py
from flask import Flask, request, jsonify
import numpy as np
from serving.pipeline import RFAEPipeline

app = Flask(__name__)
pipe = RFAEPipeline()  # load models once

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True, "dims": pipe.dim, "T1": pipe.T1, "T2": pipe.T2}

@app.route("/infer/flows", methods=["POST"])
def infer_flows():
    """
    Body JSON:
    {
      "rows": [
         {"features": [v1, v2, ..., v_dim], "meta": {...}},
         ...
      ]
    }
    """
    data = request.get_json(force=True)
    rows = data.get("rows", [])
    if not rows:
        return jsonify({"error": "no rows"}), 400
    X = np.array([r["features"] for r in rows], dtype=np.float32)
    probs, pred = pipe.predict_batch(X)
    out = []
    for i, r in enumerate(rows):
        out.append({
            "pred": int(pred[i]),              # 0 benign / 1 attack
            "p_attack": float(probs[i]),
            "meta": r.get("meta", {})
        })
    return jsonify({"results": out})
    
if __name__ == "__main__":
    # CPU/RAM friendly
    app.run(host="0.0.0.0", port=8008, threaded=True)
