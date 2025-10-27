from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os

MODEL_DIR = os.environ.get("MODEL_DIR", "bert")

app = Flask(__name__)
device = torch.device("cpu")
torch.set_num_threads(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device); model.eval()

@app.get("/health")
def health():
    return {"ok": True, "model_dir": MODEL_DIR}

def predict_batch(texts, max_len=256):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return probs, preds

@app.post("/infer/payload")
def infer_payload():
    data = request.get_json(force=True)
    texts = data.get("texts", [])
    if not texts:
        return jsonify({"error": "no texts"}), 400
    probs, preds = predict_batch(texts)
    out = [{"p_attack": float(p[1]), "pred": int(pred)} for p, pred in zip(probs, preds)]
    return jsonify({"results": out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8010, threaded=True)
