#!/usr/bin/env python3
# runtime/sniffer.py
# Sniffer hợp nhất: HTTP->BERT, flows->RF-AE, heuristic SYN-flood + port-scan
# Writes alerts to alerts.jsonl (absolute path). Designed to be CPU-light.

import os, sys, time, json, signal, argparse, threading, requests
from collections import deque, defaultdict

# ===== Backend endpoints =====
BERT_URL = os.environ.get("BERT_URL", "http://127.0.0.1:8010/infer/payload")
FLOW_URL = os.environ.get("FLOW_URL", "http://127.0.0.1:8010/infer/flows")

# ===== Tunables =====
PAYLOAD_BATCH = int(os.environ.get("PAYLOAD_BATCH", "8"))
FLOW_BATCH = int(os.environ.get("FLOW_BATCH", "64"))
PAYLOAD_TIMEOUT = float(os.environ.get("PAYLOAD_TIMEOUT", "1.0"))
FLOW_TIMEOUT = float(os.environ.get("FLOW_TIMEOUT", "1.0"))

FLOW_IDLE_SECONDS = float(os.environ.get("FLOW_IDLE_SECONDS", "10.0"))
SNAPSHOT_EVERY_PKTS = int(os.environ.get("SNAPSHOT_EVERY_PKTS", "200"))
SNAPSHOT_INTERVAL_SECONDS = float(os.environ.get("SNAPSHOT_INTERVAL_SECONDS", "2.0"))

WIN = float(os.environ.get("IDS_WIN_SECS", "2.0"))
SYN_FLOOD_THRESH = int(os.environ.get("IDS_SYN_PER_DST_THRESH", "300"))
SCAN_DPORT_THRESH = int(os.environ.get("IDS_SCAN_DPORT_THRESH", "40"))

ALERT_FILE = os.path.abspath(os.environ.get("ALERT_FILE", "alerts.jsonl"))

# Debounce settings (optional)
LAST_ALERT = {}
ALERT_COOLDOWN = float(os.environ.get("IDS_ALERT_COOLDOWN", "2.0"))

# ===== Scapy import (optional) =====
try:
    from scapy.all import sniff, TCP, Raw, IP
    SCAPY_OK = True
except Exception:
    SCAPY_OK = False

# ===== RF meta (feature names order) =====
NUM_COLS, COL_INDEX = [], {}
for p in ["models/rf_binary.joblib", "models/flow/rf_binary.joblib"]:
    if os.path.exists(p):
        try:
            import joblib
            meta = joblib.load(p)
            NUM_COLS = meta.get("num_cols", meta.get("numeric_cols", []))
            COL_INDEX = {c: i for i, c in enumerate(NUM_COLS)}
            break
        except Exception:
            pass
if not NUM_COLS:
    NUM_COLS = ["col_%02d" % i for i in range(78)]
    COL_INDEX = {c: i for i, c in enumerate(NUM_COLS)}

FAST_MAP = {
    "Tot Fwd Pkts": lambda s: float(s.get("pkt", 0)),
    "TotLen Fwd Pkts": lambda s: float(s.get("bytes", 0)),
    "SYN Flag Cnt": lambda s: float(s.get("syn", 0)),
    "RST Flag Cnt": lambda s: float(s.get("rst", 0)),
    "PSH Flag Cnt": lambda s: float(s.get("psh", 0)),
    "ACK Flag Cnt": lambda s: float(s.get("ack", 0)),
    "Flow IAT Mean": lambda s: float(s["iat_sum"] / max(1, s["pkt"] - 1)) if s["pkt"] > 1 else 0.0,
    "Flow IAT Max": lambda s: float(s.get("iat_max", 0.0)),
    "Flow IAT Min": lambda s: 0.0 if s.get("iat_min", 1e9) == 1e9 else float(s.get("iat_min", 0.0)),
}

# ===== State =====
FLOW_STATS = {}; LAST_SEEN = {}; LAST_SNAP = {}
SYN_PER_DST = defaultdict(deque)
DPORTS_PER_SRC = defaultdict(deque)

payload_buf, payload_meta = [], []
payload_lock = threading.Lock()
payload_timer = None

flow_buf = []
flow_lock = threading.Lock()
flow_timer = None

def now_ts(): return time.time()

def write_alert(obj, key=None):
    ts_now = now_ts()
    if key is not None:
        last = LAST_ALERT.get(key, 0.0)
        if ts_now - last < ALERT_COOLDOWN:
            return
        LAST_ALERT[key] = ts_now
    obj["ts"] = ts_now
    line = json.dumps(obj, ensure_ascii=False)
    # ensure directory writable; write append
    with open(ALERT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print("[ALERT]", line)

# ===== Payload batching =====
def flush_payload_batch():
    global payload_buf, payload_meta, payload_timer
    with payload_lock:
        if not payload_buf: return
        texts = list(payload_buf); metas = list(payload_meta)
        payload_buf.clear(); payload_meta.clear()
        if payload_timer: payload_timer.cancel(); payload_timer = None
    try:
        r = requests.post(BERT_URL, json={"texts": texts}, timeout=5)
        r.raise_for_status()
        for meta, out in zip(metas, r.json().get("results", [])):
            p = float(out.get("p_attack", 0.0)); pred = int(out.get("pred", 0))
            if pred == 1 or p >= float(meta.get("tb", 0.5)):
                write_alert({"type":"http_payload_bert","p":p,"pred":pred,"meta":meta},
                            key=("payload", meta.get("src"), meta.get("dst")))
    except Exception as e:
        print("[WARN] payload batch error:", e)

def schedule_payload_flush():
    global payload_timer
    with payload_lock:
        if payload_timer: payload_timer.cancel()
        payload_timer = threading.Timer(PAYLOAD_TIMEOUT, flush_payload_batch)
        payload_timer.daemon = True
        payload_timer.start()

# ===== Flow batching =====
def vec_from_stats(st):
    vec = [0.0]*len(NUM_COLS)
    for fname, fn in FAST_MAP.items():
        if fname in COL_INDEX:
            try: vec[COL_INDEX[fname]] = float(fn(st))
            except: vec[COL_INDEX[fname]] = 0.0
    return vec

def push_flow_rows(rows):
    global flow_buf, flow_timer
    if not rows: return
    with flow_lock:
        flow_buf.extend(rows)
        if len(flow_buf) >= FLOW_BATCH:
            flush_flow_batch()
        else:
            schedule_flow_flush()

def flush_flow_batch():
    global flow_buf, flow_timer
    with flow_lock:
        if not flow_buf: return
        rows = list(flow_buf); flow_buf.clear()
        if flow_timer: flow_timer.cancel(); flow_timer = None
    try:
        r = requests.post(FLOW_URL, json={"rows": rows}, timeout=10)
        r.raise_for_status()
        for out in r.json().get("results", []):
            if int(out.get("pred", 0)) == 1:
                write_alert({"type":"flow_attack","p":out.get("p_attack"),"meta":out.get("meta", {})},
                            key=("flow", out.get("meta", {}).get("flow")))
    except Exception as e:
        print("[WARN] flow batch error:", e)

def schedule_flow_flush():
    global flow_timer
    with flow_lock:
        if flow_timer: flow_timer.cancel()
        flow_timer = threading.Timer(FLOW_TIMEOUT, flush_flow_batch)
        flow_timer.daemon = True
        flow_timer.start()

# ===== Flow update =====
def flow_key(pkt):
    try:
        ip = pkt[IP]; tcp = pkt[TCP]
        return (ip.src, ip.dst, int(tcp.sport), int(tcp.dport), "TCP")
    except Exception:
        return None

def update_flow_from_pkt(pkt, ts):
    k = flow_key(pkt)
    if not k: return
    st = FLOW_STATS.get(k)
    if st is None:
        st = {"pkt":0,"bytes":0,"fin":0,"syn":0,"rst":0,"psh":0,"ack":0,
              "last_ts":ts,"iat_sum":0.0,"iat_max":0.0,"iat_min":1e9}
    tcp = pkt[TCP]; flags = int(tcp.flags)
    size = len(bytes(pkt))
    fin = 1 if (flags & 0x01) else 0
    syn = 1 if (flags & 0x02) else 0
    rst = 1 if (flags & 0x04) else 0
    psh = 1 if (flags & 0x08) else 0
    ack = 1 if (flags & 0x10) else 0

    if st["pkt"] > 0:
        iat = ts - st["last_ts"]
        st["iat_sum"] += iat
        st["iat_max"] = max(st["iat_max"], iat)
        st["iat_min"] = min(st["iat_min"], iat)
    st["last_ts"] = ts

    st["pkt"] += 1; st["bytes"] += size
    st["fin"] += fin; st["syn"] += syn; st["rst"] += rst; st["psh"] += psh; st["ack"] += ack
    FLOW_STATS[k] = st
    LAST_SEEN[k] = ts

    if st["pkt"] % max(1, SNAPSHOT_EVERY_PKTS) == 0:
        vec = vec_from_stats(st)
        push_flow_rows([{"features": vec, "meta": {"flow": k, "pkt": st["pkt"], "bytes": st["bytes"], "snapshot": True}}])
        LAST_SNAP[k] = ts

def flush_idle_and_snapshots():
    now = now_ts()
    rows = []
    for k, t in list(LAST_SEEN.items()):
        if now - t > FLOW_IDLE_SECONDS:
            st = FLOW_STATS.pop(k, None)
            LAST_SEEN.pop(k, None); LAST_SNAP.pop(k, None)
            if not st: continue
            rows.append({"features": vec_from_stats(st), "meta": {"flow": k, "pkt": st["pkt"], "bytes": st["bytes"], "idle_flush": True}})
    for k, st in list(FLOW_STATS.items()):
        last = LAST_SNAP.get(k, 0.0)
        if now - last >= SNAPSHOT_INTERVAL_SECONDS:
            rows.append({"features": vec_from_stats(st), "meta": {"flow": k, "pkt": st["pkt"], "bytes": st["bytes"], "snapshot": True}})
            LAST_SNAP[k] = now
    if rows: push_flow_rows(rows)

# ===== Heuristics =====
def syn_flood_heuristic(dst, dport, ts):
    dq = SYN_PER_DST[(dst, dport)]
    dq.append(ts)
    while dq and ts - dq[0] > WIN:
        dq.popleft()
    if len(dq) >= SYN_FLOOD_THRESH:
        write_alert({"type":"syn_flood", "dst": dst, "dport": dport, "syn_in_win": len(dq), "win_secs": WIN},
                    key=("syn", dst, dport))

def portscan_heuristic(src, dst, dport, ts):
    dq = DPORTS_PER_SRC[(src, dst)]
    dq.append((ts, dport))
    while dq and ts - dq[0][0] > WIN:
        dq.popleft()
    uniq = {p for _, p in dq}
    if len(uniq) >= SCAN_DPORT_THRESH:
        write_alert({"type":"port_scan", "src": src, "dst": dst, "unique_dports_in_win": len(uniq), "win_secs": WIN},
                    key=("scan", src, dst))

# ===== HTTP detection =====
HTTP_METHODS = [b"GET", b"POST", b"HEAD", b"PUT", b"DELETE", b"OPTIONS", b"PATCH"]
def looks_like_http(raw_bytes):
    if not raw_bytes: return False
    line0 = raw_bytes.split(b"\r\n", 1)[0]
    for m in HTTP_METHODS:
        if line0.startswith(m + b" "): return True
    if b"HTTP/1." in line0: return True
    return False

# ===== Packet handler =====
def handle_pkt(pkt):
    ts = now_ts()
    try:
        if pkt.haslayer(TCP) and pkt.haslayer(IP):
            ip = pkt[IP]; tcp = pkt[TCP]
            flags = int(tcp.flags)
            if flags & 0x02:
                syn_flood_heuristic(ip.dst, int(tcp.dport), ts)
                portscan_heuristic(ip.src, ip.dst, int(tcp.dport), ts)
            if pkt.haslayer(Raw):
                raw = bytes(pkt[Raw])
                if looks_like_http(raw):
                    txt = raw.decode("utf-8", errors="replace")
                    meta = {"src": ip.src, "dst": ip.dst, "flow": (ip.src, ip.dst, int(tcp.sport), int(tcp.dport), "TCP"), "tb": 0.5}
                    with payload_lock:
                        payload_buf.append(txt); payload_meta.append(meta)
                        if len(payload_buf) >= PAYLOAD_BATCH: flush_payload_batch()
                        else: schedule_payload_flush()
            update_flow_from_pkt(pkt, ts)
            if len(LAST_SEEN) % 64 == 0:
                flush_idle_and_snapshots()
    except Exception as e:
        print("[WARN] handle_pkt:", e)

# ===== Shutdown =====
def _shutdown(signum=None, frame=None):
    print("[*] Shutdown: flush buffers ...")
    try:
        flush_payload_batch(); flush_idle_and_snapshots(); flush_flow_batch()
    finally:
        sys.exit(0)
signal.signal(signal.SIGINT, _shutdown); signal.signal(signal.SIGTERM, _shutdown)

# ===== Simulate =====
def simulate_from_csv(payload_csv="data/payload/test.csv"):
    import pandas as pd, time
    if os.path.exists(payload_csv):
        df = pd.read_csv(payload_csv).dropna(subset=["text"])
        for _, r in df.iterrows():
            txt = str(r["text"])
            with payload_lock:
                payload_buf.append(txt); payload_meta.append({"src":"sim","dst":"sim","flow":None,"tb":0.5})
                if len(payload_buf) >= PAYLOAD_BATCH: flush_payload_batch()
                else: schedule_payload_flush()
            time.sleep(0.01)
    time.sleep(2.0); flush_payload_batch()

def main_loop(interface=None, simulate=False):
    if simulate:
        print("[*] SIMULATE mode"); simulate_from_csv(); return
    if not SCAPY_OK:
        print("[!] scapy missing. pip install scapy or use --simulate"); return
    print("[*] sniffing on", interface or "<default>", "(CTRL+C to stop)")
    kwargs = dict(store=False, prn=handle_pkt, promisc=True, filter="tcp")
    if interface: kwargs["iface"] = interface
    sniff(**kwargs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None, help="network interface to sniff")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    main_loop(args.iface, args.simulate)
