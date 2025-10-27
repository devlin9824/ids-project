#!/usr/bin/env python3
# scripts/make_splits_downsample.py
# Tạo train/val/test memory-safe bằng cách đọc chunk và downsample attack.

import os, csv, random
from collections import defaultdict

IN = "data/cic2018_binary_numeric.csv"
OUT_DIR = "data/splits"
SEED = 42
random.seed(SEED)

# Mục tiêu (có thể chỉnh)
TRAIN_BENIGN_KEEP = 999999999          # giữ hết benign cho train (sau khi chia tỉ lệ)
TRAIN_ATTACK_CAP_PER_CLASS = None      # không dùng theo class (ta làm tổng attack cap)
TRAIN_ATTACK_CAP_TOTAL = 270_000       # ~2x benign (ước lượng), bạn có thể tăng/giảm

VAL_BENIGN_CAP = 50_000
VAL_ATTACK_CAP  = 50_000
TEST_BENIGN_CAP = 50_000
TEST_ATTACK_CAP = 50_000

# Tỉ lệ phân phối trước khi cap (dùng random): 70/15/15
RATIOS = (0.7, 0.15, 0.15)

os.makedirs(OUT_DIR, exist_ok=True)

# mở file out
files = {
    'train': open(os.path.join(OUT_DIR, "train.csv"), 'w', newline=''),
    'val':   open(os.path.join(OUT_DIR, "val.csv"), 'w', newline=''),
    'test':  open(os.path.join(OUT_DIR, "test.csv"), 'w', newline=''),
}
writers = {k: csv.writer(v) for k,v in files.items()}

# đọc header
with open(IN, 'r') as f:
    r = csv.reader(f)
    header = next(r)
# tìm cột label
label_idx = None
for i,c in enumerate(header):
    if 'binary' in c.lower() or c.lower()=='label':
        label_idx = i; break
assert label_idx is not None, "Không tìm được cột binary_label"
# ghi header cho cả 3
for w in writers.values():
    w.writerow(header)

# counters để cap
count = {
    'train': {'benign':0, 'attack':0},
    'val':   {'benign':0, 'attack':0},
    'test':  {'benign':0, 'attack':0},
}

def want_write(split, y):
    if split=='train':
        if y=='benign':
            return True  # giữ nguyên
        else:
            return count['train']['attack'] < TRAIN_ATTACK_CAP_TOTAL
    elif split=='val':
        if y=='benign':
            return count['val']['benign'] < VAL_BENIGN_CAP
        else:
            return count['val']['attack'] < VAL_ATTACK_CAP
    else: # test
        if y=='benign':
            return count['test']['benign'] < TEST_BENIGN_CAP
        else:
            return count['test']['attack'] < TEST_ATTACK_CAP

def pick_split():
    r = random.random()
    if r < RATIOS[0]: return 'train'
    elif r < RATIOS[0]+RATIOS[1]: return 'val'
    return 'test'

processed = 0
with open(IN, 'r') as f:
    r = csv.reader(f)
    next(r)  # skip header
    for row in r:
        y = (row[label_idx] or '').strip().lower()
        y = 'attack' if y=='attack' else 'benign'
        split = pick_split()
        if want_write(split, y):
            writers[split].writerow(row)
            count[split][y] += 1
        processed += 1
        if processed % 200000 == 0:
            print("Processed", processed, "rows",
                  "| train", count['train'],
                  "| val", count['val'],
                  "| test", count['test'])

# đóng file
for f in files.values(): f.close()
print("DONE. Final counts:", count)
