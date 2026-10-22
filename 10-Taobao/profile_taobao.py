"""
profile_taobao.py
=================

Read-only profile of the Taobao "User Behavior" dataset (Alibaba; Kaggle /
Tianchi), the unrelated fourth shop for Paper 2. The 3.7 GB file is read in
chunks; nothing is cleaned or modelled.

File: 1-Data/taobao/raw/UserBehavior.csv, no header:
  user_id, item_id, category_id, behavior (pv | cart | fav | buy), timestamp (Unix s)

Dates are shown in China time (UTC+8), the shop's local time.
Output: 5-Results/taobao__profile.json
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "1-Data" / "taobao" / "raw" / "UserBehavior.csv"
OUT = ROOT / "5-Results" / "taobao__profile.json"
CHUNK = 5_000_000
CHINA = 8 * 3600

rows, mix, days, users = 0, {}, {}, set()
ts_min, ts_max, dup_in_chunk = None, None, 0
for chunk in pd.read_csv(RAW, header=None, names=["user", "item", "cat", "behavior", "ts"],
                         dtype={"user": "int32", "item": "int32", "cat": "int32",
                                "behavior": "category", "ts": "int64"}, chunksize=CHUNK):
    rows += len(chunk)
    for k, v in chunk["behavior"].value_counts().items():
        mix[k] = mix.get(k, 0) + int(v)
    d = pd.to_datetime(chunk["ts"] + CHINA, unit="s", errors="coerce").dt.date.astype(str)
    for k, v in d.value_counts().items():
        days[k] = days.get(k, 0) + int(v)
    users.update(chunk["user"].unique().tolist())
    ts_min = min(ts_min, int(chunk["ts"].min())) if ts_min is not None else int(chunk["ts"].min())
    ts_max = max(ts_max, int(chunk["ts"].max())) if ts_max is not None else int(chunk["ts"].max())
    dup_in_chunk += int(chunk.duplicated().sum())

big_days = {k: v for k, v in sorted(days.items()) if v >= 100_000}
profile = {
    "rows": rows, "behavior_mix": mix, "users": len(users),
    "ts_min_china": str(pd.to_datetime(ts_min + CHINA, unit="s")),
    "ts_max_china": str(pd.to_datetime(ts_max + CHINA, unit="s")),
    "days_with_100k_plus_rows": big_days,
    "rows_outside_those_days": rows - sum(big_days.values()),
    "exact_duplicates_within_chunks": dup_in_chunk,
}
OUT.write_text(json.dumps(profile, indent=2), encoding="utf-8")
print(json.dumps(profile, indent=2))
