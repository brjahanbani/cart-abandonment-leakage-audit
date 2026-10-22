"""
build_taobao_events.py
======================

Cleaning + sessionization of the Taobao "User Behavior" dataset (Alibaba,
China, 25 Nov - 3 Dec 2017), the unrelated fourth shop for Paper 2, with
paper 1's own rules (0-Shared/clean_events.py), as for every other dataset.

Steps
  1. Read the 3.7 GB file in chunks and keep a fixed 10% user sample
     (user_id divisible by 10). Every kept user keeps all their events, so
     sessions stay whole. 10%, not more, because building session keys for
     ~10M events already uses several GB on this machine.
  2. Keep the nine data days (China time, UTC+8), per profile_taobao.py:
     55,576 rows across the file carry impossible dates (1902-2037).
  3. Drop exact duplicate rows.
  4. Paper 1's vocabulary: pv -> view, cart -> addtocart, buy -> transaction;
     fav keeps its name (an extra event type, like REES46's removals).
     timestamp in ms (UTC epoch, as for the other datasets).
  5. Paper 1's bot rules and 30-minute sessions, unchanged.
  6. Keep cart sessions; also save every clean event (history features).
Timestamps are whole seconds, so same-second order is unknown (as REES46).

Output: 1-Data/taobao/clean/events_cart_sessions.parquet, events_clean.parquet (git-ignored)
        5-Results/taobao__clean.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from clean_events import remove_bots, sessionize  # noqa: E402  (paper 1's rules)

RAW = ROOT / "1-Data" / "taobao" / "raw" / "UserBehavior.csv"
CLEAN = ROOT / "1-Data" / "taobao" / "clean"
STATS = ROOT / "5-Results" / "taobao__clean.json"
SAMPLE_MOD = 10
CHINA = 8 * 3600
FIRST_DAY, LAST_DAY = pd.Timestamp("2017-11-25"), pd.Timestamp("2017-12-03")
EVENT_MAP = {"pv": "view", "cart": "addtocart", "buy": "transaction", "fav": "fav"}

stats = {"sample": f"users with user_id % {SAMPLE_MOD} == 0"}
parts = []
for chunk in pd.read_csv(RAW, header=None, names=["user", "item", "cat", "behavior", "ts"],
                         dtype={"user": "int32", "item": "int32", "cat": "int32",
                                "behavior": "category", "ts": "int64"}, chunksize=5_000_000):
    parts.append(chunk[chunk["user"] % SAMPLE_MOD == 0])
df = pd.concat(parts, ignore_index=True)
del parts
stats["sample_rows"] = len(df)
stats["sample_users"] = int(df["user"].nunique())

day = pd.to_datetime(df["ts"] + CHINA, unit="s", errors="coerce").dt.normalize()
in_window = day.between(FIRST_DAY, LAST_DAY)
stats["rows_outside_window_dropped"] = int((~in_window).sum())
df = df[in_window.values]

dup = df.duplicated()
stats["exact_duplicates_dropped"] = int(dup.sum())
df = df[~dup]

df = pd.DataFrame({
    "timestamp": df["ts"].values * 1000,
    "visitorid": df["user"].values,
    "event": df["behavior"].astype(str).map(EVENT_MAP).values,
    "itemid": df["item"].values,
    "category_id": df["cat"].values,
})
assert df["event"].notna().all(), "unmapped behavior"
df["event"] = df["event"].astype("category")

df, bot_stats = remove_bots(df)
stats["rr_rule_bot_visitors"] = bot_stats["n_bot_visitors"]
stats["rr_rule_bot_events"] = bot_stats["n_bot_events"]
stats["rr_rule_transactions_removed"] = int(bot_stats["transactions_removed_by_bot_filter"])
stats["clean_event_mix"] = {k: int(v) for k, v in bot_stats["clean_event_mix"].items()}

df = sessionize(df).drop(columns=["prev_ts", "gap_ms", "new_session", "session_id"])
stats["sessions"] = int(df["session_key"].nunique())
cart_keys = df.loc[df["event"] == "addtocart", "session_key"].unique()
cart = df[df["session_key"].isin(cart_keys)].reset_index(drop=True)
has_txn = (cart["event"] == "transaction").groupby(cart["session_key"]).any()
stats["cart_sessions"] = int(len(cart_keys))
stats["cart_sessions_with_transaction"] = int(has_txn.sum())
stats["cart_session_transaction_rate"] = round(float(has_txn.mean()), 4)
stats["cart_session_events"] = len(cart)

CLEAN.mkdir(parents=True, exist_ok=True)
cart.to_parquet(CLEAN / "events_cart_sessions.parquet", index=False)
df[["timestamp", "visitorid", "event", "itemid", "session_key"]].to_parquet(CLEAN / "events_clean.parquet", index=False)
STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
