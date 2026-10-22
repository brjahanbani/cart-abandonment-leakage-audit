"""
build_rees46_events.py
======================

Cleaning + sessionization of one month of REES46 "eCommerce Events History
in Cosmetics Shop" (October 2019), using paper 1's own code for every rule
the two datasets share (0-Shared/clean_events.py: bot rules, 30-minute
sessions), so the third dataset goes through the same pipeline as the first.

Steps
  1. Load; event_time has whole-second resolution (Retail Rocket: ms).
  2. Drop exact duplicate rows (5.2%; 87% are repeated remove_from_cart).
     Paper 1 had no such step; the data needs it.
  3. Map to paper 1's vocabulary: cart -> addtocart, purchase -> transaction;
     view and remove_from_cart keep their names. timestamp in ms, user_id is
     the visitor id (REES46 has persistent user ids; Retail Rocket's
     visitorid plays the same role).
  4. Paper 1's bot rules, unchanged (remove_bots). Unlike Retail Rocket,
     they remove some purchases here (users whose whole month is one burst).
  5. Paper 1's 30-minute sessions (sessionize). The site's own user_session
     is kept as a column for a later sensitivity check, not used.
  6. Keep cart sessions (at least one addtocart) and write their events.
Rows keep their file order inside one second (stable sorts), because the
order of same-second events is unknown.

Output: 1-Data/rees46/clean/events_cart_sessions.parquet (git-ignored)
        1-Data/rees46/clean/events_clean.parquet  every clean event, all
            sessions (history features count purchases outside cart
            sessions too, as paper 1's probe did)
        5-Results/rees46__clean.json (aggregate counts)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from clean_events import remove_bots, sessionize  # noqa: E402  (paper 1's rules)

_ap = argparse.ArgumentParser()
_ap.add_argument("--month", default="2019-Oct", help="raw file name without .csv, e.g. 2019-Nov")
MONTH = _ap.parse_known_args()[0].month
# October keeps its original locations; other months get their own folder and file suffix.
RAW = ROOT / "1-Data" / "rees46" / "raw" / f"{MONTH}.csv"
CLEAN = ROOT / "1-Data" / "rees46" / "clean" / ("" if MONTH == "2019-Oct" else MONTH)
STATS = ROOT / "5-Results" / ("rees46__clean.json" if MONTH == "2019-Oct" else f"rees46__clean__{MONTH}.json")
EVENT_MAP = {"view": "view", "cart": "addtocart", "remove_from_cart": "remove_from_cart",
             "purchase": "transaction"}

stats = {}
df = pd.read_csv(RAW, dtype={
    "event_type": "category", "product_id": "int32", "category_id": "int64",
    "category_code": "category", "brand": "category", "price": "float32",
    "user_id": "int32", "user_session": "string"})
stats["raw_rows"] = len(df)

# 2. exact duplicates
dup = df.duplicated()
stats["exact_duplicates_dropped"] = df.loc[dup, "event_type"].value_counts().to_dict()
df = df[~dup]

# 3. paper 1 vocabulary
df = pd.DataFrame({
    "timestamp": pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC").astype("int64") // 10**6,
    "visitorid": df["user_id"].values,
    "event": df["event_type"].astype(str).map(EVENT_MAP).values,
    "itemid": df["product_id"].values,
    "price": df["price"].values,
    "category_id": df["category_id"].values,
    "brand": df["brand"].values,
    "site_session": df["user_session"].values,
})
assert df["event"].notna().all(), "unmapped event type"
df["event"] = df["event"].astype("category")

# 4. paper 1's bot rules
df, bot_stats = remove_bots(df)
stats["rr_rule_bot_visitors"] = bot_stats["n_bot_visitors"]
stats["rr_rule_bot_events"] = bot_stats["n_bot_events"]
stats["rr_rule_transactions_removed"] = int(bot_stats["transactions_removed_by_bot_filter"])
stats["clean_event_mix"] = {k: int(v) for k, v in bot_stats["clean_event_mix"].items()}

# 5. paper 1's 30-minute sessions
df = sessionize(df).drop(columns=["prev_ts", "gap_ms", "new_session", "session_id"])
stats["sessions"] = int(df["session_key"].nunique())

# 6. cart sessions
cart_keys = df.loc[df["event"] == "addtocart", "session_key"].unique()
cart = df[df["session_key"].isin(cart_keys)].reset_index(drop=True)
has_txn = cart.groupby("session_key")["event"].apply(lambda e: (e == "transaction").any())
stats["cart_sessions"] = int(len(cart_keys))
stats["cart_sessions_with_transaction"] = int(has_txn.sum())
stats["cart_session_transaction_rate"] = round(float(has_txn.mean()), 4)
stats["cart_session_events"] = len(cart)
stats["cart_visitors"] = int(cart["visitorid"].nunique())
stats["window"] = [str(pd.to_datetime(df["timestamp"].min(), unit="ms")),
                   str(pd.to_datetime(df["timestamp"].max(), unit="ms"))]

CLEAN.mkdir(parents=True, exist_ok=True)
cart.to_parquet(CLEAN / "events_cart_sessions.parquet", index=False)
df[["timestamp", "visitorid", "event", "itemid", "session_key"]].to_parquet(
    CLEAN / "events_clean.parquet", index=False)
STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
print(json.dumps(stats, indent=2))
