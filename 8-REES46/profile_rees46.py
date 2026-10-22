"""
profile_rees46.py
=================

Read-only profile of one month of the public REES46 "eCommerce Events
History in Cosmetics Shop" data (Kaggle), before any cleaning decision.
Prints the facts and writes aggregate counts only.

Input:  1-Data/rees46/raw/2019-Oct.csv  (git-ignored)
Output: 5-Results/rees46__profile.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "1-Data" / "rees46" / "raw" / "2019-Oct.csv"
OUT = ROOT / "5-Results" / "rees46__profile.json"
GAP_S = 30 * 60

df = pd.read_csv(RAW, dtype={
    "event_type": "category", "product_id": "int32", "category_id": "int64",
    "category_code": "category", "brand": "category", "price": "float32",
    "user_id": "int32", "user_session": "string"})
df["ts"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S UTC").astype("int64") // 10**9
df = df.drop(columns="event_time")

p = {"rows": len(df), "columns": list(df.columns)}
p["nulls"] = df.isna().sum()[lambda s: s > 0].to_dict()
p["event_mix"] = df["event_type"].value_counts().to_dict()
p["window"] = [str(pd.to_datetime(df["ts"].min(), unit="s")), str(pd.to_datetime(df["ts"].max(), unit="s"))]
p["users"] = int(df["user_id"].nunique())
p["products"] = int(df["product_id"].nunique())
p["site_sessions"] = int(df["user_session"].nunique())
p["exact_duplicate_rows"] = int(df.duplicated().sum())
p["price_le_0"] = int((df["price"] <= 0).sum())
p["price_le_0_by_event"] = df.loc[df["price"] <= 0, "event_type"].value_counts()[lambda s: s > 0].to_dict()

# Site sessions: one user each?
sess_users = df.dropna(subset=["user_session"]).groupby("user_session")["user_id"].nunique()
p["site_sessions_with_several_users"] = int((sess_users > 1).sum())

# Paper 1's session rule: 30-minute gap per user (Retail Rocket had no session id).
df = df.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
gap = df.groupby("user_id")["ts"].diff()
df["sess"] = (gap.isna() | (gap > GAP_S)).cumsum()
p["gap30_sessions"] = int(df["sess"].nunique())

ev = df["event_type"].astype(str)
g = df.assign(is_cart=ev.eq("cart"), is_buy=ev.eq("purchase")).groupby("sess")
s = pd.DataFrame({"n": g.size(), "carts": g["is_cart"].sum(), "buys": g["is_buy"].sum()})
cart_s = s[s["carts"] > 0]
p["gap30_cart_sessions"] = len(cart_s)
p["gap30_cart_sessions_with_purchase"] = int((cart_s["buys"] > 0).sum())
p["gap30_cart_session_purchase_rate"] = round(float((cart_s["buys"] > 0).mean()), 4)
p["gap30_sessions_with_purchase_but_no_cart"] = int(((s["buys"] > 0) & (s["carts"] == 0)).sum())

# Timing between the first add-to-cart and the first purchase in cart sessions.
t_cart = df[ev.eq("cart")].groupby("sess")["ts"].min()
t_buy = df[ev.eq("purchase")].groupby("sess")["ts"].min()
both = pd.concat([t_cart.rename("cart"), t_buy.rename("buy")], axis=1).dropna()
p["purchase_before_first_cart"] = int((both["buy"] < both["cart"]).sum())
p["purchase_same_second_as_first_cart"] = int((both["buy"] == both["cart"]).sum())

# Ties: several events in the same second for one user (order is then unknown).
p["rows_sharing_a_second_with_another_row_of_same_user"] = int(df.duplicated(["user_id", "ts"], keep=False).sum())

# Retail Rocket bot rules (velocity > 500 events/h, or interval CV < 0.05 with >= 20 events).
u = df.groupby("user_id")["ts"].agg(["size", "min", "max"])
span_h = ((u["max"] - u["min"]) / 3600).clip(lower=1 / 60)
velocity_bots = set(u.index[(u["size"] / span_h) > 500])
d = df.assign(dt=gap).dropna(subset=["dt"]).groupby("user_id")["dt"].agg(["size", "mean", "std"])
cv_bots = set(d.index[(d["size"] >= 20) & (d["mean"] > 0) & ((d["std"] / d["mean"]) < 0.05)]) | \
          set(d.index[(d["size"] >= 20) & (d["mean"] == 0)])
bots = velocity_bots | cv_bots
p["rr_rule_bot_users"] = len(bots)
p["rr_rule_bot_events"] = int(df["user_id"].isin(bots).sum())
p["rr_rule_bot_purchases"] = int((df["user_id"].isin(bots) & ev.eq("purchase")).sum())

per_user = df.groupby("user_id").size()
p["events_per_user"] = {k: round(float(v), 1) for k, v in per_user.describe(percentiles=[.5, .9, .99]).items()}

OUT.write_text(json.dumps(p, indent=2, default=str), encoding="utf-8")
print(json.dumps(p, indent=2, default=str))
