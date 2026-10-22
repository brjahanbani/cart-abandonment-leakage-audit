"""
leak_size_mechanism.py
======================

Why does the completed-session leak (L2) differ in size between datasets
(XGBoost 0.995 Retail Rocket, 0.76-0.78 flower shop, 0.96 REES46)? Paper 1
traced it to the session endpoint (Sec. IV-B): timing features run to the
session's last event, and how a session ends differs by outcome. This script
measures, on the cart sessions each analysis used, how sharply the endpoint
separates buyers from non-buyers:

  abandoned sessions  share whose last event is an add-to-cart
                      (recency from the last cart = 0)
  purchased sessions  share whose last event is the purchase, and the median
                      time from the last purchase to the session's end
  both                median session age, purchased vs abandoned

Output: 5-Results/paper2__leak-size-mechanism.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "1-Data"


def endpoint_stats(name, ev, purchase):
    ev = ev.sort_values(["session_key", "timestamp"], kind="stable")
    ev["event"] = ev["event"].astype(str)
    g = ev.groupby("session_key", sort=False)
    s = pd.DataFrame({"start": g["timestamp"].min(), "end": g["timestamp"].max(),
                      "last_event": g["event"].last()})
    is_buy, is_cart = ev["event"].eq(purchase), ev["event"].eq("addtocart")
    s["bought"] = is_buy.groupby(ev["session_key"], sort=False).any()
    s["last_cart"] = ev.loc[is_cart, "timestamp"].groupby(ev.loc[is_cart, "session_key"]).max()
    s["last_buy"] = ev.loc[is_buy, "timestamp"].groupby(ev.loc[is_buy, "session_key"]).max()
    s = s[s["last_cart"].notna()]
    s["age_s"] = (s["end"] - s["start"]) / 1000
    p, a = s[s["bought"]], s[~s["bought"]]
    return {
        "dataset": name, "cart_sessions": len(s), "purchase_rate": round(s["bought"].mean(), 4),
        "abandoned_ending_on_add_to_cart": round((a["end"] == a["last_cart"]).mean(), 4),
        "purchased_ending_on_add_to_cart": round((p["end"] == p["last_cart"]).mean(), 4),
        "purchased_ending_on_purchase": round((p["last_event"] == purchase).mean(), 4),
        "median_s_after_last_purchase": round(((p["end"] - p["last_buy"]) / 1000).median(), 1),
        "median_age_purchased_s": round(p["age_s"].median(), 1),
        "median_age_abandoned_s": round(a["age_s"].median(), 1),
    }


cols = ["timestamp", "event", "session_key"]
rows = [endpoint_stats("retail_rocket", pd.read_csv(D / "events_clean.csv", usecols=cols), "transaction")]

own = pd.read_parquet(D / "own" / "clean" / "events_clean.parquet", columns=cols + ["user_agent"])
own_from = own.loc[own["user_agent"].notna(), "timestamp"].min()
first_add = own[own["event"] == "addtocart"].groupby("session_key")["timestamp"].min()
own = own[own["session_key"].isin(first_add.index[first_add >= own_from])]
rows.append(endpoint_stats("flower_shop", own[cols], "checkout_final"))

ree = pd.read_parquet(D / "rees46" / "clean" / "events_cart_sessions.parquet", columns=cols)
start = ree.groupby("session_key")["timestamp"].min()
ree = ree[ree["session_key"].isin(start.index[start >= pd.Timestamp("2019-10-09").value // 10**6])]
rows.append(endpoint_stats("rees46", ree, "transaction"))

tao = pd.read_parquet(D / "taobao" / "clean" / "events_cart_sessions.parquet", columns=cols)
rows.append(endpoint_stats("taobao", tao, "transaction"))

out = pd.DataFrame(rows)
out.to_csv(ROOT / "5-Results" / "paper2__leak-size-mechanism.csv", index=False)
print(out.T.to_string())
