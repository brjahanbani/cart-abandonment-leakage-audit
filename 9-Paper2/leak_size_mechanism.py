"""
leak_size_mechanism.py
======================

Why does the completed-session leak (L2) differ in size between datasets
(XGBoost 0.995 Retail Rocket, 0.76-0.78 flower shop, 0.96 REES46 October and
November, 0.93 Taobao)? Paper 1
traced it to the session endpoint (Sec. IV-B): timing features run to the
session's last event, and how a session ends differs by outcome. This script
measures, on the cart sessions each analysis used, how sharply the endpoint
separates buyers from non-buyers:

  abandoned sessions  share whose last event is an add-to-cart
                      (recency from the last cart = 0)
  purchased sessions  share whose last event is the purchase, and the median
                      time from the last purchase to the session's end
  both                median session age, purchased vs abandoned

The flower shop's front end logs a cart-page event (checkout_cart) automatically
right after an add-to-cart (median 0.6 s, column auto_cart_page_median_lag_s). As logged, almost no session ends on
an add-to-cart; the *_merged columns count such an event, within AUTO_GAP_MS of
an add-to-cart in the same session, as part of that add-to-cart. On the other
datasets, which log no cart-page event, merged equals logged.

Output: 5-Results/paper2__leak-size-mechanism.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "1-Data"
AUTO_GAP_MS = 3000


def ending_shares(ev, purchase):
    # Same definition as below: the session ends at the time of its last add-to-cart.
    end = ev.groupby("session_key", sort=False)["timestamp"].max()
    is_cart = ev["event"].eq("addtocart")
    last_cart = ev.loc[is_cart, "timestamp"].groupby(ev.loc[is_cart, "session_key"]).max()
    bought = ev["event"].eq(purchase).groupby(ev["session_key"], sort=False).any()
    ends_on_cart = end[last_cart.index].eq(last_cart)
    bought = bought[last_cart.index]
    return round(ends_on_cart[~bought].mean(), 4), round(ends_on_cart[bought].mean(), 4)


def endpoint_stats(name, ev, purchase, auto_page=None):
    ev = ev.sort_values(["session_key", "timestamp"], kind="stable")
    ev["event"] = ev["event"].astype(str)
    prev = ev.shift(1)
    auto = (ev["event"].eq(auto_page) & prev["event"].eq("addtocart")
            & prev["session_key"].eq(ev["session_key"])
            & (ev["timestamp"] - prev["timestamp"]).le(AUTO_GAP_MS))
    merged_ab, merged_pu = ending_shares(ev[~auto], purchase)
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
        "auto_cart_page_events": int(auto.sum()),
        "abandoned_ending_on_add_to_cart_merged": merged_ab,
        "purchased_ending_on_add_to_cart_merged": merged_pu,
        "purchased_ending_on_purchase": round((p["last_event"] == purchase).mean(), 4),
        "median_s_after_last_purchase": round(((p["end"] - p["last_buy"]) / 1000).median(), 1),
        "median_age_purchased_s": round(p["age_s"].median(), 1),
        "median_age_abandoned_s": round(a["age_s"].median(), 1),
        # Median time from the add-to-cart to the automatic cart-page event (empty where none).
        "auto_cart_page_median_lag_s": (round(float(((ev["timestamp"] - prev["timestamp"])[auto]
                                                     / 1000).median()), 1) if auto.any() else None),
    }


cols = ["timestamp", "event", "session_key"]
rows = [endpoint_stats("retail_rocket", pd.read_csv(D / "events_clean.csv", usecols=cols), "transaction")]

own = pd.read_parquet(D / "own" / "clean" / "events_clean.parquet", columns=cols + ["user_agent"])
own_from = own.loc[own["user_agent"].notna(), "timestamp"].min()
first_add = own[own["event"] == "addtocart"].groupby("session_key")["timestamp"].min()
own = own[own["session_key"].isin(first_add.index[first_add >= own_from])]
rows.append(endpoint_stats("flower_shop", own[cols], "checkout_final", auto_page="checkout_cart"))

ree = pd.read_parquet(D / "rees46" / "clean" / "events_cart_sessions.parquet", columns=cols)
start = ree.groupby("session_key")["timestamp"].min()
ree = ree[ree["session_key"].isin(start.index[start >= pd.Timestamp("2019-10-09").value // 10**6])]
rows.append(endpoint_stats("rees46", ree, "transaction"))
del ree

nov = pd.read_parquet(D / "rees46" / "clean" / "2019-Nov" / "events_cart_sessions.parquet", columns=cols)
rows.append(endpoint_stats("rees46_nov", nov, "transaction"))
del nov

tao =pd.read_parquet(D / "taobao" / "clean" / "events_cart_sessions.parquet", columns=cols)
rows.append(endpoint_stats("taobao", tao, "transaction"))

out = pd.DataFrame(rows)
out.to_csv(ROOT / "5-Results" / "paper2__leak-size-mechanism.csv", index=False)
print(out.T.to_string())
