"""
extras_funnel_product.py
========================

Group B of Paper 2: two tests that only this shop allows, run under group
A's exact protocol (replicate_paper1.py: same sessions, splits, cap, models,
seeds), each against the group-A configuration it extends.

  funnel     completed-session table: counts_only + the session's checkout
             steps (cart/address/payment step count, reached payment).
             These steps happen after the first add-to-cart, so they are a
             leak of a kind paper 1's nine questions do not ask about.
  product    first add-to-cart table: the 13 prefix features + what is known
             about the cart at that moment: same-day (daily) product, lead
             time in days, price, category views and searches so far, mobile.
             Point-in-time correct; no customer id or login is used.

Output: 5-Results/own-data__extras-funnel-product.csv (aggregate only).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import replicate_paper1 as a  # group A data and protocol

OUT = Path(__file__).resolve().parent.parent / "5-Results" / "own-data__extras-funnel-product.csv"
CHECKOUT_STEPS = ["checkout_cart", "checkout_address", "checkout_payment"]

completed = a.completed.copy()
for sk, g in a.sessions.items():
    completed.at[sk, "checkout_step_count"] = int(g["event"].isin(CHECKOUT_STEPS).sum())
    completed.at[sk, "reached_payment"] = int((g["event"] == "checkout_payment").any())

prefix = a.prefix.copy()
for sk in prefix.index:
    g = a.sessions[sk]
    t0 = a.cart.at[sk, "first_add_ts"]
    p = g[g["timestamp"] <= t0]
    first_add = p[p["event"] == "addtocart"].iloc[0]
    prefix.at[sk, "first_item_daily"] = int(bool(a.cart.at[sk, "first_item_daily"]))
    prefix.at[sk, "first_item_lead_days"] = a.cart.at[sk, "first_item_lead_days"]
    prefix.at[sk, "first_item_price_m"] = first_add["price"] / 1e6
    prefix.at[sk, "category_views"] = int((p["event"] == "view_category").sum())
    prefix.at[sk, "searches"] = int((p["event"] == "search").sum())
    prefix.at[sk, "is_mobile"] = int(first_add["channel"] == "mobile")
PRODUCT = ["first_item_daily", "first_item_lead_days", "first_item_price_m",
           "category_views", "searches", "is_mobile"]
FUNNEL = ["checkout_step_count", "reached_payment"]
COUNTS = a.COMPLETED_CONFIGS["counts_only"]

PLAN = [("completed", "counts_only", COUNTS, completed),
        ("completed", "counts_plus_funnel", COUNTS + FUNNEL, completed),
        ("first_add_k0", "prefix13", a.PREFIX13, prefix),
        ("first_add_k0", "prefix13_plus_product", a.PREFIX13 + PRODUCT, prefix)]
SPLITS = {"completed": ["stratified", "visitor_grouped", "temporal"],
          "first_add_k0": ["visitor_grouped", "temporal"]}

def main():
    """Fit group A's models on both extensions and write the result file."""
    rows = []
    for table, config, cols, df in PLAN:
        for kind in SPLITS[table]:
            for seed in a.SEEDS:
                tr, te = a.split(df, kind, seed, table)
                X_tr, X_te, y_tr, y_te = a.prepare(df, cols, tr, te)
                for name, m in a.models(seed, y_tr, extra=False).items():
                    if name == "LR":
                        sc = StandardScaler().fit(X_tr)
                        m.fit(sc.transform(X_tr), y_tr)
                        prob = m.predict_proba(sc.transform(X_te))[:, 1]
                    else:
                        m.fit(X_tr, y_tr)
                        prob = m.predict_proba(X_te)[:, 1]
                    rows.append({"table": table, "config": config, "split": kind, "model": name,
                                 "seed": seed, "roc_auc": roc_auc_score(y_te, prob)})
    runs = pd.DataFrame(rows)
    summary = (runs.groupby(["table", "config", "split", "model"], sort=False)["roc_auc"]
               .agg(roc_auc="mean", roc_sd="std").reset_index())
    summary.round(4).to_csv(OUT, index=False)

    print("paid sessions that reached payment:",
          int(((completed["y"] == 1) & (completed["reached_payment"] == 1)).sum()), "of", int(completed["y"].sum()),
          "| unpaid that reached payment:", int(((completed["y"] == 0) & (completed["reached_payment"] == 1)).sum()))
    print("purchase rate by first item: daily", round(prefix.loc[prefix["first_item_daily"] == 1, "y"].mean(), 4),
          "| scheduled", round(prefix.loc[prefix["first_item_daily"] == 0, "y"].mean(), 4))
    print(summary.pivot_table(index=["table", "config"], columns=["model", "split"], values="roc_auc",
                              sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
