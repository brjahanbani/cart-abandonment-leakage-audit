"""
history_window_test.py
======================

Group B of Paper 2: the Retail Rocket aggregate-window probe
(6-Point-In-Time/probes/agg_probe.py, 2026-10-02) repeated on the
flower-shop data, with the same features, windows, model and splits.

Two history features at the first add-to-cart (k=0):
  item_cr   the carted item's smoothed conversion rate
            (purchases + 1) / (add-to-carts + 2)
  vis_txns  the visitor's purchase count
Each computed three ways:
  asof      only events strictly before the first add (point-in-time correct)
  full_loo  the whole data, minus this session's own events (future, no self)
  full      the whole data, including this session (future + self)

Same as the probe: XGBoost (200 trees, depth 4, learning rate 0.1, hist,
scale_pos_weight), seeds 11..55, no cap, missing values left to XGBoost;
splits visitor-grouped (GroupShuffleSplit on the visitor id) and temporal
(earliest 80% of sessions by start time train).

Mapping to this shop (stated, not hidden):
  visitor   = browser id (anonymousId), as Retail Rocket's visitorid is.
              No customer id or login is used in any feature.
  purchase  = a `checkout_final` event (vis_txns counts them per browser,
              as the probe counted `transaction` events per visitor).
  item purchases come from the order lines of paid orders, timed at the
              order's creation, because `final` events carry no product id.
              A session's own item purchases (for full_loo) are the lines of
              the orders linked to its `final` events (final_order_link).
  Counting starts at the first tracked event (2026-08-09) for both adds and
  purchases, so the two halves of the ratio cover the same period.
Rows: the 825 first-add rows of replicate_paper1.py (from 2026-08-23).

Output: 5-Results/own-data__history-window.csv (aggregate only).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "1-Data" / "own" / "clean"
OUT = ROOT / "5-Results" / "own-data__history-window.csv"
SEEDS = [11, 22, 33, 42, 55]
PURCHASE = "checkout_final"
BASE = ["view_count", "addtocart_count", "unique_items_count", "view_to_cart_ratio",
        "seconds_since_first_event", "seconds_since_first_cart", "seconds_since_last_cart",
        "seconds_since_previous_event", "inter_event_interval_mean", "events_after_cart",
        "repeated_views_of_cart_items", "hour_of_day", "day_of_week"]
# Retail Rocket probe results, XGBoost ROC-AUC (visitor / temporal), from
# docs/point_in_time_log.md (2026-10-02; the probe printed, it saved no file).
RETAIL_ROCKET = {
    ("none", "base"): (0.645, 0.664),
    ("asof", "agg_only"): (0.611, 0.633), ("asof", "base+agg"): (0.683, 0.711),
    ("full_loo", "agg_only"): (0.630, 0.658), ("full_loo", "base+agg"): (0.679, 0.726),
    ("full", "agg_only"): (0.971, 0.977), ("full", "base+agg"): (0.973, 0.977),
}

ev = pd.read_parquet(CLEAN / "events_clean.parquet")
cart = pd.read_parquet(CLEAN / "cart_sessions_labels.parquet").set_index("session_key")
orders = pd.read_parquet(CLEAN / "orders_clean.parquet")
lines = pd.read_parquet(CLEAN / "order_lines_clean.parquet")
link = pd.read_parquet(CLEAN / "final_order_link.parquet")

# The 825 first-add rows and their 13 features, exactly as in group A.
import replicate_paper1 as group_a  # noqa: E402  (runs group A's data preparation)
pref = group_a.prefix.copy()
pref["t0"] = cart.loc[pref.index, "first_add_ts"]
pref["item"] = cart.loc[pref.index, "first_item"]

tracking_start = int(ev["timestamp"].min())

# --- item side: add-to-carts (events) and purchases (paid order lines) ------
adds = ev[ev["event"] == "addtocart"]
add_times = {i: np.sort(g.values) for i, g in adds.groupby("itemid")["timestamp"]}
paid_lines = lines.merge(orders.loc[orders["paid"], ["id", "created_ms"]],
                         left_on="order_id", right_on="id")
paid_lines = paid_lines[paid_lines["created_ms"] >= tracking_start]
buy_times = {i: np.sort(g.values) for i, g in paid_lines.groupby("product_id")["created_ms"]}

own_adds = adds.groupby(["session_key", "itemid"]).size()
session_orders = link[link["paid"]].groupby("session_key")["order_id"].agg(set)
lines_by_order = lines.groupby("order_id")["product_id"].agg(list)

# --- visitor side: purchase events per browser -------------------------------
finals = ev[ev["event"] == PURCHASE]
vis_buy_times = {v: np.sort(g.values) for v, g in finals.groupby("visitorid")["timestamp"]}
own_finals = finals.groupby("session_key").size()

EMPTY = np.array([], dtype="int64")
rows = {}
for sk, r in pref.iterrows():
    item, t0, vis = r["item"], r["t0"], r["visitorid"]
    at, bt, vt = add_times.get(item, EMPTY), buy_times.get(item, EMPTY), vis_buy_times.get(vis, EMPTY)
    own_item_adds = int(own_adds.get((sk, item), 0))
    own_item_buys = sum(lines_by_order.get(o, []).count(item) for o in session_orders.get(sk, ()))
    carts_before, buys_before = np.searchsorted(at, t0, "left"), np.searchsorted(bt, t0, "left")
    rows[sk] = {
        "item_buys_before": int(buys_before),
        "item_cr_asof": (buys_before + 1) / (carts_before + 2),
        "item_cr_full_loo": (len(bt) - own_item_buys + 1) / (len(at) - own_item_adds + 2),
        "item_cr_full": (len(bt) + 1) / (len(at) + 2),
        "vis_txns_asof": int(np.searchsorted(vt, t0, "left")),
        "vis_txns_full_loo": len(vt) - int(own_finals.get(sk, 0)),
        "vis_txns_full": len(vt),
    }
pref = pref.join(pd.DataFrame(rows).T.astype(float))
assert (pref["item_cr_full_loo"] > 0).all() and (pref["vis_txns_full_loo"] >= 0).all()


def auc(feats, kind, seed):
    if kind == "temporal":
        r = pref["start_ts"].rank(method="first")
        tr = (r <= 0.8 * len(pref)).values
    else:
        i_tr, _ = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                       .split(pref, groups=pref["visitorid"]))
        tr = np.zeros(len(pref), bool)
        tr[i_tr] = True
    te = ~tr
    y = pref["y"].values
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                      scale_pos_weight=(y[tr] == 0).sum() / (y[tr] == 1).sum(),
                      random_state=seed, eval_metric="logloss", n_jobs=-1)
    m.fit(pref[feats].values[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(pref[feats].values[te])[:, 1])


def main():
    """Fit the probe's models on every window and write the result file."""
    out = []
    plan = [("none", "base", BASE)]
    for window in ["asof", "full_loo", "full"]:
        agg = [f"item_cr_{window}", f"vis_txns_{window}"]
        plan += [(window, "agg_only", agg), (window, "base+agg", BASE + agg)]
    for window, name, feats in plan:
        res = {kind: [auc(feats, kind, s) for s in SEEDS] for kind in ["visitor_grouped", "temporal"]}
        rr = RETAIL_ROCKET[(window, name)]
        out.append({"window": window, "features": name,
                    "visitor_auc": np.mean(res["visitor_grouped"]), "visitor_sd": np.std(res["visitor_grouped"]),
                    "temporal_auc": np.mean(res["temporal"]), "temporal_sd": np.std(res["temporal"]),
                    "rr_visitor_auc": rr[0], "rr_temporal_auc": rr[1]})
    out = pd.DataFrame(out)
    out.round(4).to_csv(OUT, index=False)

    print(f"rows {len(pref)}, positive rate {pref['y'].mean():.4f}")
    print("rows whose browser bought before the first add:", int((pref["vis_txns_asof"] > 0).sum()),
          "| ever bought (full_loo > 0):", int((pref["vis_txns_full_loo"] > 0).sum()))
    print("rows whose first item was bought before the add:", int((pref["item_buys_before"] > 0).sum()))
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
