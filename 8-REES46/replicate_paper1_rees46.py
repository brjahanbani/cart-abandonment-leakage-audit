"""
replicate_paper1_rees46.py
==========================

Group A on the third dataset: paper 1's evaluations repeated on REES46
(cosmetics shop, October 2019), with nothing added and nothing left out.
Run after build_rees46_events.py.

The protocol is paper 1's, imported from 0-Shared/paper1_protocol.py (the
same code the flower-shop replication uses). With 205k cart sessions the
features are computed in bulk instead of one session at a time; before any
model is fitted, the bulk features are checked against paper 1's reference
functions on 3,000 random sessions and the script stops on any difference.

Also repeats paper 1's label checks (Section IV-C): purchases before the
first add-to-cart, and purchased sessions that also abandoned a carted item
(item level; REES46 purchase rows carry the product id).

Two runs (see probe_temporal_shift.py for why):
  python replicate_paper1_rees46.py                 all of October, paper 1's
                                                    rules as written
  python replicate_paper1_rees46.py --from 2019-10-09
                                                    clean window: Oct 1-8 hold
                                                    bursts of automated-looking
                                                    traffic (flat 24-hour profile,
                                                    one-visit user ids, 20 s
                                                    visits) that paper 1's
                                                    per-visitor bot rules cannot
                                                    see; sessions starting
                                                    before the date are dropped

Outputs (aggregate only; <tag> = "all-october" or "from-2019-10-09"):
  5-Results/rees46__paper1-replication__<tag>__runs.csv
  5-Results/rees46__paper1-replication__<tag>__summary.csv
  5-Results/rees46__label-checks__<tag>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from paper1_protocol import (  # noqa: E402
    PREFIX13, completed_features, prefix_features, retail_rocket_reference, run_plan, summarize)
import burst_check  # noqa: E402

CLEAN = ROOT / "1-Data" / "rees46" / "clean"
RESULTS = ROOT / "5-Results"
PURCHASE = "transaction"
CHECK_N = 3000

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["rees46", "taobao"], default="rees46",
                    help="taobao: 10-Taobao/build_taobao_events.py output, same protocol")
parser.add_argument("--month", default="2019-Oct", help="e.g. 2019-Nov (built by build_rees46_events.py)")
parser.add_argument("--from", dest="window_from", default=None,
                    help="keep sessions starting on or after this date (YYYY-MM-DD)")
parser.add_argument("--exclude-automated-days", action="store_true",
                    help="drop sessions on days the shared burst rule (0-Shared/burst_check.py) "
                         "classifies as automated traffic")
ARGS, _ = parser.parse_known_args()
TAOBAO = ARGS.dataset == "taobao"
PREFIX = ARGS.dataset  # output file prefix
OCT = ARGS.month == "2019-Oct" and not TAOBAO
if TAOBAO:
    CLEAN = ROOT / "1-Data" / "taobao" / "clean"
elif not OCT:
    CLEAN = CLEAN / ARGS.month
if ARGS.window_from:
    TAG = f"from-{ARGS.window_from}"
elif ARGS.exclude_automated_days:
    TAG = "no-automated-days"
else:
    TAG = "all-october" if OCT else ("all" if TAOBAO else "all-month")
TAG = TAG if (OCT or TAOBAO) else f"{ARGS.month}__{TAG}"

ev = pd.read_parquet(CLEAN / "events_cart_sessions.parquet",
                     columns=["timestamp", "visitorid", "event", "itemid", "session_key"])
ev["event"] = ev["event"].astype(str)
starts = ev.groupby("session_key")["timestamp"].min()
BURST, AUTOMATED_DATES = burst_check.check(
    starts.values, ev["event"].eq(PURCHASE).groupby(ev["session_key"]).any().reindex(starts.index).values)
if ARGS.window_from:
    cutoff_ms = pd.Timestamp(ARGS.window_from).value // 10**6
    ev = ev[ev["session_key"].isin(starts.index[starts >= cutoff_ms])]
if ARGS.exclude_automated_days and BURST["automated"]:
    start_day = pd.to_datetime(starts, unit="ms").dt.date
    ev = ev[ev["session_key"].isin(start_day.index[~start_day.isin(AUTOMATED_DATES)])]
print(f"run: {TAG}; burst rule: {BURST}")
ev = ev.sort_values(["session_key", "timestamp"], kind="stable").reset_index(drop=True)
key = ev["session_key"]
is_view, is_cart, is_buy = ev["event"].eq("view"), ev["event"].eq("addtocart"), ev["event"].eq(PURCHASE)


def pyround(values, digits):
    """Python's round(), element by element, as paper 1's function uses."""
    return [round(float(v), digits) for v in values]


# ============================================================================
# COMPLETED-SESSION TABLE (bulk)
# ============================================================================
g = ev.groupby(key, sort=False)
first, last, n = g["timestamp"].min(), g["timestamp"].max(), g.size()
views = is_view.groupby(key, sort=False).sum()
carts = is_cart.groupby(key, sort=False).sum()
last_cart = ev.loc[is_cart, "timestamp"].groupby(key[is_cart], sort=False).max()
completed = pd.DataFrame({
    "visitorid": g["visitorid"].first(),
    "start_ts": first,
    "event_count": n,
    "view_count": views,
    "addtocart_count": carts,
    "view_to_cart_ratio": pyround(views / carts, 4),
    "unique_items_count": g["itemid"].nunique(),
    "session_age_seconds": pyround((last - first) / 1000.0, 2),
    "inter_event_interval_mean": pyround(np.where(n > 1, (last - first) / 1000.0 / (n - 1).clip(lower=1), 0.0), 3),
    "recency_last_cart_seconds": pyround((last - last_cart.reindex(first.index)) / 1000.0, 2),
    "y": is_buy.groupby(key, sort=False).any().astype(int),
})

# ============================================================================
# FIRST ADD-TO-CART TABLE, k=0 (bulk)
# ============================================================================
t0 = ev.loc[is_cart, "timestamp"].groupby(key[is_cart], sort=False).min()
first_buy = ev.loc[is_buy, "timestamp"].groupby(key[is_buy], sort=False).min()
bought_before = first_buy.reindex(t0.index) <= t0
risk = t0[~bought_before.fillna(False)]
ev_t0 = key.map(t0)
pre = ev[(ev["timestamp"] <= ev_t0) & key.isin(risk.index)]
pk = pre["session_key"]
pg = pre.groupby(pk, sort=False)
p_first, p_last, p_n = pg["timestamp"].min(), pg["timestamp"].max(), pg.size()
p_views = pre["event"].eq("view").groupby(pk, sort=False).sum()
p_carts = pre["event"].eq("addtocart").groupby(pk, sort=False).sum()
p_last_cart = pre.loc[pre["event"].eq("addtocart"), "timestamp"].groupby(pk[pre["event"].eq("addtocart")], sort=False).max()
second_last = pre.loc[pg.cumcount(ascending=False) == 1].set_index("session_key")["timestamp"]
cart_pairs = pre.loc[pre["event"].eq("addtocart"), ["session_key", "itemid"]].drop_duplicates()
view_rows = pre.loc[pre["event"].eq("view"), ["session_key", "itemid"]]
repeated = view_rows.merge(cart_pairs, on=["session_key", "itemid"]).groupby("session_key").size()
r0 = risk.reindex(p_first.index)
cut = pd.to_datetime(r0.values, unit="ms")
prefix = pd.DataFrame({
    "visitorid": pg["visitorid"].first(),
    "start_ts": first.reindex(p_first.index),
    "view_count": p_views,
    "addtocart_count": p_carts,
    "unique_items_count": pg["itemid"].nunique(),
    "view_to_cart_ratio": p_views / p_carts,
    "seconds_since_first_event": (r0 - p_first) / 1000.0,
    "seconds_since_first_cart": 0.0,
    "seconds_since_last_cart": (r0 - p_last_cart.reindex(p_first.index)) / 1000.0,
    "seconds_since_previous_event": (r0 - second_last.reindex(p_first.index)) / 1000.0,
    "inter_event_interval_mean": np.where(p_n >= 2, (p_last - p_first) / 1000.0 / (p_n - 1).clip(lower=1), np.nan),
    "events_after_cart": (pre["timestamp"] > pk.map(t0)).groupby(pk, sort=False).sum(),
    "repeated_views_of_cart_items": repeated.reindex(p_first.index).fillna(0).astype(int),
    "hour_of_day": cut.hour,
    "day_of_week": cut.dayofweek,
})
prefix["y"] = (first_buy.reindex(prefix.index) > r0).fillna(False).astype(int)
assert (prefix["seconds_since_previous_event"].dropna() >= 0).all()

# ============================================================================
# CHECK: bulk features == paper 1's reference functions (3,000 sessions)
# ============================================================================
rng = np.random.default_rng(0)
sample = rng.choice(completed.index.values, size=CHECK_N, replace=False)
sample_ev = dict(tuple(ev[key.isin(set(sample))].groupby("session_key")))
mismatch = []
for sk in sample:
    ref = completed_features(sample_ev[sk], PURCHASE)
    for col, v in ref.items():
        if not np.isclose(completed.at[sk, col], v, atol=1.5e-3):
            mismatch.append(("completed", sk, col, completed.at[sk, col], v))
    if sk in prefix.index:
        ref = prefix_features(sample_ev[sk], t0[sk])
        for col, v in ref.items():
            if not np.isclose(prefix.at[sk, col], v, atol=1e-9, equal_nan=True):
                mismatch.append(("prefix", sk, col, prefix.at[sk, col], v))
    elif sk in t0.index and not bought_before.get(sk, False):
        mismatch.append(("prefix", sk, "missing row", None, None))
assert not mismatch, f"bulk features differ from paper 1's functions: {mismatch[:5]}"
print(f"check passed: {CHECK_N} sessions, every completed and prefix feature equals paper 1's functions")

# ============================================================================
# LABEL CHECKS (paper 1, Section IV-C)
# ============================================================================
paid = completed.index[completed["y"] == 1]
pairs = lambda mask: set(map(tuple, ev.loc[mask, ["session_key", "itemid"]].drop_duplicates().values))  # noqa: E731
carted, bought, removed = pairs(is_cart), pairs(is_buy), pairs(ev["event"].eq("remove_from_cart"))
paid_set = set(paid)
not_bought = {(s, i) for s, i in carted if s in paid_set and (s, i) not in bought}
label = {
    "cart_sessions": len(completed),
    "purchased_sessions": len(paid),
    "purchase_before_first_add_removed": int(bought_before.fillna(False).sum()),
    "purchased_sessions_with_carted_item_not_bought": len({s for s, _ in not_bought}),
    "purchased_sessions_with_carted_item_not_bought_not_removed": len({s for s, i in not_bought if (s, i) not in removed}),
    "first_add_rows": len(prefix),
    "first_add_positive_rate": round(float(prefix["y"].mean()), 4),
    "burst_rule_on_whole_month": BURST,
}
(RESULTS / f"{PREFIX}__label-checks__{TAG}.json").write_text(json.dumps(label, indent=2), encoding="utf-8")
print(json.dumps(label, indent=2))

# ============================================================================
# MODELS (paper 1's plan)
# ============================================================================
def main():
    """Fit every model in paper 1's plan and write the two result files."""
    runs = run_plan(completed, prefix[["visitorid", "start_ts", "y"] + PREFIX13])
    summary = summarize(runs, retail_rocket_reference(RESULTS))
    runs.round(4).to_csv(RESULTS / f"{PREFIX}__paper1-replication__{TAG}__runs.csv", index=False)
    summary.round(4).to_csv(RESULTS / f"{PREFIX}__paper1-replication__{TAG}__summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
