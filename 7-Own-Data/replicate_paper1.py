"""
replicate_paper1.py
===================

Group A of Paper 2: paper 1's evaluations repeated on the flower-shop data,
with nothing added and nothing left out. Run after build_own_events.py and
build_own_tables.py.

Every protocol choice is copied from the paper-1 script that produced the
published number:

  COMPLETED-SESSION TABLE  (paper 1 Tables IV-VI)
    features   2-Completed-Session/build_cart_session_table.py,
               build_session_features (transaction -> checkout_final)
    configs    target_proxy, original_completed, timing_only, counts_only
    splits     2-Completed-Session/reanalysis_split_robustness.py:
               stratified 80/20 | GroupShuffleSplit on the visitor id |
               temporal by session start, cut = round(0.8 n)
    models     LR (balanced, max_iter 2000, train-only StandardScaler) and
               XGBoost (100 trees, hist, scale_pos_weight), seeds 11..55;
               DT (balanced, depth 8) + RF (100 trees, balanced) at seed 42,
               original_completed, stratified only (Table IV)

  FIRST ADD-TO-CART, k=0  (paper 1 Section VI)
    features   3-Decision-Time/build_prefix_dataset.py, the 13 prefix features
    risk set   sessions with a purchase at or before the first add are dropped
    label      a purchase strictly after the first add
    splits     3-Decision-Time/train_prefix_models.py: GroupShuffleSplit on
               the visitor id | temporal by session start, n_train = int(0.8 n)
    models     LR + XGB at seeds 11..55; DT + RF at seed 42 per split;
               bootstrap 95% CI (1,000 resamples) for LR and XGB at seed 42

  Both: 99th-percentile cap fitted on the training part only, missing -> 0.

The feature functions and the whole protocol live in 0-Shared/paper1_protocol.py,
shared with the REES46 replication (8-REES46/).

Unavoidable data differences (reported, not removed): the purchase event is
`checkout_final`; sessions also contain category views, searches, removes and
checkout steps; the visitor id is the browser id (`anonymousId`), as Retail
Rocket's `visitorid` is; sessions start from 2026-08-23 (complete tracking).

Outputs (aggregate only):
  5-Results/own-data__paper1-replication__runs.csv     one row per fitted model
  5-Results/own-data__paper1-replication__summary.csv  mean/sd + Retail Rocket
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from paper1_protocol import (  # noqa: E402,F401  (re-exported for group B scripts)
    COMPLETED_CONFIGS, PREFIX13, SEEDS, completed_features, models, prefix_features,
    prepare, retail_rocket_reference, run_plan, split, summarize)

CLEAN = ROOT / "1-Data" / "own" / "clean"
RESULTS = ROOT / "5-Results"
PURCHASE = "checkout_final"

# ============================================================================
# DATA: cart sessions from the day tracking became complete
# ============================================================================
ev = pd.read_parquet(CLEAN / "events_clean.parquet")
cart = pd.read_parquet(CLEAN / "cart_sessions_labels.parquet").set_index("session_key")
window_from = int(ev.loc[ev["user_agent"].notna(), "timestamp"].min())
cart = cart[cart["first_add_ts"] >= window_from]
ev = ev[ev["session_key"].isin(cart.index)].sort_values(["session_key", "timestamp"], kind="stable")
sessions = dict(tuple(ev.groupby("session_key")))

meta = pd.DataFrame({
    "visitorid": cart["visitorid"],
    "start_ts": {k: g["timestamp"].min() for k, g in sessions.items()},
})
completed = meta.join(pd.DataFrame({k: completed_features(g, PURCHASE) for k, g in sessions.items()}).T)

rows, dropped = {}, 0
for k, g in sessions.items():
    t0 = cart.at[k, "first_add_ts"]
    purchase_ts = g.loc[g["event"] == PURCHASE, "timestamp"].values
    if (purchase_ts <= t0).any():
        dropped += 1  # already purchased at or before the decision point
        continue
    rows[k] = {**prefix_features(g, t0), "y": int((purchase_ts > t0).any())}
prefix = meta.join(pd.DataFrame(rows).T, how="inner")
for df in (completed, prefix):
    for col in df.columns.drop("visitorid"):
        df[col] = pd.to_numeric(df[col])
assert "event_count" not in prefix.columns
assert (prefix["seconds_since_previous_event"].dropna() >= 0).all()


def main():
    """Fit every model in paper 1's plan and write the two result files."""
    runs = run_plan(completed, prefix)
    summary = summarize(runs, retail_rocket_reference(RESULTS))
    runs.round(4).to_csv(RESULTS / "own-data__paper1-replication__runs.csv", index=False)
    summary.round(4).to_csv(RESULTS / "own-data__paper1-replication__summary.csv", index=False)
    print(f"window from {pd.to_datetime(window_from, unit='ms')}; completed sessions {len(completed)} "
          f"(paid {completed['y'].mean():.4f}); first-add rows {len(prefix)} "
          f"(positive {prefix['y'].mean():.4f}), dropped (purchase before first add) {dropped}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
