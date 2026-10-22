"""
build_detection_matrix.py
=========================

Paper 2's backbone: for every leak mechanism and every dataset, is the leak
present, does paper 1's nine-question checklist flag it, and what does it
cost in ROC-AUC. Every number is read from a committed result file (or, for
the bot check, computed here from the cleaned events); nothing is typed in
except Retail Rocket's published label-check share and probe values, whose
sources are named.

Effect = XGBoost ROC-AUC, shown as the range over the splits that were run,
leaky configuration vs its fair comparator.

Burst check (the REES46 finding, applied to all three datasets), rule in
0-Shared/burst_check.py: a day is flagged above 1.5x the median daily number
of cart sessions; the traffic counts as automated when the flagged days'
purchase rate is below 60% of the other days'. Reported with the share of
sessions starting 00:00-05:59 (automated traffic shows a flat day).

Outputs: 5-Results/paper2__detection-matrix.csv, 5-Results/paper2__burst-check.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
import burst_check  # noqa: E402  (the rule, shared with the REES46 runs)

R = ROOT / "5-Results"
D = ROOT / "1-Data"

# Retail Rocket values not in a result file of this repo:
RR_MIXED_SHARE = "11.3% of purchased sessions (paper 1, Sec. IV-C; no remove event)"
RR_PROBE = R / "own-data__history-window.csv"  # carries the Oct 2 probe values as rr_* columns


def rng(values):
    lo, hi = min(values), max(values)
    return f"{lo:.2f}" if round(lo, 2) == round(hi, 2) else f"{lo:.2f}-{hi:.2f}"


# ============================================================================
# BURST CHECK
# ============================================================================
def burst(name, path, purchase, ts_unit="ms", offset_ms=0):
    """offset_ms shifts timestamps into the shop's local time before days and
    night hours are formed (Taobao: China, UTC+8; the others: as stored)."""
    if path.suffix == ".csv":
        ev = pd.read_csv(path, usecols=["timestamp", "event", "session_key"])
    else:
        ev = pd.read_parquet(path, columns=["timestamp", "event", "session_key"])
    ev["event"] = ev["event"].astype(str)
    g = ev.groupby("session_key")
    s = pd.DataFrame({"start": g["timestamp"].min(),
                      "cart": ev["event"].eq("addtocart").groupby(ev["session_key"]).any(),
                      "buy": ev["event"].eq(purchase).groupby(ev["session_key"]).any()})
    s = s[s["cart"]]
    summary, _ = burst_check.check(s["start"].values + offset_ms, s["buy"].values)
    return {"dataset": name, **summary}


bursts = pd.DataFrame([
    burst("retail_rocket", D / "events_clean.csv", "transaction"),
    burst("flower_shop", D / "own" / "clean" / "events_clean.parquet", "checkout_final"),
    burst("rees46", D / "rees46" / "clean" / "events_clean.parquet", "transaction"),
    burst("rees46_nov", D / "rees46" / "clean" / "2019-Nov" / "events_clean.parquet", "transaction"),
    burst("taobao", D / "taobao" / "clean" / "events_clean.parquet", "transaction", offset_ms=8 * 3600 * 1000),
])
bursts.to_csv(R / "paper2__burst-check.csv", index=False)

# ============================================================================
# EFFECTS FROM RESULT FILES
# ============================================================================
rr = pd.read_csv(R / "split_robustness_summary.csv")
own = pd.read_csv(R / "own-data__paper1-replication__summary.csv")
ree = pd.read_csv(R / "rees46__paper1-replication__from-2019-10-09__summary.csv")
ree_all = pd.read_csv(R / "rees46__paper1-replication__all-october__summary.csv")
own_funnel = pd.read_csv(R / "own-data__extras-funnel-product.csv")
own_hist = pd.read_csv(RR_PROBE)
ree_hist = pd.read_csv(R / "rees46__history-window.csv")
own_tables = json.loads((R / "own-data__tables.json").read_text(encoding="utf-8"))
ree_label = json.loads((R / "rees46__label-checks__from-2019-10-09.json").read_text())
# November 2019: held-out month, opened after the rules and thresholds were committed.
nov = pd.read_csv(R / "rees46__paper1-replication__2019-Nov__all-month__summary.csv")
nov_hist = pd.read_csv(R / "rees46__history-window__2019-Nov__all-month.csv")
nov_label = json.loads((R / "rees46__label-checks__2019-Nov__all-month.json").read_text())
# Taobao: the unrelated shop, opened after every rule and threshold was fixed.
tao = pd.read_csv(R / "taobao__paper1-replication__all__summary.csv")
tao_hist = pd.read_csv(R / "taobao__history-window__all.csv")
tao_label = json.loads((R / "taobao__label-checks__all.json").read_text())


def xgb(df, config, col="roc_auc", cfg_col="config", model_col="model", table=None):
    m = (df[cfg_col] == config) & (df[model_col] == "XGB")
    if table is not None and "table" in df:
        m &= df["table"] == table
    return df.loc[m, col].tolist()


def hist(df, window, feats, prefix=""):
    row = df[(df["window"] == window) & (df["features"] == feats)].iloc[0]
    return [row[f"{prefix}visitor_auc"], row[f"{prefix}temporal_auc"]]


def first_add_range(df):
    return df.loc[(df["table"] == "first_add_k0"), "roc_auc"].tolist()


rows = []


def add(mech, dataset, present, v1_q, v1_flags, leaky="", fair="", note=""):
    rows.append({"mechanism": mech, "dataset": dataset, "present": present, "v1_question": v1_q,
                 "v1_flags": v1_flags, "auc_leaky": leaky, "auc_fair": fair, "note": note})


# L1 exact count reconstruction (Q4)
add("L1 count reconstructs label", "retail_rocket", "yes (exact)", "Q4", "yes",
    rng(xgb(rr, "target_proxy", "roc_auc_mean")), rng(xgb(rr, "counts_only", "roc_auc_mean")),
    "event - view - cart = transactions in every session")
add("L1 count reconstructs label", "flower_shop", "no (identity holds in 0% of sessions)", "Q4", "no (correct)",
    rng(xgb(own, "target_proxy", table="completed")), rng(xgb(own, "counts_only", table="completed")),
    "extra event types break the identity; residual gain comes from post-decision events in the total")
add("L1 count reconstructs label", "rees46", "approximate (= purchases + removals)", "Q4", "partial",
    rng(xgb(ree, "target_proxy", table="completed")), rng(xgb(ree, "counts_only", table="completed")),
    "Q4 asks for exact reconstruction; a near-identity is only caught by a careful reading")
add("L1 count reconstructs label", "rees46_nov", "approximate (= purchases + removals)", "Q4", "partial",
    rng(xgb(nov, "target_proxy", table="completed")), rng(xgb(nov, "counts_only", table="completed")),
    "held-out month")
add("L1 count reconstructs label", "taobao", "approximate (= purchases + favourites)", "Q4", "partial",
    rng(xgb(tao, "target_proxy", table="completed")), rng(xgb(tao, "counts_only", table="completed")),
    "unrelated shop; favourites are the extra event type")

# L2 completed-endpoint timing (Q5, Q6)
for ds, df, col in (("retail_rocket", rr, "roc_auc_mean"), ("flower_shop", own, "roc_auc"), ("rees46", ree, "roc_auc"),
                    ("rees46_nov", nov, "roc_auc"), ("taobao", tao, "roc_auc")):
    kw = {} if ds == "retail_rocket" else {"table": "completed"}
    add("L2 completed-session timing", ds, "yes", "Q5, Q6", "yes",
        rng(xgb(df, "original_completed", col, **kw)), rng(xgb(df, "counts_only", col, **kw)),
        f"timing alone {rng(xgb(df, 'timing_only', col, **kw))}")

# L3 session label vs decision (Q3)
add("L3 label mixes purchase and abandonment", "retail_rocket", "yes", "Q3", "yes", note=RR_MIXED_SHARE)
add("L3 label mixes purchase and abandonment", "flower_shop", "yes (small)", "Q3", "yes",
    note=f"{own_tables['item_check_carted_not_removed_not_bought']} of {own_tables['item_check_paid_sessions']} "
         "paid sessions, not removed first")
add("L3 label mixes purchase and abandonment", "rees46", "yes", "Q3", "yes",
    note=f"{ree_label['purchased_sessions_with_carted_item_not_bought_not_removed']} of "
         f"{ree_label['purchased_sessions']} purchased sessions, not removed first")
add("L3 label mixes purchase and abandonment", "rees46_nov", "yes", "Q3", "yes",
    note=f"{nov_label['purchased_sessions_with_carted_item_not_bought_not_removed']} of "
         f"{nov_label['purchased_sessions']} purchased sessions, not removed first")
add("L3 label mixes purchase and abandonment", "taobao", "yes", "Q3", "yes",
    note=f"{tao_label['purchased_sessions_with_carted_item_not_bought']} of "
         f"{tao_label['purchased_sessions']} purchased sessions (no remove event)")

# L4 self-inclusion and L5 future window (history aggregates) - not in the nine questions
for ds, df, pre in (("retail_rocket", own_hist, "rr_"), ("flower_shop", own_hist, ""), ("rees46", ree_hist, ""),
                    ("rees46_nov", nov_hist, ""), ("taobao", tao_hist, "")):
    vcol = "rr_visitor_auc" if pre else "visitor_auc"
    tcol = "rr_temporal_auc" if pre else "temporal_auc"

    def h(w, f):
        r = df[(df["window"] == w) & (df["features"] == f)].iloc[0]
        return [r[vcol], r[tcol]]
    add("L4 history includes the scored purchase", ds, "yes", "none", "no (missed)",
        rng(h("full", "base+agg")), rng(h("asof", "base+agg")), "base + history; fair = history before the add")
    gain = sum(h("full_loo", "base+agg")) / 2 - sum(h("asof", "base+agg")) / 2
    add("L5 history includes later purchases", ds, "yes" if gain >= 0.03 else "weak", "none", "no (missed)",
        rng(h("full_loo", "base+agg")), rng(h("asof", "base+agg")),
        f"whole period minus this visit; mean gain {gain:+.3f} (weak below +0.03)")

# L6 post-decision funnel steps - only the flower shop logs checkout steps
add("L6 checkout steps after the prediction moment", "retail_rocket", "n/a (not logged)", "none", "n/a")
add("L6 checkout steps after the prediction moment", "flower_shop", "yes", "none", "no (missed)",
    rng(xgb(own_funnel, "counts_plus_funnel")), rng(xgb(own_funnel, "counts_only")))
add("L6 checkout steps after the prediction moment", "rees46", "n/a (not logged)", "none", "n/a")
add("L6 checkout steps after the prediction moment", "rees46_nov", "n/a (not logged)", "none", "n/a")
add("L6 checkout steps after the prediction moment", "taobao", "n/a (not logged)", "none", "n/a")

# L7 automated traffic the bot rules cannot see
for _, b in bursts.iterrows():
    present = "yes" if b["automated"] else "no"
    leaky = fair = ""
    if b["dataset"] == "rees46":
        leaky = rng(ree_all.loc[(ree_all["table"] == "first_add_k0") & (ree_all["split"] == "visitor_grouped"), "roc_auc"])
        fair = rng(ree.loc[(ree["table"] == "first_add_k0") & (ree["split"] == "visitor_grouped"), "roc_auc"])
    add("L7 automated traffic past the bot filter", b["dataset"], present, "none", "no (missed)" if present == "yes" else "n/a",
        leaky, fair, f"{b['flagged_days']} flagged days; purchase rate {b['purchase_rate_flagged']} vs "
                     f"{b['purchase_rate_other']}; night share {b['night_share_flagged']} vs {b['night_share_other']}")

matrix = pd.DataFrame(rows)
matrix.to_csv(R / "paper2__detection-matrix.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(bursts.to_string(index=False))
print(matrix.drop(columns="note").to_string(index=False))
