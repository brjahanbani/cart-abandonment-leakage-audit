"""
headline_cis.py
===============

95% confidence intervals for Paper 2's headline gaps: leaky vs fair ROC-AUC
on the same test set (paired bootstrap, 1,000 resamples), seed 42,
visitor-grouped split, XGBoost, each gap under the protocol that produced it:

  L2 completed-session timing   original_completed vs counts_only
                                paper 1's protocol (0-Shared/paper1_protocol.py)
  L4 history incl. the purchase base + history (whole period) vs base +
                                history (before the add); the Oct 2 probe's
                                protocol (200 trees, depth 4, no cap)
  L6 checkout steps             counts + checkout steps vs counts only;
                                paper 1's protocol (flower shop only)

Tables come from the scripts that produced the published numbers (imported;
their models run only under __main__). Retail Rocket's history table comes
from 6-Point-In-Time/probes/agg_probe.py, which prints its own table on import.

Output: 5-Results/paper2__headline-cis.csv
"""

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
for sub in ["0-Shared", "7-Own-Data", "8-REES46", "6-Point-In-Time/probes"]:
    sys.path.insert(0, str(ROOT / sub))
from paper1_protocol import COMPLETED_CONFIGS, models, prepare, split  # noqa: E402

SEED, N_BOOT = 42, 1000
OUT = ROOT / "5-Results" / "paper2__headline-cis.csv"


def paired_ci(y, p_leaky, p_fair):
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], p_leaky[i]) - roc_auc_score(y[i], p_fair[i]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def paper1_pair(df, leaky_cols, fair_cols, table):
    tr, te = split(df, "visitor_grouped", SEED, table)
    probs = []
    for cols in (leaky_cols, fair_cols):
        X_tr, X_te, y_tr, y_te = prepare(df, cols, tr, te)
        m = models(SEED, y_tr, extra=False)["XGB"]
        m.fit(X_tr, y_tr)
        probs.append(m.predict_proba(X_te)[:, 1])
    return y_te, probs[0], probs[1]


def probe_pair(df, leaky_cols, fair_cols):
    i_tr, i_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                      .split(df, groups=df["visitorid"]))
    y = df["y"].values
    probs = []
    for cols in (leaky_cols, fair_cols):
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                          scale_pos_weight=(y[i_tr] == 0).sum() / (y[i_tr] == 1).sum(),
                          random_state=SEED, eval_metric="logloss", n_jobs=-1)
        m.fit(df[cols].values[i_tr], y[i_tr])
        probs.append(m.predict_proba(df[cols].values[i_te])[:, 1])
    return y[i_te], probs[0], probs[1]


rows = []


def record(mech, dataset, y, pl, pf):
    lo, hi = paired_ci(y, pl, pf)
    al, af = roc_auc_score(y, pl), roc_auc_score(y, pf)
    rows.append({"mechanism": mech, "dataset": dataset, "auc_leaky": al, "auc_fair": af,
                 "gap": al - af, "gap_ci_lo": lo, "gap_ci_hi": hi, "n_test": len(y)})
    print(f"{mech:32s} {dataset:14s} {al:.3f} vs {af:.3f}  gap {al - af:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(y)}")


ORIG, COUNTS = COMPLETED_CONFIGS["original_completed"], COMPLETED_CONFIGS["counts_only"]

# --- Taobao (unrelated shop): separate run, own output file --------------------
#   python headline_cis.py taobao  ->  5-Results/paper2__headline-cis__taobao.csv
if len(sys.argv) > 1 and sys.argv[1] == "taobao":
    sys.argv = [sys.argv[0], "--dataset", "taobao"]
    import group_b_rees46 as gbt  # noqa: E402
    record("L2 completed-session timing", "taobao", *paper1_pair(gbt.a.completed, ORIG, COUNTS, "completed"))
    record("L4 history incl. the purchase", "taobao",
           *probe_pair(gbt.pref, gbt.PREFIX13 + ["item_cr_full", "vis_txns_full"],
                       gbt.PREFIX13 + ["item_cr_asof", "vis_txns_asof"]))
    pd.DataFrame(rows).round(4).to_csv(OUT.with_name("paper2__headline-cis__taobao.csv"), index=False)
    sys.exit(0)

# --- Retail Rocket -----------------------------------------------------------
rr = pd.read_csv(ROOT / "1-Data" / "completed-session" / "cart_session_features.csv")
rr["y"] = rr["has_transaction"].astype(int)
record("L2 completed-session timing", "retail_rocket", *paper1_pair(rr, ORIG, COUNTS, "completed"))
import agg_probe as rrp  # noqa: E402  (read-only probe; prints its table)
record("L4 history incl. the purchase", "retail_rocket",
       *probe_pair(rrp.pref, rrp.BASE + ["item_cr_full", "vis_txns_full"], rrp.BASE + ["item_cr_asof", "vis_txns_asof"]))
del rr, rrp
sys.modules.pop("agg_probe")  # free the Retail Rocket event log before REES46 loads
gc.collect()

# --- Flower shop ---------------------------------------------------------------
import extras_funnel_product as fx  # noqa: E402
import history_window_test as fh  # noqa: E402
record("L2 completed-session timing", "flower_shop", *paper1_pair(fx.completed, ORIG, COUNTS, "completed"))
record("L6 checkout steps", "flower_shop", *paper1_pair(fx.completed, COUNTS + fx.FUNNEL, COUNTS, "completed"))
record("L4 history incl. the purchase", "flower_shop",
       *probe_pair(fh.pref, fh.BASE + ["item_cr_full", "vis_txns_full"], fh.BASE + ["item_cr_asof", "vis_txns_asof"]))

# --- REES46, October clean window ----------------------------------------------
sys.argv = [sys.argv[0], "--from", "2019-10-09"]
import group_b_rees46 as gb  # noqa: E402
record("L2 completed-session timing", "rees46", *paper1_pair(gb.a.completed, ORIG, COUNTS, "completed"))
record("L4 history incl. the purchase", "rees46",
       *probe_pair(gb.pref, gb.PREFIX13 + ["item_cr_full", "vis_txns_full"],
                   gb.PREFIX13 + ["item_cr_asof", "vis_txns_asof"]))

pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
