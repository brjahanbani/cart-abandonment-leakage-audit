"""
headline_cis.py
===============

Paired gaps for Paper 2's leak mechanisms: leaky vs fair ROC-AUC on the same test
sessions, XGBoost, visitor-grouped split. For each gap:

  gap, gap_ci_lo/hi          seed 42; paired bootstrap over test sessions (1,000)
  gap_ci_vis_lo/hi           seed 42; paired bootstrap over test visitors (1,000),
                             which keeps a visitor's sessions together
  gap_seed_min/max           the gap at seeds 11, 22, 33, 42, 55 (each seed draws
                             its own visitor-grouped split and model)

Each comparator differs from the leaky set only in the leak (revised 2026-10-27,
after the ChatGPT review, G1):

  L1 count reconstructs     event, view and add-to-cart counts vs view and
     the label              add-to-cart counts (only the event count differs)
  L2 completed-session      original completed set vs counts only
     timing
  L2 last-add recency       original completed set vs the same set without the
                            time from the last add-to-cart to the session's end
                            (the other timing features carry the same endpoint)
  L2 not ending on a cart   the L2 models, scored only on test sessions that do
                            not end within 3 s of an add-to-cart (within-dataset
                            test of the ending explanation; 3 s also covers the
                            flower shop's automatic cart-page event)
  L4 history incl. the      history over the whole period vs the whole period
     scored purchase        minus the scored session (only the scored purchase)
  L5 history incl. later    whole period minus the scored session vs history
     purchases              before the first add-to-cart
  L4+L5 whole-period        whole period vs before the first add-to-cart: the
     history                "lifetime purchases" column as usually built (the
                            L4 definition before the revision)
  L6 checkout steps         counts + checkout steps vs counts only (flower shop)

L1, L2 and L6 follow paper 1's protocol (0-Shared/paper1_protocol.py); the history
gaps follow the Oct 2 probe's (200 trees, depth 4, no cap). Tables come from the
scripts that produced the published numbers (imported; their models run only under
__main__). Retail Rocket's history table comes from 6-Point-In-Time/probes/agg_probe.py,
which prints its own table on import.

Output: 5-Results/paper2__headline-cis.csv (Retail Rocket, flower shop, REES46 Oct);
the datasets opened last are separate runs with their own files (see below).
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
from paper1_protocol import COMPLETED_CONFIGS, SEEDS, models, prepare, split  # noqa: E402

SEED, N_BOOT = 42, 1000
OUT = ROOT / "5-Results" / "paper2__headline-cis.csv"


def paired_ci(y, p_leaky, p_fair, groups=None):
    """Percentile interval of the AUC difference; resamples sessions, or visitors if given."""
    rng = np.random.default_rng(SEED)
    if groups is None:
        draws = (rng.integers(0, len(y), size=len(y)) for _ in range(N_BOOT))
    else:
        codes, uniq = pd.factorize(groups)
        members = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
        draws = (np.concatenate([members[k] for k in rng.integers(0, len(uniq), size=len(uniq))])
                 for _ in range(N_BOOT))
    d = []
    for i in draws:
        if len(np.unique(y[i])) == 2:
            d.append(roc_auc_score(y[i], p_leaky[i]) - roc_auc_score(y[i], p_fair[i]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def paper1_pair(df, leaky_cols, fair_cols, table, seed=SEED, keep=None):
    """keep: optional boolean column of df; the test sessions are then limited to it."""
    tr, te = split(df, "visitor_grouped", seed, table)
    probs = []
    for cols in (leaky_cols, fair_cols):
        X_tr, X_te, y_tr, y_te = prepare(df, cols, tr, te)
        m = models(seed, y_tr, extra=False)["XGB"]
        m.fit(X_tr, y_tr)
        probs.append(m.predict_proba(X_te)[:, 1])
    k = df[keep].values[te] if keep else np.ones(len(te), bool)
    return y_te[k], probs[0][k], probs[1][k], df["visitorid"].values[te][k]


def probe_pair(df, leaky_cols, fair_cols, seed=SEED):
    i_tr, i_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                      .split(df, groups=df["visitorid"]))
    y = df["y"].values
    probs = []
    for cols in (leaky_cols, fair_cols):
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                          scale_pos_weight=(y[i_tr] == 0).sum() / (y[i_tr] == 1).sum(),
                          random_state=seed, eval_metric="logloss", n_jobs=-1)
        m.fit(df[cols].values[i_tr], y[i_tr])
        probs.append(m.predict_proba(df[cols].values[i_te])[:, 1])
    return y[i_te], probs[0], probs[1], df["visitorid"].values[i_te]


rows = []


def record(mech, dataset, pair):
    """pair(seed) -> (y, p_leaky, p_fair, visitor ids of the test sessions)."""
    gaps = {}
    for s in SEEDS:
        y, pl, pf, vis = pair(s)
        gaps[s] = roc_auc_score(y, pl) - roc_auc_score(y, pf)
        if s == SEED:
            al, af = roc_auc_score(y, pl), roc_auc_score(y, pf)
            lo, hi = paired_ci(y, pl, pf)
            vlo, vhi = paired_ci(y, pl, pf, groups=vis)
            n, n_vis = len(y), len(np.unique(vis))
    rows.append({"mechanism": mech, "dataset": dataset, "auc_leaky": al, "auc_fair": af,
                 "gap": al - af, "gap_ci_lo": lo, "gap_ci_hi": hi,
                 "gap_ci_vis_lo": vlo, "gap_ci_vis_hi": vhi,
                 "gap_seed_min": min(gaps.values()), "gap_seed_max": max(gaps.values()),
                 "n_test": n, "n_test_visitors": n_vis})
    print(f"{mech:32s} {dataset:14s} {al:.3f} vs {af:.3f}  gap {al - af:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] visitors [{vlo:+.3f}, {vhi:+.3f}] "
          f"seeds {min(gaps.values()):+.3f}..{max(gaps.values()):+.3f}  n={n}", flush=True)


ORIG, COUNTS = COMPLETED_CONFIGS["original_completed"], COMPLETED_CONFIGS["counts_only"]
PROXY = COMPLETED_CONFIGS["target_proxy"]
PROXY_FAIR = [c for c in PROXY if c != "event_count"]
NO_RECENCY = [c for c in ORIG if c != "recency_last_cart_seconds"]


def completed_rows(name, comp, l1=True):
    if l1:
        record("L1 count reconstructs the label", name,
               lambda s: paper1_pair(comp, PROXY, PROXY_FAIR, "completed", s))
    record("L2 completed-session timing", name,
           lambda s: paper1_pair(comp, ORIG, COUNTS, "completed", s))
    record("L2 last-add recency", name,
           lambda s: paper1_pair(comp, ORIG, NO_RECENCY, "completed", s))
    comp = comp.assign(not_ending_on_cart=comp["recency_last_cart_seconds"] > 3)
    record("L2 not ending on a cart", name,
           lambda s: paper1_pair(comp, ORIG, COUNTS, "completed", s, keep="not_ending_on_cart"))


def history_rows(name, pref, base):
    w = {k: base + [f"item_cr_{k}", f"vis_txns_{k}"] for k in ("full", "full_loo", "asof")}
    record("L4 history incl. the scored purchase", name, lambda s: probe_pair(pref, w["full"], w["full_loo"], s))
    record("L5 history incl. later purchases", name, lambda s: probe_pair(pref, w["full_loo"], w["asof"], s))
    record("L4+L5 whole-period history", name, lambda s: probe_pair(pref, w["full"], w["asof"], s))


# --- Datasets opened last: separate runs, own output files --------------------
#   python headline_cis.py taobao      ->  5-Results/paper2__headline-cis__taobao.csv
#   python headline_cis.py rees46_nov  ->  5-Results/paper2__headline-cis__rees46_nov.csv
SEPARATE = {"taobao": ["--dataset", "taobao"], "rees46_nov": ["--month", "2019-Nov"]}
if len(sys.argv) > 1 and sys.argv[1] in SEPARATE:
    name = sys.argv[1]
    sys.argv = [sys.argv[0]] + SEPARATE[name]
    import group_b_rees46 as gbt  # noqa: E402
    completed_rows(name, gbt.a.completed)
    history_rows(name, gbt.pref, gbt.PREFIX13)
    pd.DataFrame(rows).round(4).to_csv(OUT.with_name(f"paper2__headline-cis__{name}.csv"), index=False)
    sys.exit(0)

# --- Retail Rocket -----------------------------------------------------------
rr = pd.read_csv(ROOT / "1-Data" / "completed-session" / "cart_session_features.csv")
rr["y"] = rr["has_transaction"].astype(int)
completed_rows("retail_rocket", rr)
import agg_probe as rrp  # noqa: E402  (read-only probe; prints its table)
history_rows("retail_rocket", rrp.pref, rrp.BASE)
del rr, rrp
sys.modules.pop("agg_probe")  # free the Retail Rocket event log before REES46 loads
gc.collect()

# --- Flower shop (no L1: the count identity does not hold there) ---------------
import extras_funnel_product as fx  # noqa: E402
import history_window_test as fh  # noqa: E402
completed_rows("flower_shop", fx.completed, l1=False)
record("L6 checkout steps", "flower_shop",
       lambda s: paper1_pair(fx.completed, COUNTS + fx.FUNNEL, COUNTS, "completed", s))
history_rows("flower_shop", fh.pref, fh.BASE)

# --- REES46, October clean window ----------------------------------------------
sys.argv = [sys.argv[0], "--from", "2019-10-09"]
import group_b_rees46 as gb  # noqa: E402
completed_rows("rees46", gb.a.completed)
history_rows("rees46", gb.pref, gb.PREFIX13)

pd.DataFrame(rows).round(4).to_csv(OUT, index=False)
