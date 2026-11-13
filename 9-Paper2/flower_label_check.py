"""
flower_label_check.py
=====================

Paper 2, flower shop: how far does the label used (a final checkout event in the
session) agree with the shop's paid orders, on the cart sessions the analysis uses,
and do the flower-shop gaps change when the paid-order label is used instead?

  agreement   2 x 2 table, event label vs paid-order label, analysed sessions only
  gaps        L2, L6 (completed table) and L4, L4+L5 (history table), XGBoost,
              seed 42, visitor-grouped split, with each label; protocols as in
              headline_cis.py

Output: 5-Results/paper2__flower-label-check.csv (gaps) and
        5-Results/paper2__flower-label-agreement.csv (2 x 2 counts)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
for sub in ["0-Shared", "7-Own-Data"]:
    sys.path.insert(0, str(ROOT / sub))
from paper1_protocol import COMPLETED_CONFIGS, models, prepare, split  # noqa: E402
import extras_funnel_product as fx  # noqa: E402  (imports replicate_paper1 as fx.a)
import history_window_test as fh  # noqa: E402

SEED = 42
RES = ROOT / "5-Results"
cart = fx.a.cart  # analysed cart sessions, indexed by session_key

# --- Agreement on the analysed sessions --------------------------------------------
tab = pd.crosstab(cart["y_final"], cart["y_paid_order"]).rename_axis(
    index="final_event", columns="paid_order")
tab.to_csv(RES / "paper2__flower-label-agreement.csv")
print(f"analysed cart sessions {len(cart)}; agreement {(cart['y_final'] == cart['y_paid_order']).mean():.4f}")
print(tab)


def completed_gap(df, leaky, fair):
    tr, te = split(df, "visitor_grouped", SEED, "completed")
    p = []
    for cols in (leaky, fair):
        X_tr, X_te, y_tr, y_te = prepare(df, cols, tr, te)
        m = models(SEED, y_tr, extra=False)["XGB"]
        m.fit(X_tr, y_tr)
        p.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    return p


def history_gap(df, leaky, fair):
    i_tr, i_te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                      .split(df, groups=df["visitorid"]))
    y = df["y"].values
    p = []
    for cols in (leaky, fair):
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                          scale_pos_weight=(y[i_tr] == 0).sum() / (y[i_tr] == 1).sum(),
                          random_state=SEED, eval_metric="logloss", n_jobs=-1)
        m.fit(df[cols].values[i_tr], y[i_tr])
        p.append(roc_auc_score(y[i_te], m.predict_proba(df[cols].values[i_te])[:, 1]))
    return p


ORIG, COUNTS = COMPLETED_CONFIGS["original_completed"], COMPLETED_CONFIGS["counts_only"]
w = {k: fh.BASE + [f"item_cr_{k}", f"vis_txns_{k}"] for k in ("full", "full_loo", "asof")}
comp, pref = fx.completed.copy(), fh.pref.copy()
rows = []
for label in ("final_event", "paid_order"):
    if label == "paid_order":
        comp["y"] = cart.loc[comp.index, "y_paid_order"].astype(int).values
        pref["y"] = cart.loc[pref.index, "y_paid_order"].astype(int).values
    for mech, (leaky, fair) in {
        "L2 completed-session timing": completed_gap(comp, ORIG, COUNTS),
        "L6 checkout steps": completed_gap(comp, COUNTS + fx.FUNNEL, COUNTS),
        "L4 history incl. the scored purchase": history_gap(pref, w["full"], w["full_loo"]),
        "L4+L5 whole-period history": history_gap(pref, w["full"], w["asof"]),
    }.items():
        rows.append({"label": label, "mechanism": mech, "auc_leaky": leaky, "auc_fair": fair,
                     "gap": leaky - fair, "positive_rate_completed": comp["y"].mean()})
        print(f"{label:12s} {mech:38s} {leaky:.3f} vs {fair:.3f}  gap {leaky - fair:+.3f}")
pd.DataFrame(rows).round(4).to_csv(RES / "paper2__flower-label-check.csv", index=False)
