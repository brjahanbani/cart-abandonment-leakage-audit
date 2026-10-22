"""
negative_control.py
===================

Paper 2 gap 5: does the checklist stay quiet when there is no leak? The
detection matrix holds only one true negative for checklist v1, so this
script builds negative controls: feature sets constructed correctly (every
value known at the first add-to-cart), on every dataset.

  prefix13         paper 1's 13 point-in-time features
  prefix13_hist    + item conversion rate and visitor purchase count counted
                   strictly before the first add (the Oct 2 probe's "asof")
  prefix13_product + product facts known at the first add (flower shop,
                   REES46)

Two checks per configuration (group A protocol: visitor-grouped split,
99th-percentile cap fitted on train, XGBoost, seeds 11..55):
  true AUC         reference
  permuted AUC     training labels shuffled; a pipeline that cannot see the
                   label (no leak through features, split or preprocessing)
                   must score about 0.50 on the true test labels
  max single-feature AUC   the strongest single feature, max(AUC, 1 - AUC);
                   a leak usually shows up as one near-perfect feature

The checklist questions are answered for these configurations in
docs/paper2/negative_control.md (they are answered from how each feature is
built, which is the same on every dataset).

Run once per dataset (separate processes keep memory low):
  python negative_control.py retail_rocket | flower_shop | rees46_oct | rees46_nov | taobao
Output: 5-Results/paper2__negative-control__<dataset>.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
for sub in ["0-Shared", "7-Own-Data", "8-REES46", "6-Point-In-Time/probes"]:
    sys.path.insert(0, str(ROOT / sub))
from paper1_protocol import PREFIX13, SEEDS, models, prepare, split  # noqa: E402

DATASET = sys.argv[1]
HIST = ["item_cr_asof", "vis_txns_asof"]

if DATASET == "retail_rocket":
    import agg_probe as src  # noqa: E402  (prints its own table)
    pref = src.pref.copy()
    configs = {"prefix13": PREFIX13, "prefix13_hist": PREFIX13 + HIST}
elif DATASET == "flower_shop":
    import extras_funnel_product as fx  # noqa: E402
    import history_window_test as fh  # noqa: E402
    pref = fh.pref.join(fx.prefix[fx.PRODUCT])
    configs = {"prefix13": PREFIX13, "prefix13_hist": PREFIX13 + HIST,
               "prefix13_product": PREFIX13 + fx.PRODUCT}
else:
    sys.argv = [sys.argv[0]] + {"rees46_oct": ["--from", "2019-10-09"], "rees46_nov": ["--month", "2019-Nov"],
                                "taobao": ["--dataset", "taobao"]}[DATASET]
    import group_b_rees46 as gb  # noqa: E402
    pref = gb.pref
    configs = {"prefix13": PREFIX13, "prefix13_hist": PREFIX13 + HIST,
               "prefix13_product": PREFIX13 + gb.PRODUCT}

pref = pref.copy()
pref["y"] = pref["y"].astype(int)
y_all = pref["y"].values
rows = []
for name, cols in configs.items():
    true_auc, perm_auc = [], []
    for seed in SEEDS:
        tr, te = split(pref, "visitor_grouped", seed, "first_add_k0")
        X_tr, X_te, y_tr, y_te = prepare(pref, cols, tr, te)
        for target, store in ((y_tr, true_auc), (np.random.default_rng(seed).permutation(y_tr), perm_auc)):
            m = models(seed, target, extra=False)["XGB"]
            m.fit(X_tr, target)
            store.append(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]))
    single = {c: roc_auc_score(y_all, pref[c].fillna(0).values) for c in cols
              if pref[c].fillna(0).nunique() > 1}
    best = max(single, key=lambda c: max(single[c], 1 - single[c]))
    rows.append({"dataset": DATASET, "config": name, "n": len(pref),
                 "true_auc": np.mean(true_auc), "true_sd": np.std(true_auc),
                 "permuted_auc": np.mean(perm_auc), "permuted_sd": np.std(perm_auc),
                 "permuted_min": np.min(perm_auc), "permuted_max": np.max(perm_auc),
                 "max_single_feature": best,
                 "max_single_feature_auc": max(single[best], 1 - single[best])})

out = pd.DataFrame(rows)
out.round(4).to_csv(ROOT / "5-Results" / f"paper2__negative-control__{DATASET}.csv", index=False)
print(out.round(3).to_string(index=False))
