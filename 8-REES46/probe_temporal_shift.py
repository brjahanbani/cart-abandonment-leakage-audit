"""
probe_temporal_shift.py
=======================

Diagnostic for one group-A result: on REES46 the first-add-to-cart models
score 0.59-0.66 under the visitor-grouped split but 0.48-0.53 under the
temporal split (latest 20% of sessions = the last days of October).
Read-only: prints per-day volumes and purchase rates, and each prefix
feature's own ROC-AUC in the training period vs the test period, to see
whether the relationship between features and purchase changes over time.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from paper1_protocol import PREFIX13, split  # noqa: E402

import replicate_paper1_rees46 as r  # noqa: E402  (builds the tables only; models run under __main__)

prefix = r.prefix
day = pd.to_datetime(prefix["start_ts"], unit="ms").dt.date
daily = prefix.groupby(day)["y"].agg(["size", "mean"]).rename(columns={"size": "rows", "mean": "purchase_rate"})
print(daily.round(3).to_string())

tr, te = split(prefix, "temporal", 0, "first_add_k0")
print("\ntest period starts", pd.to_datetime(prefix["start_ts"].values[te].min(), unit="ms"))
rows = []
for col in PREFIX13:
    x, y = prefix[col].fillna(0).values, prefix["y"].values
    if np.unique(x).size < 2:
        continue
    rows.append({"feature": col,
                 "auc_train_period": roc_auc_score(y[tr], x[tr]),
                 "auc_test_period": roc_auc_score(y[te], x[te])})
print(pd.DataFrame(rows).round(3).to_string(index=False))
