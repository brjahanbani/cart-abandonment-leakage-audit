"""
pr_auc_table.py
===============

Paper 1 reported PR-AUC next to ROC-AUC; Paper 2 reports ROC-AUC in the text. This
table gives the PR-AUC of the Group A configurations on every dataset (XGBoost,
visitor-grouped split, mean over the five seeds), with the test positive rate, which
is the PR-AUC of an uninformative model. No model is fitted here: the values come
from the run files that the Group A scripts already wrote.

Output: 5-Results/paper2__pr-auc.csv
"""

import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "5-Results"
RUNS = {"flower_shop": "own-data__paper1-replication__runs.csv",
        "rees46_oct": "rees46__paper1-replication__from-2019-10-09__runs.csv",
        "rees46_nov": "rees46__paper1-replication__2019-Nov__all-month__runs.csv",
        "taobao": "taobao__paper1-replication__all__runs.csv"}
CONFIGS = [("completed", "target_proxy"), ("completed", "original_completed"),
           ("completed", "timing_only"), ("completed", "counts_only"), ("first_add_k0", "prefix13")]

rows = []
# Retail Rocket: paper 1's own run files, one JSON per seed.
rr_files = {("completed", c): f"completed-session__xgboost__visitor-grouped__{c.replace('_', '-')}__s*.json"
            for _, c in CONFIGS[:4]}
rr_files[("first_add_k0", "prefix13")] = "decision-time__xgboost__visitor-grouped-all-eligible__k0__s*.json"
for (table, config), pattern in rr_files.items():
    runs = [json.loads(p.read_text()) for p in sorted(RES.glob(pattern))]
    rows.append({"dataset": "retail_rocket", "table": table, "config": config, "seeds": len(runs),
                 "roc_auc": sum(r["roc_auc"] for r in runs) / len(runs),
                 "pr_auc": sum(r["pr_auc"] for r in runs) / len(runs),
                 "test_positive_rate": sum(r.get("test_positive_rate", r.get("positive_rate", float("nan")))
                                           for r in runs) / len(runs)})
for name, f in RUNS.items():
    d = pd.read_csv(RES / f)
    d = d[(d["model"] == "XGB") & (d["split"] == "visitor_grouped")]
    for table, config in CONFIGS:
        s = d[(d["table"] == table) & (d["config"] == config)]
        rows.append({"dataset": name, "table": table, "config": config, "seeds": len(s),
                     "roc_auc": s["roc_auc"].mean(), "pr_auc": s["pr_auc"].mean(),
                     "test_positive_rate": s["test_positive_rate"].mean()})
out = pd.DataFrame(rows)
out.round(4).to_csv(RES / "paper2__pr-auc.csv", index=False)
print(out.pivot(index="config", columns="dataset", values="pr_auc").round(3).to_string())
print(out.pivot(index="config", columns="dataset", values="test_positive_rate").round(3).to_string())
