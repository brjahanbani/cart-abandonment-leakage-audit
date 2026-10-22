# Reproducing Paper 2

One command, after the inputs below are in place:

```
python 9-Paper2/run_all.py          # 24 steps; --list shows them, --from N resumes
```

Every step is the script that produced a committed file in `5-Results/`. A rerun should
reproduce those files exactly. The 24 steps write 37 files (`own-data__*`, `rees46__*`,
`taobao__*`, `paper2__*`). Two committed files are not part of the run:
`rees46__profile.json` and `taobao__profile.json` are read-only data profiles from
`8-REES46/profile_rees46.py` and `10-Taobao/profile_taobao.py`. The shared code lives in `0-Shared/`: paper 1's protocol in
`paper1_protocol.py`, paper 1's cleaning rules in `clean_events.py`, and the bot rule in
`burst_check.py`.

## Inputs

| Dataset | Where | How to get it |
|---|---|---|
| Retail Rocket (paper 1) | `1-Data/events_clean.csv`, `1-Data/completed-session/cart_session_features.csv`, `1-Data/decision-time/prefix_dataset.parquet`; `5-Results/split_robustness_summary.csv` and `decision-time__*` (committed) | Raw `events.csv` from Kaggle "Retailrocket recommender system dataset" into `1-Data/raw/1-events.csv`; then `2-Completed-Session/stage1_data_cleaning_feature_engineering.py` (`events_clean.csv`), `2-Completed-Session/build_cart_session_table.py` (session table), `3-Decision-Time/build_prefix_dataset.py` (prefix table). See paper 1's `docs/results_provenance.md` |
| REES46 cosmetics shop | `1-Data/rees46/raw/2019-Oct.csv`, `2019-Nov.csv` | Kaggle "eCommerce Events History in Cosmetics Shop" |
| Taobao User Behavior | `1-Data/taobao/raw/UserBehavior.csv` (3,672,347,465 bytes, no header) | Alibaba Tianchi dataset 649 (https://tianchi.aliyun.com/dataset/649); the copy used is the Kaggle mirror "User Behavior Data from Taobao for Recommendation" (`marwa80/userbehavior`), whose file size matches |
| Flower shop | `1-Data/own/raw/*.json` | **Private.** Not distributed; see `data_statement.md` |

All of `1-Data/` is git-ignored. Only aggregate results are committed.

## Environment

Python 3.13.7, pandas 2.3.2, numpy 2.3.3, scikit-learn 1.8.0, xgboost 3.3.0, scipy 1.16.2,
pyarrow 25.0.0 (Windows 10).

## Resources

The REES46 steps load one month (about 4–5 million rows) at a time. About 5 GB of free memory is
enough; on a 12 GB laptop, close other heavy programs first. The Taobao file (3.7 GB) is read
in chunks.

**Verified rerun (2026-10-21):** all 24 steps ran in 55 minutes on the 12 GB laptop. The
slowest steps are REES46 cleaning (5 min for October, 4 for November) and Taobao groups A and
B (6 and 7 min).
All 37 files the run writes were rewritten, and `git diff 5-Results/` was empty: every
committed result was reproduced byte for byte. (An earlier run of the first 19 steps took
39 minutes.)
