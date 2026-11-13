# Reproducing Paper 2

One command, after the inputs below are in place:

```
python 9-Paper2/run_all.py          # 30 steps; --list shows them, --from N resumes
```

Every step is the script that produced a committed file in `5-Results/`. A rerun should
reproduce those files exactly. The 30 steps write 45 files (`own-data__*`, `rees46__*`,
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

**Verified rerun (2026-10-27):** all 29 steps ran in 79 minutes on the 12 GB laptop. The
slowest steps are the confidence intervals for Taobao and REES46 November (12 and 8 min,
five seeds with session and visitor bootstraps), the main confidence intervals (8 min), the
scoring rules (11 min, mostly the lenient-Q8 probes) and Taobao groups A and B (4 and 5 min).
All 44 files the run writes were rewritten, and `git status 5-Results/` was clean: every
committed result was reproduced byte for byte. (Earlier verified run, 2026-10-21: 24 steps,
55 minutes, 37 files, also byte for byte.)

**Added 2026-11-09 (not yet part of a full rerun):** step 30, `9-Paper2/burst_day_profile.py`
(about 12 s), writes `paper2__burst-days-profile.csv`, the day-level profile of REES46 October
behind the burst paragraph. `leak_size_mechanism.py` gained one column,
`auto_cart_page_median_lag_s` (the flower shop's 0.6 s); rerun alone, all its older columns were
unchanged. Both scripts are deterministic (no models, no sampling).

**Added 2026-11-12 (compact manuscript):** two values in the text follow from committed files by
taking a minimum and a maximum; no script prints them.

- Logistic regression, original completed set minus counts only: +0.08 to +0.23 on every dataset
  and split (exactly +0.082 to +0.225). Source: the LR rows (`original_completed`, `counts_only`)
  of the four `*__paper1-replication__*summary.csv` files used in the paper (flower shop, REES46
  from 9 October, REES46 November, Taobao) and of `split_robustness_summary.csv` (Retail Rocket).
- Flower shop, original completed set, single XGBoost runs: 0.737–0.825 over 15 runs (three
  splits, five seeds). Source: `own-data__paper1-replication__runs.csv`.

The ranges in the manuscript's Table 4 (Group A) and Table 9 (negative control) are ranges of
means over five seeds, as their captions say: the means of the splits (and, in the last row, of
the four models) in Table 4, and of the configurations in Table 9. Every printed cell was
re-derived from the summary files on 2026-11-12.
