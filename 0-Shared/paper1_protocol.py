"""
paper1_protocol.py
==================

Paper 1's evaluation protocol in one place, so every new dataset is run
through the same code (7-Own-Data/replicate_paper1.py, 8-REES46/...).
Each piece is copied from the paper-1 script that produced the published
number:

  completed_features  2-Completed-Session/build_cart_session_table.py
                      (build_session_features), purchase event as a parameter
  prefix_features     3-Decision-Time/build_prefix_dataset.py STEP 5, k=0
  split               2-Completed-Session/reanalysis_split_robustness.py
                      (completed table) and 3-Decision-Time/
                      train_prefix_models.py (first add-to-cart table)
  prepare, models     reanalysis_controlled_audit.py / train_prefix_models.py
  bootstrap_ci        train_prefix_models.py (1,000 resamples, 95%)
  run_plan            LR + XGB at 5 seeds; DT + RF at seed 42 where paper 1
                      fitted them; CI at seed 42 for the first add-to-cart
  retail_rocket_reference  paper 1's committed result files

Both tables need columns: visitorid, start_ts, y, and the feature columns.
"""

import glob
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEEDS = [11, 22, 33, 42, 55]
TABLE_SEED = 42
N_BOOTSTRAP = 1000
CI_ALPHA = 0.05

COMPLETED_CONFIGS = {
    "target_proxy": ["event_count", "view_count", "addtocart_count"],
    "original_completed": ["view_count", "addtocart_count", "view_to_cart_ratio",
                           "unique_items_count", "session_age_seconds",
                           "inter_event_interval_mean", "recency_last_cart_seconds"],
    "timing_only": ["session_age_seconds", "inter_event_interval_mean",
                    "recency_last_cart_seconds"],
    "counts_only": ["view_count", "addtocart_count", "view_to_cart_ratio",
                    "unique_items_count"],
}
PREFIX13 = ["view_count", "addtocart_count", "unique_items_count", "view_to_cart_ratio",
            "seconds_since_first_event", "seconds_since_first_cart", "seconds_since_last_cart",
            "seconds_since_previous_event", "inter_event_interval_mean", "events_after_cart",
            "repeated_views_of_cart_items", "hour_of_day", "day_of_week"]


# ============================================================================
# FEATURES (reference, one session at a time)
# ============================================================================
def completed_features(g, purchase):
    """build_cart_session_table.build_session_features; `purchase` is the
    event name that plays Retail Rocket's `transaction`."""
    ts = g["timestamp"].values
    events = g["event"].values
    view_cnt = (events == "view").sum()
    cart_cnt = (events == "addtocart").sum()
    v2c = view_cnt / cart_cnt if cart_cnt > 0 else float(view_cnt)
    intervals = np.diff(ts) / 1000.0
    cart_ts = ts[events == "addtocart"]
    return {
        "event_count": len(g),
        "view_count": int(view_cnt),
        "addtocart_count": int(cart_cnt),
        "view_to_cart_ratio": round(v2c, 4),
        "unique_items_count": int(g["itemid"].dropna().nunique()),
        "session_age_seconds": round((ts[-1] - ts[0]) / 1000.0, 2) if len(ts) > 1 else 0.0,
        "inter_event_interval_mean": round(intervals.mean(), 3) if len(intervals) > 0 else 0.0,
        "recency_last_cart_seconds": round((ts[-1] - cart_ts[-1]) / 1000.0, 2) if len(cart_ts) > 0 else -1.0,
        "y": int((events == purchase).any()),
    }


def prefix_features(g, t0):
    """build_prefix_dataset.py STEP 5 at k=0 (cutoff = the first add)."""
    p = g[g["timestamp"] <= t0]
    ts, events, items = p["timestamp"].values, p["event"].values, p["itemid"].values
    view_count = int((events == "view").sum())
    addtocart_count = int((events == "addtocart").sum())
    cart_ts = ts[events == "addtocart"]
    cart_items = set(pd.Series(items[events == "addtocart"]).dropna())
    cutoff = pd.to_datetime(int(t0), unit="ms")
    return {
        "view_count": view_count,
        "addtocart_count": addtocart_count,
        "unique_items_count": int(pd.Series(items).dropna().nunique()),
        "view_to_cart_ratio": view_count / addtocart_count,
        "seconds_since_first_event": (t0 - ts[0]) / 1000.0,
        "seconds_since_first_cart": (t0 - t0) / 1000.0,
        "seconds_since_last_cart": (t0 - cart_ts[-1]) / 1000.0,
        "seconds_since_previous_event": (t0 - ts[-2]) / 1000.0 if len(ts) >= 2 else np.nan,
        "inter_event_interval_mean": float((np.diff(ts) / 1000.0).mean()) if len(ts) >= 2 else np.nan,
        "events_after_cart": int((ts > t0).sum()),
        "repeated_views_of_cart_items": int(pd.Series(items[events == "view"]).dropna().isin(cart_items).sum()),
        "hour_of_day": cutoff.hour,
        "day_of_week": cutoff.dayofweek,
    }


# ============================================================================
# PROTOCOL
# ============================================================================
def split(df, kind, seed, table):
    n = len(df)
    if kind == "stratified":
        return train_test_split(np.arange(n), test_size=0.2, stratify=df["y"], random_state=seed)
    if kind == "visitor_grouped":
        return next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    .split(df, groups=df["visitorid"]))
    if table == "completed":  # reanalysis_split_robustness.py
        order = np.argsort(df["start_ts"].values, kind="stable")
        cut = int(round(0.8 * n))
    else:  # train_prefix_models.py
        order = np.argsort(df["start_ts"].values, kind="mergesort")
        cut = int(n * 0.8)
    return order[:cut], order[cut:]


def prepare(df, cols, tr, te):
    X_tr, X_te = df[cols].iloc[tr].copy(), df[cols].iloc[te].copy()
    for col in cols:
        cap = X_tr[col].quantile(0.99)
        X_tr[col], X_te[col] = X_tr[col].clip(upper=cap), X_te[col].clip(upper=cap)
    return X_tr.fillna(0), X_te.fillna(0), df["y"].values[tr], df["y"].values[te]


def bootstrap_ci(y, p, seed):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, 100 * CI_ALPHA / 2), np.percentile(aucs, 100 * (1 - CI_ALPHA / 2))


def models(seed, y_tr, extra):
    out = {
        "LR": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        "XGB": XGBClassifier(n_estimators=100, tree_method="hist", eval_metric="logloss",
                             scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                             random_state=seed, n_jobs=-1),
    }
    if extra:
        out["DT"] = DecisionTreeClassifier(class_weight="balanced", max_depth=8, random_state=seed)
        out["RF"] = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                           random_state=seed, n_jobs=-1)
    return out


def fit_predict(name, m, X_tr, y_tr, X_te):
    if name == "LR":  # training-only standardisation
        sc = StandardScaler().fit(X_tr)
        m.fit(sc.transform(X_tr), y_tr)
        return m.predict_proba(sc.transform(X_te))[:, 1]
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def run_plan(completed, prefix):
    """Every model paper 1 fitted, on one dataset's two tables."""
    plan = [("completed", c, cols, s) for c, cols in COMPLETED_CONFIGS.items()
            for s in ["stratified", "visitor_grouped", "temporal"]]
    plan += [("first_add_k0", "prefix13", PREFIX13, s) for s in ["visitor_grouped", "temporal"]]
    runs = []
    for table, config, cols, kind in plan:
        df = completed if table == "completed" else prefix
        for seed in SEEDS:
            tr, te = split(df, kind, seed, table)
            X_tr, X_te, y_tr, y_te = prepare(df, cols, tr, te)
            extra = seed == TABLE_SEED and (table == "first_add_k0" or
                                            (config == "original_completed" and kind == "stratified"))
            for name, m in models(seed, y_tr, extra).items():
                prob = fit_predict(name, m, X_tr, y_tr, X_te)
                row = {"table": table, "config": config, "split": kind, "model": name, "seed": seed,
                       "roc_auc": roc_auc_score(y_te, prob), "pr_auc": average_precision_score(y_te, prob),
                       "n_train": len(tr), "n_test": len(te), "test_positive_rate": float(y_te.mean())}
                if table == "first_add_k0" and seed == TABLE_SEED and name in ("LR", "XGB"):
                    row["ci_lo"], row["ci_hi"] = bootstrap_ci(y_te, prob, seed)
                runs.append(row)
    return pd.DataFrame(runs)


def retail_rocket_reference(results_dir):
    """Paper 1's published numbers, read from its committed result files."""
    rr = pd.read_csv(results_dir / "split_robustness_summary.csv")
    rr["split"] = rr["split"].replace({"stratified-80-20": "stratified", "visitor-grouped": "visitor_grouped"})
    rr = rr.assign(table="completed")[["table", "config", "split", "model", "roc_auc_mean"]]
    extra_rr = []
    for model, name in (("DT", "decision-tree"), ("RF", "random-forest")):
        with open(results_dir / f"completed-session__{name}__stratified-80-20__original-completed__s42.json") as fh:
            j = json.load(fh)
        extra_rr.append({"table": "completed", "config": "original_completed", "split": "stratified",
                         "model": model, "roc_auc_mean": j["roc_auc"]})
    for model, name in (("LR", "logistic-regression"), ("XGB", "xgboost"),
                        ("DT", "decision-tree"), ("RF", "random-forest")):
        for kind, tag in (("visitor_grouped", "visitor-grouped"), ("temporal", "temporal")):
            vals = []
            for f in glob.glob(str(results_dir / f"decision-time__{name}__{tag}-all-eligible__k0__s*.json")):
                with open(f) as fh:
                    vals.append(json.load(fh)["roc_auc"])
            extra_rr.append({"table": "first_add_k0", "config": "prefix13", "split": kind,
                             "model": model, "roc_auc_mean": float(np.mean(vals))})
    return pd.concat([rr, pd.DataFrame(extra_rr)]).rename(columns={"roc_auc_mean": "retail_rocket_roc_auc"})


def summarize(runs, rr):
    summary = (runs.groupby(["table", "config", "split", "model"], sort=False)
               .agg(n_seeds=("seed", "size"), roc_auc=("roc_auc", "mean"), roc_sd=("roc_auc", "std"),
                    pr_auc=("pr_auc", "mean"), ci_lo=("ci_lo", "first"), ci_hi=("ci_hi", "first"),
                    n=("n_train", lambda s: int(s.iloc[0]) + int(runs.loc[s.index[0], "n_test"])),
                    positive_rate=("test_positive_rate", "mean"))
               .reset_index().merge(rr, on=["table", "config", "split", "model"], how="left"))
    summary["roc_sd"] = summary["roc_sd"].fillna(0.0)
    return summary
