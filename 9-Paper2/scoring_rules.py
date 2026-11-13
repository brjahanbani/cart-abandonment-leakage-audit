"""
scoring_rules.py
================

Two scoring rules for the detection matrix, written after both pre-submission reviews
asked for them (2026-10-27) and applied to results that already exist:

  Q4' "approximately"  share of cart sessions in which event count - views - add-to-carts
                       equals the number of purchase events; the identity is exact when the
                       share is 100%, approximate when it holds in most sessions (>= 50%)
  Q8 "implausible"     lenient reading: Q8 partly flags a leaky configuration whose score
                       exceeds the best correctly built configuration of the same dataset
                       (negative control, Table 6) by more than a margin (0.10; 0.05 shown
                       as a sensitivity check). Scores: XGBoost, visitor-grouped split, mean
                       of five seeds, from the existing result files.

Outputs: 5-Results/paper2__q4-identity-share.csv, 5-Results/paper2__q8-lenient.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D, RES = ROOT / "1-Data", ROOT / "5-Results"


# --- Q4': how often does the count identity hold exactly? ---------------------------
def identity_share(name, ev, purchase):
    ev = ev.assign(event=ev["event"].astype(str))
    g = ev.groupby("session_key")["event"]
    n = pd.DataFrame({"events": g.size(),
                      "views": g.agg(lambda s: (s == "view").sum()),
                      "carts": g.agg(lambda s: (s == "addtocart").sum()),
                      "buys": g.agg(lambda s: (s == purchase).sum())})
    n = n[n["carts"] > 0]
    holds = (n["events"] - n["views"] - n["carts"]) == n["buys"]
    return {"dataset": name, "cart_sessions": len(n), "identity_holds": round(holds.mean(), 4)}


cols = ["timestamp", "event", "session_key"]
rows = [identity_share("retail_rocket", pd.read_csv(D / "events_clean.csv", usecols=cols), "transaction")]
own = pd.read_parquet(D / "own" / "clean" / "events_clean.parquet", columns=cols + ["user_agent"])
own_from = own.loc[own["user_agent"].notna(), "timestamp"].min()
first_add = own[own["event"] == "addtocart"].groupby("session_key")["timestamp"].min()
own = own[own["session_key"].isin(first_add.index[first_add >= own_from])]
rows.append(identity_share("flower_shop", own[cols], "checkout_final"))
ree = pd.read_parquet(D / "rees46" / "clean" / "events_cart_sessions.parquet", columns=cols)
start = ree.groupby("session_key")["timestamp"].min()
ree = ree[ree["session_key"].isin(start.index[start >= pd.Timestamp("2019-10-09").value // 10**6])]
rows.append(identity_share("rees46_oct", ree, "transaction"))
del ree
rows.append(identity_share("rees46_nov", pd.read_parquet(
    D / "rees46" / "clean" / "2019-Nov" / "events_cart_sessions.parquet", columns=cols), "transaction"))
rows.append(identity_share("taobao", pd.read_parquet(
    D / "taobao" / "clean" / "events_cart_sessions.parquet", columns=cols), "transaction"))
q4 = pd.DataFrame(rows)
q4.to_csv(RES / "paper2__q4-identity-share.csv", index=False)
print(q4.to_string(index=False))


# --- Q8: lenient reading ---------------------------------------------------------------
def runs_mean(f, table, config):
    d = pd.read_csv(RES / f)
    s = d[(d["table"] == table) & (d["config"] == config) & (d["model"] == "XGB")
          & (d["split"] == "visitor_grouped")]
    return s["roc_auc"].mean()


def history(f, window, rr=False):
    d = pd.read_csv(RES / f)
    d = d[d["features"] == "base+agg"].set_index("window")
    return d.loc[window, "rr_visitor_auc" if rr else "visitor_auc"]


RUNS = {"flower_shop": "own-data__paper1-replication__runs.csv",
        "rees46_oct": "rees46__paper1-replication__from-2019-10-09__runs.csv",
        "rees46_nov": "rees46__paper1-replication__2019-Nov__all-month__runs.csv",
        "taobao": "taobao__paper1-replication__all__runs.csv"}
HIST = {"retail_rocket": ("own-data__history-window.csv", True),
        "flower_shop": ("own-data__history-window.csv", False),
        "rees46_oct": ("rees46__history-window.csv", False),
        "rees46_nov": ("rees46__history-window__2019-Nov__all-month.csv", False),
        "taobao": ("taobao__history-window__all.csv", False)}
rr_proxy = pd.read_csv(RES / "paper2__pr-auc.csv").query(
    "dataset == 'retail_rocket' and config == 'target_proxy'")["roc_auc"].item()

cells = []  # dataset, mechanism, strict verdict, score
for ds, (f, rr) in HIST.items():
    cells += [(ds, "L4", "missed", history(f, "full", rr)), (ds, "L5", "missed", history(f, "full_loo", rr))]
cells += [("retail_rocket", "L1", "flagged", rr_proxy)]
cells += [(ds, "L1", "partly", runs_mean(f, "completed", "target_proxy"))
          for ds, f in RUNS.items() if ds != "flower_shop"]
extras = pd.read_csv(RES / "own-data__extras-funnel-product.csv")
cells += [("flower_shop", "L6", "missed", extras.query(
    "config == 'counts_plus_funnel' and split == 'visitor_grouped' and model == 'XGB'")["roc_auc"].item())]
cells += [("rees46_oct", "L7", "missed",
           runs_mean("rees46__paper1-replication__all-october__runs.csv", "first_add_k0", "prefix13"))]

neg = {ds: pd.read_csv(RES / f"paper2__negative-control__{ds}.csv")["true_auc"].max()
       for ds in ["retail_rocket", "flower_shop", "rees46_oct", "rees46_nov", "taobao"]}
out = []
for ds, mech, strict, score in cells:
    row = {"dataset": ds, "mechanism": mech, "strict": strict, "score": round(score, 4),
           "best_correct": round(neg[ds], 4), "excess": round(score - neg[ds], 4)}
    for m in (0.10, 0.05):
        row[f"lenient_{m:.2f}"] = "partly" if strict == "missed" and score - neg[ds] > m else strict
    out.append(row)
q8 = pd.DataFrame(out).sort_values(["mechanism", "dataset"])
q8.to_csv(RES / "paper2__q8-lenient.csv", index=False)
print(q8.to_string(index=False))
# L2 (flagged on all five) and L3 (flagged; a label, no score) complete the 26 cells.
fixed = {"flagged": 10}
for col in ("strict", "lenient_0.10", "lenient_0.05"):
    t = q8[col].value_counts().to_dict()
    t["flagged"] = t.get("flagged", 0) + fixed["flagged"]
    print(f"{col:13s} flagged {t.get('flagged', 0)}, partly {t.get('partly', 0)}, missed {t.get('missed', 0)}")
