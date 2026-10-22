"""
group_b_rees46.py
=================

Group B on REES46 (clean window, sessions from 2019-10-09), the same two
tests as on the flower shop where the data allows (it has no checkout funnel):

  B3 history window   6-Point-In-Time/probes/agg_probe.py repeated exactly:
                      the first carted item's conversion rate
                      (purchases + 1) / (add-to-carts + 2) and the visitor's
                      purchase count, each counted
                        asof      strictly before the first add-to-cart
                        full_loo  the whole month minus this session
                        full      the whole month including this session
                      XGBoost 200 trees, depth 4, learning rate 0.1, no cap;
                      visitor-grouped and temporal splits; seeds 11..55.
                      Counts use every clean event of October (all sessions,
                      Oct 1-8 included: they are history, not scored rows).
                      REES46 purchase rows carry the product id, as Retail
                      Rocket's transactions did. Timestamps are whole seconds,
                      so "before" means an earlier second.

  B2 product facts    group A's protocol (0-Shared/paper1_protocol.py) on the
                      13 prefix features + what is known about the cart at
                      the first add-to-cart: the item's price, whether it has
                      a brand, distinct categories seen so far, removals so far.

Outputs (aggregate only):
  5-Results/rees46__history-window.csv
  5-Results/rees46__extras-product.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from paper1_protocol import PREFIX13, SEEDS, fit_predict, models, prepare, split  # noqa: E402

# Same switches as replicate_paper1_rees46.py (--month, --from, --exclude-automated-days);
# with none given, October's clean window.
if len(sys.argv) == 1:
    sys.argv += ["--from", "2019-10-09"]
import replicate_paper1_rees46 as a  # noqa: E402  (builds the tables; models run only under __main__)

CLEAN = a.CLEAN
RESULTS = ROOT / "5-Results"
SUFFIX = "" if (a.TAG == "from-2019-10-09" and not a.TAOBAO) else f"__{a.TAG}"
# Retail Rocket probe, XGBoost ROC-AUC (visitor / temporal), docs/point_in_time_log.md.
RETAIL_ROCKET = {
    ("none", "base"): (0.645, 0.664),
    ("asof", "agg_only"): (0.611, 0.633), ("asof", "base+agg"): (0.683, 0.711),
    ("full_loo", "agg_only"): (0.630, 0.658), ("full_loo", "base+agg"): (0.679, 0.726),
    ("full", "agg_only"): (0.971, 0.977), ("full", "base+agg"): (0.973, 0.977),
}

pref = a.prefix.copy()
pref["t0"] = a.t0.reindex(pref.index)
first_add = (a.ev[a.ev["event"] == "addtocart"].groupby("session_key").head(1)
             .set_index("session_key"))
pref["item"] = first_add["itemid"].reindex(pref.index)

# ============================================================================
# B3 · HISTORY WINDOW
# ============================================================================
allev = pd.read_parquet(CLEAN / "events_clean.parquet")
allev["event"] = allev["event"].astype(str)
adds = allev[allev["event"] == "addtocart"]
buys = allev[allev["event"] == "transaction"]
add_t = {i: np.sort(g.values) for i, g in adds.groupby("itemid")["timestamp"]}
buy_t = {i: np.sort(g.values) for i, g in buys.groupby("itemid")["timestamp"]}
vis_t = {v: np.sort(g.values) for v, g in buys.groupby("visitorid")["timestamp"]}
own_item_adds = adds.groupby(["session_key", "itemid"]).size()
own_item_buys = buys.groupby(["session_key", "itemid"]).size()
own_buys = buys.groupby("session_key").size()

EMPTY = np.array([], dtype="int64")
hist = {}
for sk, item, t0, vis in zip(pref.index, pref["item"], pref["t0"], pref["visitorid"]):
    at, bt, vt = add_t.get(item, EMPTY), buy_t.get(item, EMPTY), vis_t.get(vis, EMPTY)
    oa, ob = int(own_item_adds.get((sk, item), 0)), int(own_item_buys.get((sk, item), 0))
    hist[sk] = (
        (np.searchsorted(bt, t0, "left") + 1) / (np.searchsorted(at, t0, "left") + 2),
        (len(bt) - ob + 1) / (len(at) - oa + 2),
        (len(bt) + 1) / (len(at) + 2),
        np.searchsorted(vt, t0, "left"),
        len(vt) - int(own_buys.get(sk, 0)),
        len(vt),
    )
hist = pd.DataFrame.from_dict(hist, orient="index", columns=[
    "item_cr_asof", "item_cr_full_loo", "item_cr_full",
    "vis_txns_asof", "vis_txns_full_loo", "vis_txns_full"])
pref = pref.join(hist)
assert (pref["item_cr_full_loo"] > 0).all() and (pref["vis_txns_full_loo"] >= 0).all()


# ============================================================================
# B2 · PRODUCT FACTS AT THE FIRST ADD-TO-CART
# ============================================================================
# Taobao logs no price or brand; its facts known at the first add are the
# categories seen and the items favourited so far.
cols = ["session_key", "timestamp", "event", "category_id"] + ([] if a.TAOBAO else ["price", "brand"])
cs = pd.read_parquet(CLEAN / "events_cart_sessions.parquet", columns=cols)
cs = cs[cs["session_key"].isin(pref.index)].sort_values(["session_key", "timestamp"], kind="stable")
cs["event"] = cs["event"].astype(str)
p = cs[cs["timestamp"] <= cs["session_key"].map(pref["t0"])]
pref["categories_so_far"] = p.groupby("session_key")["category_id"].nunique()
if a.TAOBAO:
    pref["favourites_so_far"] = (p["event"] == "fav").groupby(p["session_key"]).sum()
    PRODUCT = ["categories_so_far", "favourites_so_far"]
else:
    fa = p[p["event"] == "addtocart"].groupby("session_key").head(1).set_index("session_key")
    pref["first_item_price"] = fa["price"].astype(float)
    pref["first_item_has_brand"] = fa["brand"].notna().astype(int)
    pref["removals_so_far"] = (p["event"] == "remove_from_cart").groupby(p["session_key"]).sum()
    PRODUCT = ["first_item_price", "first_item_has_brand", "categories_so_far", "removals_so_far"]


def probe_auc(feats, kind, seed):
    """agg_probe.py's evaluation, unchanged."""
    if kind == "temporal":
        tr = (pref["start_ts"].rank(method="first") <= 0.8 * len(pref)).values
    else:
        i_tr, _ = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                       .split(pref, groups=pref["visitorid"]))
        tr = np.zeros(len(pref), bool)
        tr[i_tr] = True
    te, y = ~tr, pref["y"].values
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                      scale_pos_weight=(y[tr] == 0).sum() / (y[tr] == 1).sum(),
                      random_state=seed, eval_metric="logloss", n_jobs=-1)
    m.fit(pref[feats].values[tr], y[tr])
    return roc_auc_score(y[te], m.predict_proba(pref[feats].values[te])[:, 1])


def main():
    """Fit B3 (probe protocol) and B2 (group A protocol) and write both result files."""
    plan = [("none", "base", PREFIX13)]
    for w in ["asof", "full_loo", "full"]:
        agg = [f"item_cr_{w}", f"vis_txns_{w}"]
        plan += [(w, "agg_only", agg), (w, "base+agg", PREFIX13 + agg)]
    rows = []
    for w, name, feats in plan:
        res = {k: [probe_auc(feats, k, s) for s in SEEDS] for k in ["visitor_grouped", "temporal"]}
        rr = RETAIL_ROCKET[(w, name)]
        rows.append({"window": w, "features": name,
                     "visitor_auc": np.mean(res["visitor_grouped"]), "visitor_sd": np.std(res["visitor_grouped"]),
                     "temporal_auc": np.mean(res["temporal"]), "temporal_sd": np.std(res["temporal"]),
                     "rr_visitor_auc": rr[0], "rr_temporal_auc": rr[1]})
    b3 = pd.DataFrame(rows)
    b3.round(4).to_csv(RESULTS / f"{a.PREFIX}__history-window{SUFFIX}.csv", index=False)
    print(f"B3 rows {len(pref)}; bought before first add (visitor): {(pref['vis_txns_asof'] > 0).sum()}, "
          f"buys at some point outside this session: {(pref['vis_txns_full_loo'] > 0).sum()}")
    print(b3.round(3).to_string(index=False))


    rows = []
    for config, cols in (("prefix13", PREFIX13), ("prefix13_plus_product", PREFIX13 + PRODUCT)):
        for kind in ["visitor_grouped", "temporal"]:
            for seed in SEEDS:
                tr, te = split(pref, kind, seed, "first_add_k0")
                X_tr, X_te, y_tr, y_te = prepare(pref, cols, tr, te)
                for name, m in models(seed, y_tr, extra=False).items():
                    rows.append({"config": config, "split": kind, "model": name, "seed": seed,
                                 "roc_auc": roc_auc_score(y_te, fit_predict(name, m, X_tr, y_tr, X_te))})
    b2 = (pd.DataFrame(rows).groupby(["config", "split", "model"], sort=False)["roc_auc"]
          .agg(roc_auc="mean", roc_sd="std").reset_index())
    b2.round(4).to_csv(RESULTS / f"{a.PREFIX}__extras-product{SUFFIX}.csv", index=False)
    print(b2.pivot_table(index="config", columns=["model", "split"], values="roc_auc", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
