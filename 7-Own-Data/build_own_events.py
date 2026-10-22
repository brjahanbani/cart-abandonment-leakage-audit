"""
build_own_events.py
===================

Cleaning + sessionization for the self-collected flower-shop dataset
(phpMyAdmin JSON exports of `analytics_event` and `cart_event`), mirroring
the Retail Rocket decision-time lineage (0-Shared/clean_events.py) wherever
the two datasets allow it, and documenting every place they differ.

WHAT IS DIFFERENT FROM RETAIL ROCKET
------------------------------------
- Two source tables. `analytics_event` has views, category views, searches,
  cart removals and checkout steps but NO add-to-cart; add-to-carts live only
  in `cart_event` (with price and quantity). They are merged into one event
  log on `anonymousId` (the browser-level id, the equivalent of Retail
  Rocket's `visitorid`).
- No transaction event. The checkout step `final` is logged when payment
  succeeded and the order is complete (confirmed by the shop owner,
  2026-10-19), so it plays the role of Retail Rocket's `transaction`.
  Repeated `final` events in one session (page refresh vs a second order)
  are resolved against the orders table in build_own_tables.py; see
  docs/own_dataset.md.
- Category 223 and its subcategories are same-day ("daily") products; all
  other categories are scheduled delivery. Product events carry no category
  id, so this split needs the product -> category mapping.
- A user-agent column exists, so crawlers are removed by user agent first;
  the Retail Rocket velocity / CV bot rules are then applied unchanged.
- The front end double-fires some events (same visitor, event, item and
  payload within 2 s). Those repeats are dropped; Retail Rocket had no such
  rule because it had no such duplicates.
- `cart_event` starts on 2026-07-14, `analytics_event` on 2026-08-09. Cart
  rows before the first analytics event have no surrounding behaviour and
  are dropped.

Nothing here builds model features. The output is a clean, sessionized event
log; feature tables (completed-session and point-in-time) are built from it
in later scripts.

Data stays local: 1-Data/own/ is git-ignored. The only committed output is
an aggregate-counts JSON in 5-Results/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
from clean_events import remove_bots, sessionize  # noqa: E402  (same rules as Retail Rocket)

RAW_DIR = ROOT / "1-Data" / "own" / "raw"
OUT_DIR = ROOT / "1-Data" / "own" / "clean"
PROFILE_PATH = ROOT / "5-Results" / "own-data__profile.json"

CRAWLER_UA = r"bot|crawl|spider|slurp|headless|python|curl|wget|lighthouse"
DOUBLE_FIRE_SECONDS = 2


def load_phpmyadmin(path):
    """phpMyAdmin JSON export: a list of header / database / table blocks."""
    blocks = json.load(open(path, encoding="utf-8"))
    tables = [b for b in blocks if b.get("type") == "table"]
    assert len(tables) == 1, f"{path.name}: expected exactly one table block"
    return pd.DataFrame(tables[0]["data"])


def to_ms(series):
    return (pd.to_datetime(series).astype("int64") // 1_000_000).astype("int64")


stats = {}

# ============================================================================
# STEP 1 — LOAD + TYPE
# ============================================================================
a = load_phpmyadmin(RAW_DIR / "analytics_event.json")
c = load_phpmyadmin(RAW_DIR / "cart_event.json")
assert a["id"].is_unique and c["id"].is_unique, "duplicate primary keys in export"
assert a["anonymousId"].notna().all() and c["anonymousId"].notna().all()
stats["raw_rows"] = {"analytics_event": len(a), "cart_event": len(c)}

a["timestamp"] = to_ms(a["createdAt"])
c["timestamp"] = to_ms(c["createdAt"])
payload = a["payload"].map(lambda p: json.loads(p) if isinstance(p, str) else {})

# ============================================================================
# STEP 2 — ONE EVENT VOCABULARY
# ============================================================================
step = payload.map(lambda p: p.get("step"))
event = a["eventType"].map({"view": "view", "viewCategory": "view_category",
                            "search": "search", "cart_remove": "cart_remove"})
event = event.where(a["eventType"] != "checkout_step", "checkout_" + step.astype(str))
assert event.notna().all(), f"unmapped eventType: {set(a.loc[event.isna(), 'eventType'])}"

ev_a = pd.DataFrame({
    "timestamp": a["timestamp"],
    "visitorid": a["anonymousId"],
    "customer_id": a["customerId"],
    "event": event,
    "itemid": a["productId"],
    "category_id": a["categoryId"],
    "channel": a["channel"],
    "user_agent": a["userAgent"],
    "search_keywords": payload.map(lambda p: p.get("keywords")),
    "search_result_count": payload.map(lambda p: p.get("resultCount")),
    "quantity": payload.map(lambda p: p.get("count")),
    "price": np.nan,
    "source": "analytics_event",
})

first_analytics_ts = int(a["timestamp"].min())
before = c["timestamp"] < first_analytics_ts
stats["cart_rows_before_analytics_start_dropped"] = int(before.sum())
c = c[~before]
ev_c = pd.DataFrame({
    "timestamp": c["timestamp"],
    "visitorid": c["anonymousId"],
    "customer_id": c["customerId"],
    "event": "addtocart",
    "itemid": c["productId"],
    "category_id": None,
    "channel": c["channel"],
    "user_agent": None,
    "search_keywords": None,
    "search_result_count": np.nan,
    "quantity": pd.to_numeric(c["count"]),
    "price": pd.to_numeric(c["price"]),
    "source": "cart_event",
})
ev = pd.concat([ev_a, ev_c], ignore_index=True)

# Sanity: product events must carry a product, prices must be positive.
needs_item = ev["event"].isin(["view", "addtocart", "cart_remove"])
assert ev.loc[needs_item, "itemid"].notna().all(), "product event without productId"
assert (ev.loc[ev["event"] == "addtocart", "price"] > 0).all(), "non-positive price"
stats["bulk_quantity_adds_kept"] = int(((ev["event"] == "addtocart") & (ev["quantity"] > 2)).sum())

# ============================================================================
# STEP 3 — CRAWLERS BY USER AGENT (new: Retail Rocket had no user agent)
# ============================================================================
crawler = ev["user_agent"].fillna("").str.contains(CRAWLER_UA, case=False)
crawler_visitors = set(ev.loc[crawler, "visitorid"])
is_crawler = ev["visitorid"].isin(crawler_visitors)
stats["crawler_visitors"] = len(crawler_visitors)
stats["crawler_events"] = int(is_crawler.sum())
stats["crawler_events_that_were_adds_or_checkout"] = int(
    (is_crawler & ev["event"].str.match("addtocart|checkout_")).sum())
ev = ev[~is_crawler]

# ============================================================================
# STEP 4 — FRONT-END DOUBLE FIRES (new)
# ============================================================================
ev = ev.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)
key = ["visitorid", "event", "itemid", "category_id", "search_keywords", "quantity"]
gap = ev.groupby(key, dropna=False)["timestamp"].diff()
double = gap.notna() & (gap <= DOUBLE_FIRE_SECONDS * 1000)
stats["double_fire_events_dropped"] = ev.loc[double, "event"].value_counts().to_dict()
ev = ev[~double].reset_index(drop=True)

# ============================================================================
# STEP 5 — RETAIL ROCKET BOT RULES (unchanged thresholds)
# ============================================================================
mix_before = ev["event"].value_counts()
ev, bot_stats = remove_bots(ev)
stats["rr_rule_bot_visitors"] = bot_stats["n_bot_visitors"]
stats["rr_rule_bot_events"] = bot_stats["n_bot_events"]
removed = mix_before.sub(ev["event"].value_counts(), fill_value=0)
stats["rr_rule_bot_events_by_type"] = removed[removed > 0].astype(int).to_dict()

# ============================================================================
# STEP 6 — SESSIONS (same 30-minute gap) + person key for grouped splits
# ============================================================================
ev = sessionize(ev)
# One person can use several browsers (anonymousIds) once logged in. For
# visitor-grouped splits, group by the customer id when the browser ever
# logged in (its most frequent customer id), else by the browser id.
cust = (ev.dropna(subset=["customer_id"]).groupby("visitorid")["customer_id"]
        .agg(lambda s: s.value_counts().index[0]))
ev["person_key"] = ev["visitorid"].map(cust).radd("c_").fillna("a_" + ev["visitorid"])
stats["browsers_with_several_customer_ids"] = int(
    (ev.dropna(subset=["customer_id"]).groupby("visitorid")["customer_id"].nunique() > 1).sum())

# ============================================================================
# STEP 7 — PROFILE (aggregate counts only)
# ============================================================================
g = ev.groupby("session_key")
s = pd.DataFrame({
    "n": g.size(),
    "adds": g["event"].apply(lambda x: (x == "addtocart").sum()),
    "final": g["event"].apply(lambda x: (x == "checkout_final").any()),
    "start": g["timestamp"].min(),
})
first_add = ev[ev["event"] == "addtocart"].groupby("session_key")["timestamp"].min()
first_final = ev[ev["event"] == "checkout_final"].groupby("session_key")["timestamp"].min()
cart = s[s["adds"] > 0].copy()
cart["final_before_first_add"] = first_final.reindex(cart.index) < first_add.reindex(cart.index)
weeks = pd.to_datetime(cart["start"], unit="ms").dt.to_period("W").astype(str)

stats.update({
    "clean_events": len(ev),
    "clean_event_mix": ev["event"].value_counts().to_dict(),
    "window": [str(pd.to_datetime(ev["timestamp"].min(), unit="ms")),
               str(pd.to_datetime(ev["timestamp"].max(), unit="ms"))],
    "visitors": int(ev["visitorid"].nunique()),
    "persons": int(ev["person_key"].nunique()),
    "items": int(ev["itemid"].nunique()),
    "sessions": len(s),
    "single_event_session_share": round(float((s["n"] == 1).mean()), 4),
    "cart_sessions": len(cart),
    "cart_sessions_reaching_final": int(cart["final"].sum()),
    "cart_session_final_rate": round(float(cart["final"].mean()), 4),
    "cart_sessions_final_before_first_add": int(cart["final_before_first_add"].sum()),
    "final_sessions_without_add_in_session": int((s["final"] & (s["adds"] == 0)).sum()),
    "cart_sessions_per_week": weeks.value_counts().sort_index().to_dict(),
})

OUT_DIR.mkdir(parents=True, exist_ok=True)
ev.to_parquet(OUT_DIR / "events_clean.parquet", index=False)
PROFILE_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(stats, indent=2, ensure_ascii=False))
