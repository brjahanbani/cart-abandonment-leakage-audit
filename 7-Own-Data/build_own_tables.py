"""
build_own_tables.py
===================

Second step of the flower-shop pipeline (run after build_own_events.py).
Cleans the reference tables (orders, products, categories), links every
`checkout_final` event to the order it completed, and builds one row per
cart session with its label and product type. No model features yet.

INPUTS (1-Data/own/raw/, phpMyAdmin JSON)
  b2c_order          one row per website order, 2024-06 -> export date
  product            product snapshot AS OF THE EXPORT (not historical)
  product_category   product <-> category, many-to-many
  category           category tree; `dateShifter` = minimum lead time in days
  + 1-Data/own/clean/events_clean.parquet from build_own_events.py

  product_order      order lines; product id inside `productCopy`, a JSON
                     copy of the product AS OF PURCHASE TIME

PERSONAL DATA
  b2c_order.driverName / driverNumber are dropped on load and never written.
  transactionId (payment-gateway reference) is dropped too: not needed.
  product_order.customerDescription (free-text gift note) is dropped unread.

DECISIONS (each one checked in docs/own_dataset.md)
  - Paid order  = statusId 8. 4,939 of 4,943 status-8 orders have paidPrice
    > 0; status 2 (1,468 orders, 1,458 unpaid) is an order created at the
    payment step but never paid. Code meanings are inferred from the data.
  - Daily product = in category 223 or its child 224 (dateShifter 0, shop
    owner's definition). No product is in both groups.
  - A `final` event belongs to the same customer's order created closest to
    it in time, within 1 hour (the order row is written at the payment step,
    median 63 s before `final`).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "1-Data" / "own" / "raw"
CLEAN = ROOT / "1-Data" / "own" / "clean"
STATS_PATH = ROOT / "5-Results" / "own-data__tables.json"

DAILY_CATEGORIES = {"223", "224"}
PAID_STATUS = "8"
LINK_WINDOW_S = 3600
PERSONAL_COLUMNS = ["driverName", "driverNumber", "transactionId"]


def load_phpmyadmin(name):
    with open(RAW / f"{name}.json", encoding="utf-8") as fh:
        blocks = json.load(fh)
    tables = [b for b in blocks if b.get("type") == "table"]
    assert len(tables) == 1, f"{name}: expected exactly one table block"
    return pd.DataFrame(tables[0]["data"])


stats = {}

# ============================================================================
# STEP 1 — ORDERS
# ============================================================================
orders = load_phpmyadmin("b2c_order").drop(columns=PERSONAL_COLUMNS)
assert orders["id"].is_unique
assert orders["deletedAt"].isna().all() and (orders["orderInPerson"] == "0").all()
orders = orders.drop(columns=["deletedAt", "branchId", "orderInPerson"])  # empty / constant
for col in ["cartItemsTotalPrice", "deliveryPrice", "discountPrice",
            "paidPrice", "finalPrice", "remainedPrice"]:
    orders[col] = pd.to_numeric(orders[col])
orders["created_ms"] = pd.to_datetime(orders["createdAt"]).astype("int64") // 1_000_000
orders["paid"] = orders["statusId"] == PAID_STATUS
stats["orders_total"] = len(orders)
stats["orders_window"] = [orders["createdAt"].min(), orders["createdAt"].max()]
stats["order_status_counts"] = orders["statusId"].value_counts().to_dict()
stats["status8_with_paidPrice_gt0"] = int((orders["paid"] & (orders["paidPrice"] > 0)).sum())
stats["order_type_counts"] = orders["orderType"].value_counts().to_dict()

# Order lines. Only the product id and the purchase-time `ready` flag are
# taken out of productCopy; the rest of the copy is product catalogue data.
lines = load_phpmyadmin("product_order").drop(columns=["customerDescription"])
copy = lines["productCopy"].map(json.loads)
lines = pd.DataFrame({
    "line_id": lines["id"],
    "order_id": lines["b2cOrderId"],
    "product_id": copy.map(lambda d: str(d["id"])),
    "ready_at_purchase": copy.map(lambda d: d.get("ready")),
    "quantity": pd.to_numeric(lines["count"]),
    "item_price": pd.to_numeric(lines["priceItemprice"]),
    "line_type": lines["orderType"],
    "createdAt": lines["createdAt"],
})
assert lines["line_id"].is_unique
assert set(lines["order_id"]) == set(orders["id"]), "order lines and orders do not cover each other"
stats["order_lines"] = len(lines)
stats["order_lines_per_order"] = lines.groupby("order_id").size().value_counts().sort_index().to_dict()
stats["order_lines_quantity_0"] = int((lines["quantity"] == 0).sum())

# ============================================================================
# STEP 2 — PRODUCTS + CATEGORIES
# ============================================================================
category = load_phpmyadmin("category")
lead_days = category.set_index("id")["dateShifter"].astype(int)
pc = load_phpmyadmin("product_category")
assert set(pc["categoryId"]) <= set(category["id"]), "mapping uses unknown category"
pc["daily"] = pc["categoryId"].isin(DAILY_CATEGORIES)
pc["lead_days"] = pc["categoryId"].map(lead_days)
per_product = pc.groupby("productId").agg(
    is_daily=("daily", "any"), any_scheduled=("daily", lambda x: (~x).any()),
    min_lead_days=("lead_days", "min"), n_categories=("categoryId", "size"))
assert not (per_product["is_daily"] & per_product["any_scheduled"]).any(), \
    "a product is both daily and scheduled"

product = load_phpmyadmin("product")
image_cols = [c for c in product.columns if c.startswith("image")]
product = product.drop(columns=image_cols)[
    ["id", "name", "createdAt", "updatedAt", "activeB2c", "activeB2b"]]
product = product.merge(per_product.drop(columns="any_scheduled"),
                        left_on="id", right_index=True, how="left")
assert set(lines["product_id"]) <= set(product["id"]), "ordered product missing from product table"
daily_flag = product.set_index("id")["is_daily"]
stats["line_type_vs_product_daily"] = pd.crosstab(
    lines["line_type"], lines["product_id"].map(daily_flag).fillna("unmapped").astype(str)
).to_dict()
stats["products"] = len(product)
stats["products_daily"] = int(product["is_daily"].eq(True).sum())
stats["products_updated_after_tracking_start"] = int(
    (pd.to_datetime(product["updatedAt"]) >= "2026-08-09").sum())

# ============================================================================
# STEP 3 — LINK `final` EVENTS TO ORDERS
# ============================================================================
ev = pd.read_parquet(CLEAN / "events_clean.parquet")
assert set(ev.loc[ev["event"].isin(["view", "addtocart"]), "itemid"]) <= set(product["id"]), \
    "event references a product missing from the product table"
ev["customer"] = ev["person_key"].where(ev["person_key"].str.startswith("c_")).str[2:]

finals = ev[ev["event"] == "checkout_final"][["session_key", "timestamp", "customer"]]
cand = finals.reset_index().merge(
    orders[["id", "customerId", "created_ms", "statusId", "paid", "orderType"]],
    left_on="customer", right_on="customerId")
cand["delta_s"] = (cand["timestamp"] - cand["created_ms"]) / 1000
cand = cand[cand["delta_s"].abs() <= LINK_WINDOW_S]
link = cand.loc[cand.groupby("index")["delta_s"].apply(lambda s: s.abs().idxmin())]
link = link.rename(columns={"id": "order_id"}).drop(columns=["index", "customerId"])

tracking_start = ev["timestamp"].min()
in_window = orders[orders["created_ms"] >= tracking_start]
stats["final_events"] = len(finals)
stats["final_events_without_customer"] = int(finals["customer"].isna().sum())
stats["final_events_linked"] = len(link)
stats["distinct_orders_linked"] = int(link["order_id"].nunique())
stats["linked_order_status"] = link.drop_duplicates("order_id")["statusId"].value_counts().to_dict()
stats["final_minus_order_created_s_median"] = float(link["delta_s"].median())
per_sess = link.groupby("session_key")["order_id"].agg(["size", "nunique"])
stats["sessions_repeat_final_same_order"] = int(((per_sess["size"] > 1) & (per_sess["nunique"] == 1)).sum())
stats["sessions_with_several_orders"] = int((per_sess["nunique"] > 1).sum())
stats["orders_since_tracking_start"] = len(in_window)
stats["paid_orders_since_tracking_start"] = int(in_window["paid"].sum())
stats["paid_orders_with_final"] = int(in_window.loc[in_window["paid"], "id"].isin(link["order_id"]).sum())
stats["unpaid_status2_orders_with_final"] = int(
    in_window.loc[in_window["statusId"] == "2", "id"].isin(link["order_id"]).sum())

# ============================================================================
# STEP 4 — ONE ROW PER CART SESSION (label + product type, no features)
# ============================================================================
adds = ev[ev["event"] == "addtocart"].merge(
    product[["id", "is_daily", "min_lead_days", "activeB2c"]], left_on="itemid", right_on="id")
first_add = adds.sort_values("timestamp", kind="stable").groupby("session_key").first()
cart = pd.DataFrame({
    "visitorid": first_add["visitorid"],
    "person_key": first_add["person_key"],
    "first_add_ts": first_add["timestamp"],
    "first_item": first_add["itemid"],
    "first_item_daily": first_add["is_daily"],
    "first_item_lead_days": first_add["min_lead_days"],
    "n_adds": adds.groupby("session_key").size(),
    "share_daily_adds": adds.groupby("session_key")["is_daily"].mean(),
})
final_after = finals.merge(cart["first_add_ts"], left_on="session_key", right_index=True)
final_after = final_after[final_after["timestamp"] > final_after["first_add_ts"]]
cart["y_final"] = cart.index.isin(final_after["session_key"])
paid_link = link[link["paid"]].merge(cart["first_add_ts"], left_on="session_key", right_index=True)
paid_link = paid_link[paid_link["timestamp"] > paid_link["first_add_ts"]]
cart["y_paid_order"] = cart.index.isin(paid_link["session_key"])

stats["cart_sessions"] = len(cart)
stats["y_final_rate"] = round(float(cart["y_final"].mean()), 4)
stats["y_paid_order_rate"] = round(float(cart["y_paid_order"].mean()), 4)
stats["label_agreement"] = round(float((cart["y_final"] == cart["y_paid_order"]).mean()), 4)
stats["cart_sessions_first_item_daily"] = round(float(cart["first_item_daily"].mean()), 4)
stats["y_final_rate_by_first_item"] = {
    ("daily" if k else "scheduled"): round(float(v), 4)
    for k, v in cart.groupby("first_item_daily")["y_final"].mean().items()}

# Item-level label check (paper 1, Sec. IV-C): does a "purchased" session
# also contain a carted item that was never bought? Purchased items = lines
# of the session's linked paid orders. An item counts as abandoned only if it
# was carted, not removed afterwards in the session, and not bought.
bought = (paid_link[["session_key", "order_id"]].drop_duplicates()
          .merge(lines[["order_id", "product_id"]], on="order_id")
          .groupby("session_key")["product_id"].agg(set))
carted = adds.groupby("session_key")["itemid"].agg(set)
removed = ev[ev["event"] == "cart_remove"].groupby("session_key")["itemid"].agg(set)
paid_sessions = cart.index[cart["y_paid_order"]]
mixed = mixed_kept = bought_not_carted = 0
for sk in paid_sessions:
    b, c_ = bought.get(sk, set()), carted[sk]
    mixed += bool(c_ - b)
    mixed_kept += bool(c_ - b - removed.get(sk, set()))
    bought_not_carted += bool(b - c_)
stats["item_check_paid_sessions"] = len(paid_sessions)
stats["item_check_carted_item_not_bought"] = mixed
stats["item_check_carted_not_removed_not_bought"] = mixed_kept
stats["item_check_bought_item_not_carted_in_session"] = bought_not_carted
cart["mixed_purchase_abandon"] = cart.index.isin(
    [sk for sk in paid_sessions if carted[sk] - bought.get(sk, set()) - removed.get(sk, set())])

# Leakage candidate (latest-snapshot join): the product table is the state at
# export time, AFTER every session. Record purchase rate by today's status.
snap = cart.join(product.set_index("id")["activeB2c"], on="first_item")
stats["y_final_rate_by_first_item_active_now"] = {
    f"daily={d}, activeB2c_now={a}": [round(float(g["y_final"].mean()), 4), len(g)]
    for (d, a), g in snap.groupby(["first_item_daily", "activeB2c"])}

# ============================================================================
# STEP 5 — WRITE
# ============================================================================
orders.to_parquet(CLEAN / "orders_clean.parquet", index=False)
lines.to_parquet(CLEAN / "order_lines_clean.parquet", index=False)
product.to_parquet(CLEAN / "products_clean.parquet", index=False)
link.to_parquet(CLEAN / "final_order_link.parquet", index=False)
cart.reset_index().to_parquet(CLEAN / "cart_sessions_labels.parquet", index=False)
STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
