# Flower-shop dataset: data documentation

Second, independent dataset for validating the leakage checklist (Paper 2). It's a live
production site: an online flower shop whose platform the author develops. The shop is kept
anonymous (decided 2026-10-21).
Last export: **2026-10-19**. Status: cleaned and labelled. First audit run: see `docs/own_dataset_audit.md`.

| | |
|---|---|
| Pipeline | `7-Own-Data/build_own_events.py` → `7-Own-Data/build_own_tables.py` |
| Raw input | `1-Data/own/raw/*.json` (phpMyAdmin exports of the shop's production database) |
| Clean output | `1-Data/own/clean/*.parquet` |
| Committed numbers | `5-Results/own-data__profile.json`, `5-Results/own-data__tables.json` (aggregate counts only) |
| Privacy | `1-Data/own/` is git-ignored. Driver name/phone, the gateway transaction id and customers' gift notes are dropped on load. No recipient names, phones or addresses are in the exports. Raw rows are never printed in logs or docs |

Rerun (≈3 min), from `7-Own-Data/`: `build_own_events.py` → `build_own_tables.py` → `replicate_paper1.py` → `extras_funnel_product.py` → `history_window_test.py`

---

## 1. Source tables

| Table | Rows | Covers | Role |
|---|---|---|---|
| `analytics_event` | 62,464 | 2026-08-09 → 10-19 | Clickstream: `view`, `viewCategory`, `search`, `cart_remove`, `checkout_step` (payload `step` = cart / address / payment / final) |
| `cart_event` | 1,908 | 2026-07-14 → 10-19 | Add-to-cart, with price and quantity. **The only source of add-to-carts** |
| `b2c_order` | 6,590 | 2024-06-18 → 2026-10-19 | One row per website order: status, Ready/Future type, delivery date and slot, prices |
| `product_order` | 7,182 | 2024-06-18 → 2026-10-19 | Order lines: order id, quantity, price, and `productCopy`, a JSON copy of the product **as of purchase time** (product id taken from it). The free-text gift note `customerDescription` is dropped unread |
| `product` | 1,956 | snapshot at export | Name, active flags, created/updated time. **Current state only, no history** |
| `product_category` | 3,043 | — | Product ↔ category (many-to-many) |
| `category` | 69 | — | Tree (`parentId`); `dateShifter` = minimum lead time in days |

Join keys: `anonymousId` (browser) links the two event tables; `customerId` links logged-in events to orders; `productId` links events to products.

Facts confirmed by the owner (2026-10-19):
- `final` = payment succeeded and the order is done.
- Category 223 and its subcategory are same-day products; every other category is scheduled delivery.
- Timestamps are the shop's local time, with no daylight-saving changes in the data period.
- `userAgent` and `categoryId` were added on 2026-08-23.
- Only customers order on the website; staff orders for others are placed outside the site.
- Visitors normally add to cart anonymously and must log in right before the shipping step (earlier login is rare). From login on, every row should carry the customer id. One browser (`anonymousId`) = one customer, and the `anonymousId` should not change.

## 2. Event cleaning (`build_own_events.py`)

Retail Rocket rules are reused unchanged from `0-Shared/clean_events.py` (bot thresholds, 30-minute session gap). Steps marked **new** exist only here.

| # | Step | Effect |
|---|---|---|
| 1 | Load, type-check ids and timestamps | 0 bad timestamps, ids unique |
| 2 | **New:** merge `cart_event` as `addtocart`; drop cart rows before tracking started (Jul 14 – Aug 9, no surrounding behaviour) | −378 cart rows → 1,530 adds |
| 3 | **New:** crawlers by user agent (bot/crawl/spider/headless/…) | −2,223 visitors, −2,224 events (all views and category views; no adds or checkouts) |
| 4 | **New:** front-end double fires: same browser, event, item and payload within 2 s | −639 events (371 search, 91 `final`, 83 view, 79 category view, 15 other) |
| 5 | Retail Rocket bot rules (>500 events/h, or interval CV < 0.05 with ≥20 events) | −6 visitors, −83 events (browsing only) |
| 6 | Sessions: 30-minute gap per browser | 12,791 sessions |
| 7 | **New:** `person_key` = customer id if the browser ever logged in, else browser id (for visitor-grouped splits) | 8,508 browsers → 8,366 persons |

Kept but flagged: 5 bulk adds (quantity 6–11).

Result: **61,048 events**. View 27,725 · category view 24,973 · checkout cart 2,441 · address 1,673 · add 1,530 · payment 1,047 · search 750 · cart remove 460 · final 449.

## 3. Reference tables and label (`build_own_tables.py`)

**Order status codes**, inferred from the data:

| statusId | Orders | Reading |
|---|---|---|
| 8 | 4,943 | **Paid / completed.** 4,939 have paidPrice > 0 |
| 2 | 1,468 | Created at the payment step, **never paid** (1,458 unpaid). Abandonment at the payment gateway |
| 3 | 88 | Unpaid, probably cancelled |
| 9, 10, 11, others | 91 | Rare, mostly paid. Meaning not confirmed |

**Product type.** Every product is either daily (287 products: category 223/224, lead time 0) or scheduled (lead time 1–5 days); none is both. All viewed and carted products exist in the mapping.

**Linking `final` events to orders.** The order row is written at the payment step. `final` follows it after a median of **58 s**. Each `final` is matched to the same customer's nearest order within 1 hour:
- 412 of 449 `final` events link to 347 distinct orders; 332 of those orders (96%) are status 8.
- **Repeat `final` events:** 47 sessions repeat `final` for the **same** order (page refresh), and 9 sessions place several orders. So count orders, not `final` events.
- Since tracking began there were 341 paid orders, and **332 (97%) have a `final` event**. Of 182+ unpaid status-2 orders, only 5 have a `final` (nonsense rows, kept and flagged).

**Session label.** One row per cart session (`cart_sessions_labels.parquet`):

| Label | Definition | Rate |
|---|---|---|
| `y_final` | A `final` event after the session's first add | 34.1% |
| `y_paid_order` | A linked status-8 order after the first add | 32.9% |

The two agree on **98.9%** of the 972 cart sessions. `y_final` is the main label because it's available in the event stream itself; `y_paid_order` is the audited check.

**Item-level label check** (paper 1, §IV-C). Purchased items come from the order lines of each session's linked paid orders.

| | Sessions (of 320 paid cart sessions) |
|---|---|
| Carted an item that wasn't bought | 33 |
| …and didn't remove it first (a true purchase + abandonment mix) | **3 (0.9%)** |
| Bought an item not carted in that session | 0 |

Retail Rocket had 11.3% of purchased sessions with this defect, because it has no remove event. Here the `cart_remove` event resolves almost all of it. The 3 sessions are flagged in `mixed_purchase_abandon`.

**Order lines vs product type.** Lines of daily products are `Ready` in 2,437 of 2,689 cases (252 daily products were ordered for a later day). Scheduled products are `Future` in 2,843 of 2,844. 1,649 older lines (mostly 2024) are products with no current category mapping; this doesn't affect any tracked session.

## 4. Size

| | |
|---|---|
| Cart sessions | **972** (828 after the Aug 23 tracking change) |
| Paid | 331 by `y_final` |
| Growth | ~115 cart sessions/week, so ≈1,900 by mid-Dec and ≈2,600 by end-Jan |
| First carted item is daily | 61% of cart sessions |
| Purchase rate by first item | **daily 44.8%**, scheduled 17.4% |

This is enough to detect large leakage effects (AUC gaps of 0.2–0.3; SE ≈ 0.04 on a 20% test split) but not small ones (≤0.03). Use it as the **real-world second dataset** next to a large public one (REES46).

## 5. Data-quality issues and tracking changes

1. **Tracking change on Aug 23.** `userAgent` and `categoryId` start then, along with the `viewCategory` event. Before Aug 23, crawlers can't be removed by user agent: single-event sessions are 47.9% before and 38.6% after. Cart sessions are unaffected (crawlers never add), but **any feature using user agent, device detail or category views is undefined for Aug 9–22.** Recommended modelling window: **from 2026-08-23** (828 cart sessions). Keep Aug 9–22 as a documented example of a tracking change.
2. **Double-fired events** from the front end (step 4 above), including 91 duplicate `final` events.
3. **Repeat `final` events for one order** (page refresh), 47 sessions. Handled through the order link.
4. **5 unpaid orders with a `final` event** and 9 paid orders without one. Kept as is.
5. **Traffic dip** in the week of Oct 12 (views and adds both fell), so it's real demand, not missing logs.
6. **`product` is an export-time snapshot.** `updatedAt` for 364 products is after tracking started, so any product attribute joined from it is a *latest* join, not an *as-of* join.
7. One order line has quantity 0. Kept as is.
8. **`cart_event.customerId` is written after the add.** 54% of cart rows that carry a customer id come before the browser's first logged-in event, so the id is filled in at a later login. Never use it as "logged in at add time"; use the first analytics event with a customer id. Details in `docs/own_dataset_audit.md`.
9. **Customer id missing after login** on 21% of analytics rows, including a third of `final` events (possibly the bank-gateway return page). That's why `final` events are linked to orders through the browser → customer mapping.
10. **8 browsers carry more than one customer id** (against the one-customer rule). `person_key` takes the most frequent one.

## 6. Leakage traps this dataset offers (for the checklist)

| Trap | Where | Paper-1 analogue | Effect (measured ones from `docs/own_dataset_audit.md`) |
|---|---|---|---|
| Checkout funnel steps (`payment` reached) | events | Target proxy | **Measured (B1):** 0.82–0.88 vs 0.57–0.63 without |
| Purchase history counted over the whole data, including the current purchase | events, orders | Self-inclusion aggregate (Retail Rocket probe: 0.66 → 0.98 AUC) | **Measured (B3):** 0.93–0.94 vs ~0.6 |
| Completed-session timing (age, recency to the last event) | events | Completed-endpoint timing | **Measured (A):** XGB 0.76–0.78 vs 0.58–0.62 counts only |
| Customer id on cart rows (written at a later login) | cart_event | Post-decision value in a pre-decision row | Observed in the data; not evaluated (login features parked, group C) |
| Product active flag from the export-time snapshot | products | Latest-snapshot join | **Small.** Purchase rate is 40.7% for daily items inactive now vs 48.3% for active ones; it doesn't encode the purchase |
| Order `updatedAt` / status at export time | orders | Post-outcome field | Large if used as a feature |
| Product state from `productCopy` (purchase-time snapshot) | order lines | Missingness that encodes the label | Looks point-in-time-correct, but it **exists only for purchased items**, so "has a snapshot" equals the label |
| Cart in one session, purchase in a later one | events | Label ambiguity | 43 sessions reach `final` without an add in the same session |
| Tracking change (Aug 23) | events | — (new) | Features undefined for 2 weeks; affects temporal splits |

## 7. Still missing (optional)
- **Meanings of the rare status codes** 9, 10, 11 and of `paymentType` 1 vs 4. Nothing depends on them.
