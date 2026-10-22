# Paper 2, first results: the paper-1 audit on the flower-shop data

Rerun and rewritten 2026-10-20. Data: `docs/own_dataset.md`. Visual summary (private
artifact): https://claude.ai/artifact/HE2RPDrJoPB4jVTa82Ae8f

The results come in three groups:

| Group | What | Script | Output (`5-Results/`) |
|---|---|---|---|
| **A · same as paper 1** | Paper 1's evaluations, nothing added or left out | `7-Own-Data/replicate_paper1.py` | `own-data__paper1-replication__{runs,summary}.csv` |
| **B · useful extras** | Tests only this shop allows, run under group A's protocol or (B3) under the Retail Rocket probe's | `7-Own-Data/extras_funnel_product.py`, `7-Own-Data/history_window_test.py` | `own-data__extras-funnel-product.csv`, `own-data__history-window.csv` |
| **C · parked** | Login as a feature | — | — |

Population: 828 cart sessions from 2026-08-23 (the day tracking became complete), 33.6% paid.
The first-add-to-cart table has 825 (3 paid before their first add, removed as paper 1 removed 172).
Each test set holds ~165 sessions, so **differences below ~0.05 are noise** (bootstrap 95% CIs are about ±0.09).

---

## Group A · Same as paper 1

### Setup, line by line

| Part | Paper 1 (Retail Rocket) | Flower shop | |
|---|---|---|---|
| Sessions | 30-minute gap | 30-minute gap | same |
| Bot rules | >500 events/h or interval CV < 0.05 | Same, plus crawlers by user agent and front-end double fires | data needs it |
| Purchase event | `transaction` | `checkout_final` (98.9% agreement with paid orders) | data differs |
| Event types in a session | view, add, transaction | Also category views, searches, removes, checkout steps | data differs |
| Feature function and 4 configurations | paper 1 code | copied | same |
| 13 point-in-time features | paper 1 code | copied | same |
| Seeds, 80/20, cap and scaling fitted on train only | 11, 22, 33, 42, 55 | same | same |
| Visitor id for the grouped split | `visitorid` (browser) | `anonymousId` (browser) | same |
| Time split | sessions sorted by start | same | same |
| Models | LR + XGB at 5 seeds; DT + RF at seed 42; bootstrap CI at seed 42 (first add) | same | same |

### Completed-session table (ROC-AUC, mean of 5 seeds)

| Feature set | Model | Flower shop: strat / visitor / time | Retail Rocket: strat / visitor / time |
|---|---|---|---|
| Count shortcut (events, views, adds) | LR | 0.73 / 0.72 / 0.69 | 0.994 / 0.993 / 0.995 |
| | XGB | 0.70 / 0.71 / 0.65 | 1.000 / 1.000 / 1.000 |
| Original completed (counts + 3 timing) | LR | 0.74 / 0.72 / 0.70 | 0.831 / 0.832 / 0.842 |
| | XGB | 0.77 / 0.78 / 0.76 | 0.995 / 0.995 / 0.996 |
| | DT, seed 42 | 0.73 / — / — | 0.934 / — / — |
| | RF, seed 42 | 0.83 / — / — | 0.984 / — / — |
| Timing only | LR | 0.67 / 0.64 / 0.68 | 0.801 / 0.804 / 0.811 |
| | XGB | 0.70 / 0.69 / 0.67 | 0.869 / 0.871 / 0.876 |
| Counts only | LR | 0.63 / 0.63 / 0.57 | 0.606 / 0.610 / 0.620 |
| | XGB | 0.60 / 0.62 / 0.58 | 0.606 / 0.612 / 0.618 |

### First add-to-cart, 13 point-in-time features (ROC-AUC)

| Model | Flower shop: visitor / time | 95% CI at seed 42: visitor / time | Retail Rocket: visitor / time |
|---|---|---|---|
| LR, 5 seeds | 0.57 / 0.63 | 0.42–0.61 / 0.53–0.72 | 0.630 / 0.641 |
| XGB, 5 seeds | 0.57 / 0.59 | 0.45–0.64 / 0.49–0.68 | 0.622 / 0.633 |
| DT, seed 42 | 0.53 / 0.53 | — | 0.646 / 0.647 |
| RF, seed 42 | 0.58 / 0.55 | — | 0.582 / 0.568 |
| **Range, 4 models** | **0.53–0.63** | | **0.57–0.65** |

### Label checks (paper 1, §IV-C)

| | Retail Rocket | Flower shop |
|---|---|---|
| Paid before the first add-to-cart (removed) | 172 | 3 |
| Paid sessions that also abandoned a carted item | 11.3% | 0.9% (3 of 320) |

The shop logs cart removals, which resolves most of this defect.

### What group A shows
1. **Paper 1's main result repeats.** At the first add-to-cart both shops land around 0.6 (0.53–0.63 vs 0.57–0.65); the intervals overlap fully.
2. **The completed-session leak repeats but is smaller** (original completed XGB 0.76–0.78 vs counts only 0.58–0.62; Retail Rocket 0.995 vs 0.61). The reason is measured (`docs/paper2/robustness.md`): only 0.7% of abandoned visits here end right on an add-to-cart (Retail Rocket 60.6%), so "time since the last add-to-cart" doesn't separate the classes. Buyers end on the purchase about as often as on Retail Rocket (72% vs 69%). *(Corrected 2026-10-20: an earlier version said buyers keep browsing after paying; the data shows it's the non-buyers who keep browsing.)*
3. **The count shortcut doesn't exist here.** event − view − cart = purchases holds in 0% of sessions because the shop logs more event types.

---

## Group B · Useful extras

### B1 · Checkout steps as features (`extras_funnel_product.py`, group A protocol)

**Why:** this shop has a checkout funnel and Retail Rocket doesn't. The steps happen after the first add-to-cart, a kind of leak none of the nine questions asks about.

| Features | LR: strat / visitor / time | XGB: strat / visitor / time |
|---|---|---|
| Counts only (group A) | 0.63 / 0.63 / 0.57 | 0.60 / 0.62 / 0.58 |
| Counts + checkout steps | **0.88 / 0.86 / 0.82** | **0.85 / 0.87 / 0.83** |

Every paid session passed the payment step (278 of 278), and 205 unpaid sessions did too.

### B2 · Product facts known at the first add (`extras_funnel_product.py`, group A protocol)

**Why:** fair features (known at the moment of the add) that answer the earlier question of whether better features help without leaking. Added: same-day (daily) product, lead time, price, category views and searches so far, mobile. No customer id or login.

| Features | LR: visitor / time | XGB: visitor / time |
|---|---|---|
| 13 features (group A) | 0.57 / 0.63 | 0.57 / 0.59 |
| 13 + product facts | **0.66 / 0.71** | **0.64 / 0.62** |

Gain +0.04 to +0.09. Daily products are bought in 44.2% of cart sessions, scheduled ones in 17.0%.

### B3 · History over the wrong time window (`history_window_test.py`, Retail Rocket probe protocol)

**Why:** repeats our Oct 2 Retail Rocket probe (`6-Point-In-Time/probes/agg_probe.py`) exactly: same two features (the carted item's purchase rate, the visitor's purchase count), three time windows, the same XGBoost settings (200 trees, depth 4), seeds and splits. The visitor is the browser id; no customer id or login is used in any feature.

| Features (XGBoost) | Time window | Flower shop: visitor / time | Retail Rocket probe: visitor / time |
|---|---|---|---|
| 13 base features | — | 0.59 / 0.61 | 0.645 / 0.664 |
| + history | before the add (correct) | 0.62 / 0.59 | 0.683 / 0.711 |
| + history | whole data minus this visit | 0.69 / 0.67 | 0.679 / 0.726 |
| + history | **whole data including this visit** | **0.94 / 0.93** | **0.973 / 0.977** |

1. **Counting the visit's own purchase is catastrophic in both shops.** A routine "lifetime purchases" column is enough.
2. **Future purchases leak more here** (+0.08 over base): 29 of 825 visitors had bought before their first add, but 117 buy at some point.
3. **Correctly timed history doesn't help yet** (within noise): two months of data leaves few visitors with a past.

The base score differs from group A (0.59/0.61 vs 0.57/0.59) because the probe uses its own XGBoost settings and no cap, as on Retail Rocket (0.645 vs 0.622 there).

---

## Group C · Parked: login as a feature

Login came up because a customer id was needed for two jobs: confirming the label against the orders table (done: 98.9% agreement) and, at one point, for grouping. Neither needs a "logged in" feature, and none is used in groups A or B.

It's parked for two reasons:
- **The signal is a funnel artifact.** The site asks for login right before shipping, so "logged in" mostly means "already in checkout".
- **The login time can't be measured reliably.** The two sources disagree: cart rows say 78% were logged in at the first add, the event stream says 22.6%.

What stays is a fact about the data, recorded in `docs/own_dataset.md`: the site writes the customer id onto earlier cart rows at a later login, and 21% of events after a login lack the id. The earlier login experiments are in git history (commits `d42b274`, `f211db4`) and are not part of the results.

---

## The nine questions on this data

| # | Question | Answer here | Leak flagged? |
|---|---|---|---|
| 1 | Exact prediction moment? | First add-to-cart (k=0) | — |
| 2 | Which event creates the label? | `checkout_final` (payment succeeded); 98.9% agreement with paid orders | — |
| 3 | Label of the decision or the session? | 3 of 320 paid sessions (0.9%) mix purchase and abandonment | Minor |
| 4 | Event count reconstructable from others? | No, the identity holds in 0% of sessions | Correctly no |
| 5 | Duration/recency/average uses the final event? | Yes: age, mean interval and recency run to the session's last event | **Yes** (group A) |
| 6 | Endpoint selected differently by class? | Yes: paid sessions run past `final`, unpaid ones end at the last activity | **Yes** (same leak) |
| 7 | Preprocessing fitted on train only? | Yes (cap and scaler) | — |
| 8 | Plausible after removing suspect groups? | Counts only 0.57–0.63; 13 point-in-time features 0.53–0.63 | — |
| 9 | Rerunnable? | Yes: five scripts, committed aggregate outputs | — |

**Leaks the nine questions miss**, as candidates for checklist v2:

| Proposed question | Evidence here |
|---|---|
| Does any feature record a step of the process that produces the label (funnel, checkout, payment) after the prediction moment? | B1: 0.82–0.88 vs 0.57–0.63 |
| Is any aggregate (customer history, item popularity) computed over a window that includes the scored case or later? | B3: 0.93–0.94 vs ~0.6, on both shops |
| Did the logging schema change during the data period? | Observed (`userAgent`/`categoryId` added Aug 23); not measured as AUC |

## Caveats
- Small sample: 825–828 sessions, ~165 per test set.
- k=0 only; later cutoffs would leave too few sessions at this size.
- The temporal split is one fixed cut, so its spread across seeds is 0.
- In B3, the visitor's purchase count counts `final` events, as the probe counted `transaction` events; a page refresh after paying adds a second event (47 sessions).
