# Paper 2 · Robustness: held-out month, unrelated shop, confidence intervals, leak-size mechanism

Run 2026-10-20. Covers items 2–4 of the hub's gap list (`README.md`).

## 1. Held-out month: REES46 November 2019

The bot rule and its thresholds (1.5× median daily cart sessions; flagged days' purchase
rate below 60% of the others) were fixed in commit `9ef4265`, **before** November was opened.
The rule lives in `0-Shared/burst_check.py`, and November was run through the same scripts
(`8-REES46/*.py --month 2019-Nov`).

| | |
|---|---|
| Raw rows | 4,635,837; 246,693 exact duplicates dropped |
| Paper 1's bot rules | 512 users, 16,706 events, 1,705 purchase rows removed |
| Cart sessions | **181,223**, 17.2% with a purchase |

**Bot rule, applied blind:** it flags 2 days, **Nov 22 and Nov 28** (the run-up to Black
Friday and Thanksgiving), and classifies both as **not automated**: purchase rate 18.5% vs
17.1%, night share 8.0% vs 9.5%. That's a correct negative on a real sales spike, so the
whole month is used.

**Group A, November (ROC-AUC, mean of 5 seeds):**

| Feature set | Model | Strat / visitor / time | October clean | Retail Rocket |
|---|---|---|---|---|
| Count shortcut | XGB | 0.91 / 0.91 / 0.92 | 0.91 | 1.000 |
| Original completed | XGB | 0.965 / 0.965 / 0.966 | 0.96 | 0.995 |
| | DT / RF, seed 42 | 0.89 / 0.95 | 0.89 / 0.94 | 0.934 / 0.984 |
| Timing only | XGB | 0.87 / 0.87 / 0.87 | 0.86 | 0.869–0.876 |
| Counts only | XGB | 0.74 / 0.74 / 0.75 | 0.72–0.73 | 0.606–0.618 |
| **First add-to-cart, 4 models** | | visitor 0.49–0.56, time 0.50–0.56 | 0.49–0.55 | 0.57–0.65 |

At the first add-to-cart, the visitor and time splits agree in November (LR 0.54 / 0.55, XGB
0.54 / 0.55; 95% CIs within 0.537–0.559). The October time-split collapse doesn't appear,
which supports the conclusion that it was caused by the bot days.

**Label check, November:** 272 sessions bought before their first add-to-cart; 1,178 of 31,222
purchased sessions (3.8%) also abandoned a carted item they didn't remove (October 3.9%).

**Group B, November:**

| Features (XGBoost, visitor / time) | November | October clean |
|---|---|---|
| 13 base features | 0.56 / 0.56 | 0.55 / 0.55 |
| + history before the add (correct) | 0.56 / 0.57 | 0.56 / 0.56 |
| + history, whole month minus this visit | 0.65 / 0.64 | 0.62 / 0.60 |
| + history, whole month **including this visit** | **0.87 / 0.89** | **0.89 / 0.90** |
| Product facts (gain over the 13 features, LR and XGB) | +0.01 to +0.02 | +0.01 to +0.03 |

Every October result repeats in the held-out month.

## 2. Confidence intervals for the headline gaps

`9-Paper2/headline_cis.py` → `5-Results/paper2__headline-cis.csv`. Leaky minus fair ROC-AUC
on the same test set (seed 42, visitor-grouped split, XGBoost), paired bootstrap with 1,000
resamples. Each gap uses the protocol that produced it.

| Gap | Retail Rocket | Flower shop | REES46 (Oct, clean) |
|---|---|---|---|
| L2 completed-session timing | +0.396 [0.382, 0.409] | +0.122 [0.021, 0.228] | +0.243 [0.235, 0.252] |
| L4 history incl. the purchase | +0.275 [0.263, 0.287] | +0.338 [0.255, 0.427] | +0.332 [0.322, 0.341] |
| L6 checkout steps | — | +0.255 [0.161, 0.353] | — |

Test sets: 8,705–8,871 (Retail Rocket), 162 (flower shop), ~23,100 (REES46). **Every gap's
interval excludes zero.** The flower shop's intervals are wide, but even its lowest bound is
positive.

## 3. Why the completed-session leak differs in size

`9-Paper2/leak_size_mechanism.py` → `5-Results/paper2__leak-size-mechanism.csv`, on the cart
sessions each analysis used.

| | Retail Rocket | REES46 | Flower shop |
|---|---|---|---|
| **Abandoned visits that end right on an add-to-cart** | **60.6%** | **48.3%** | **0.7%** |
| Purchased visits that end on an add-to-cart | 2.1% | 2.0% | 0.0% |
| Purchased visits that end on the purchase | 68.8% | 88.4% | 72.3% |
| Median visit length, purchased / abandoned (s) | 821 / 172 | 1,239 / 508 | 993 / 606 |
| L2 gap (XGB, section 2) | +0.40 | +0.24 | +0.12 |

The leak follows **how abandoned visits end.** When most non-buyers leave right after adding
to cart, "time since the last add-to-cart, up to the visit's end" is zero for them and
positive for buyers, so it nearly gives away the label. In the flower shop, non-buyers keep
browsing after adding to cart, and the signal fades. How buyers end doesn't explain the
difference: they end on the purchase in 69–88% of visits everywhere. Retail Rocket's 60.6% and
2.1% reproduce paper 1's published figures (§IV-B).

*(This corrects an earlier, unmeasured explanation, "flower-shop buyers keep browsing after
paying", which appeared in `docs/own_dataset_audit.md` and on the results page.)*

## 4. Unrelated shop: Taobao (added 2026-10-21)

**Data.** Taobao "User Behavior" (Alibaba; China, 25 Nov – 3 Dec 2017; 100,150,807 rows,
987,994 users). A fixed 10% user sample (user ID divisible by 10) keeps each sampled user's
full history: 98,768 users, **323,471 cart sessions**, 10.6% with a purchase. Paper 1's
cleaning rules (`10-Taobao/build_taobao_events.py`) found no bots. 5,455 sampled rows with
impossible dates and 6 duplicates were dropped. The group A and B scripts are the same as
REES46's, run with `--dataset taobao`.

**The rules were fixed before the data was opened.** The bot rule and its thresholds date
from commit `9ef4265`. The leak-size prediction below was committed in `6a2a65c`, before
Taobao's group A result existed.

**Group A (ROC-AUC, XGBoost; strat / visitor / time):**

| Feature set | Taobao | Retail Rocket |
|---|---|---|
| Count shortcut (purchases + favourites remain) | 0.96 / 0.96 / 0.96 | 1.000 |
| Original completed | 0.930 / 0.930 / 0.925 | 0.995 |
| Timing only | 0.74 / 0.74 / 0.73 | 0.87 |
| Counts only | 0.78 / 0.78 / 0.77 | 0.61–0.62 |
| **First add-to-cart, 4 models** | visitor 0.48–0.55 · time 0.48–0.55 | 0.57–0.65 |

**Leak-size prediction: confirmed, with one part unresolved.** Taobao's abandoned visits end on an
add-to-cart in 36.4% of cases, but its buyers do so too (15.2%), because they often keep browsing
after paying (median 6 min). The difference of 21.2 points is smaller than REES46's 46.3, so the
committed prediction was a completed-session leak below REES46's (+0.24) and above the flower
shop's (+0.12).

| | Retail Rocket | REES46 | Taobao | Flower shop |
|---|---|---|---|---|
| Difference in how visits end (points) | 58.5 | 46.3 | 21.2 | 0.7 |
| L2 gap, paired bootstrap 95% CI | +0.396 [0.382, 0.409] | +0.243 [0.235, 0.252] | **+0.155 [0.150, 0.160]** | +0.122 [0.021, 0.228] |

Smaller than REES46: yes. Larger than the flower shop: not resolvable, because the flower
shop's interval contains Taobao's value.

**Group B:** history including the scored purchase gives 0.74 / 0.76 against 0.55 / 0.54 for
the base features; the paired gap is **+0.165 [0.157, 0.173]**. The effect is smaller than on
the other datasets, plausibly because half the visitors (153,120 of 310,413) had bought before
their first add-to-cart, so one more purchase changes their count less (not tested). Correct
history adds +0.02 to +0.03; later purchases add a further +0.01 (weak); product facts
(categories and favourites so far) add +0.01 to +0.02.

**Label check:** 13,058 cart sessions (4.0%) bought before their first add-to-cart, and
**31,026 of 34,204 purchased sessions (90.7%) also carted an item they didn't buy**. Taobao's
cart works like a wish list, so the session-level label (paper 1's Q3) is far more ambiguous
here than on the other datasets (0.9–11.3%).

**Bot rule:** no day flagged (China time), a correct negative.
