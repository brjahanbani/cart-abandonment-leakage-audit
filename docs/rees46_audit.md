# Paper 2, third dataset: REES46 cosmetics shop (October 2019)

Run 2026-10-20. Public data: Kaggle "eCommerce Events History in Cosmetics Shop" (REES46),
file `2019-Oct.csv`, in git-ignored `1-Data/rees46/raw/`. Group A (exact repeat of
paper 1) and group B (history window, product facts; REES46 has no checkout funnel).

| Step | Script (`8-REES46/`) | Output (`5-Results/`) |
|---|---|---|
| Profile (read-only) | `profile_rees46.py` | `rees46__profile.json` |
| Clean + 30-minute sessions | `build_rees46_events.py` | `rees46__clean.json` |
| Group A, all October / clean window | `replicate_paper1_rees46.py` / `... --from 2019-10-09` | `rees46__paper1-replication__{all-october,from-2019-10-09}__{runs,summary}.csv`, `rees46__label-checks__*.json` |
| Why the clean window | `probe_temporal_shift.py` | printed only |
| Group B, clean window | `group_b_rees46.py` | `rees46__history-window.csv`, `rees46__extras-product.csv` |

The protocol is paper 1's own code, shared with the flower-shop run through
`0-Shared/paper1_protocol.py`. With 205k sessions the features are computed in bulk. Before
any model runs, they're checked against paper 1's reference functions on 3,000 random
sessions; the check passed for every feature.

## 1. Data and cleaning

| | |
|---|---|
| Raw rows | 4,102,283 (view 1.86M, cart 1.23M, remove_from_cart 0.76M, purchase 0.25M) |
| Exact duplicate rows dropped | 213,155 (87% repeated `remove_from_cart`) |
| Paper 1's bot rules (unchanged) | 441 users, 12,405 events, **1,280 purchase rows** removed (0.5%). These users' whole month is a burst inside one minute |
| Paper 1's 30-minute sessions | 655,665; **205,351 cart sessions**, 12.3% with a purchase |

Mapping: `cart` → add-to-cart, `purchase` → transaction, `user_id` → visitor id (persistent,
like Retail Rocket's `visitorid`). The site's own `user_session` isn't used.

Differences from Retail Rocket that the paper must state:
- Timestamps are in **whole seconds** (Retail Rocket: milliseconds), so the order of same-second events is unknown. Prefix features are unaffected, because all events at or before the cutoff second enter the prefix, as in paper 1. No purchase shares a second with a session's first add-to-cart.
- REES46 logs `remove_from_cart`, so paper 1's count identity (event − view − cart = purchases) can't hold exactly here. The remainder is purchases + removals.

## 2. Bursts of automated-looking traffic (Oct 1–8)

Most days have about 5,000 cart sessions, ~16% of which buy. Oct 2, 6 and 7 have 14,000–23,000, and Oct 1 and 8 about 8,400:

| | Burst days (Oct 2, 6, 7) | Other days |
|---|---|---|
| Cart sessions | 57,548 from 53,881 different users | 147,803 |
| Sessions by hour | Flat, ~1,000 an hour including 2–4 AM | Day curve (≈40 at night, ≈330 at peak) |
| Median visit | 20 s, 4 events | 548 s, 9 events |
| Purchase rate | 4.6% | 15.3% |

A flat round-the-clock profile from tens of thousands of one-visit user ids looks like automated traffic that rotates identities. Paper 1's bot rules judge each visitor separately, so each fake identity has too few events to be caught.

**The clean window starts Oct 9** (sessions starting before it are dropped): 115,580 cart sessions, 16.1% with a purchase. Both runs are reported.

## 3. Group A results (ROC-AUC, mean of 5 seeds)

### Completed-session table

| Feature set | Model | All October: strat / visitor / time | **Clean (Oct 9–31): strat / visitor / time** | Retail Rocket: strat / visitor / time |
|---|---|---|---|---|
| Count shortcut | LR | 0.92 / 0.92 / 0.89 | 0.89 / 0.89 / 0.90 | 0.994 / 0.993 / 0.995 |
| | XGB | 0.94 / 0.94 / 0.91 | 0.91 / 0.91 / 0.91 | 1.000 / 1.000 / 1.000 |
| Original completed | LR | 0.84 / 0.84 / 0.79 | 0.79 / 0.80 / 0.80 | 0.831 / 0.832 / 0.842 |
| | XGB | 0.97 / 0.97 / 0.96 | **0.96 / 0.96 / 0.96** | 0.995 / 0.995 / 0.996 |
| | DT, seed 42 | 0.92 | 0.89 | 0.934 |
| | RF, seed 42 | 0.96 | 0.94 | 0.984 |
| Timing only | LR | 0.85 / 0.85 / 0.79 | 0.79 / 0.79 / 0.79 | 0.801 / 0.804 / 0.811 |
| | XGB | 0.90 / 0.90 / 0.86 | **0.86 / 0.86 / 0.86** | 0.869 / 0.871 / 0.876 |
| Counts only | LR | 0.74 / 0.74 / 0.66 | 0.66 / 0.66 / 0.66 | 0.606 / 0.610 / 0.620 |
| | XGB | 0.79 / 0.80 / 0.72 | **0.72 / 0.72 / 0.73** | 0.606 / 0.612 / 0.618 |

### First add-to-cart, 13 point-in-time features

| Model | All October: visitor / time | **Clean: visitor / time** | 95% CI at seed 42, clean: visitor / time | Retail Rocket: visitor / time |
|---|---|---|---|---|
| LR, 5 seeds | 0.60 / 0.48 | **0.54 / 0.55** | 0.52–0.54 / 0.54–0.56 | 0.630 / 0.641 |
| XGB, 5 seeds | 0.65 / 0.53 | **0.53 / 0.53** | 0.53–0.55 / 0.52–0.54 | 0.622 / 0.633 |
| DT, seed 42 | 0.66 / 0.53 | 0.54 / 0.54 | — | 0.646 / 0.647 |
| RF, seed 42 | 0.59 / 0.49 | 0.50 / 0.49 | — | 0.582 / 0.568 |
| **Range** | 0.48–0.66 | **0.49–0.55** | | 0.57–0.65 |

Rows: 205,123 (all October) and 115,414 (clean); 228 and 166 sessions that bought before their first add-to-cart were removed.

### Label checks (paper 1, §IV-C)

| | Retail Rocket | Flower shop | REES46, clean window |
|---|---|---|---|
| Bought before the first add-to-cart (removed) | 172 | 3 | 166 |
| Purchased sessions with a carted item not bought | 11.3% (no remove event) | — | 50.4% (9,357 of 18,560) |
| … and not removed first | — | 0.9% | **3.9%** (722) |

## 4. What it shows

1. **The completed-session leak repeats strongly.** On clean data, original completed features reach 0.96 with XGBoost against 0.72 for counts alone. Timing alone gives 0.86, almost exactly Retail Rocket's 0.87. This is much closer to paper 1 than the flower shop was (XGB 0.76–0.78). The reason is measured in `docs/paper2/robustness.md`: 48.3% of REES46's abandoned visits end right on an add-to-cart (Retail Rocket 60.6%, flower shop 0.7%).
2. **The count shortcut becomes a strong proxy instead of an exact one** (0.89–0.91). The identity fails only because removals are counted too.
3. **At the first add-to-cart, paper 1's 13 features carry almost no signal here (0.49–0.55)**, lower than Retail Rocket (0.57–0.65) and the flower shop (0.53–0.63). The intervals are narrow, so this is a real difference, not noise. The gap between leaky and fair evaluation is even larger than in paper 1.
4. **Undetected automated traffic inflates a point-in-time score.** With the burst days included, the visitor split gives 0.59–0.66, because the models learn to tell burst sessions from shoppers. The time split collapses (0.48–0.53) because the bursts sit only in its training weeks. Paper 1's per-visitor bot rules don't catch identity-rotating traffic. Candidate checklist question: *Is the evaluated population free of traffic the bot filter can't see? Check daily volumes and hour-of-day profiles before trusting a score.*

## 5. Group B (clean window)

### B3 · History over the wrong time window

The Oct 2 Retail Rocket probe (`6-Point-In-Time/probes/agg_probe.py`) repeated exactly: the
first carted item's purchase rate and the visitor's purchase count, three time windows, the
probe's XGBoost settings (200 trees, depth 4), seeds and splits. Counts use every clean
October event, including sessions without an add-to-cart and Oct 1–8 as history.

| Features (XGBoost) | Time window | REES46: visitor / time | Flower shop | Retail Rocket probe |
|---|---|---|---|---|
| 13 base features | — | 0.55 / 0.55 | 0.59 / 0.61 | 0.645 / 0.664 |
| + history | before the add (correct) | 0.56 / 0.56 | 0.62 / 0.59 | 0.683 / 0.711 |
| + history | whole month minus this visit | 0.62 / 0.60 | 0.69 / 0.67 | 0.679 / 0.726 |
| + history | **whole month including this visit** | **0.89 / 0.90** | **0.94 / 0.93** | **0.973 / 0.977** |
| history alone | before the add (correct) | 0.53 / 0.52 | 0.56 / 0.58 | 0.611 / 0.633 |
| history alone | whole month including this visit | 0.89 / 0.90 | 0.94 / 0.93 | 0.971 / 0.977 |

- **Self-inclusion is catastrophic in all three shops** (0.89–0.98 against 0.55–0.66 without it).
- **Future purchases without this visit leak here too** (+0.05 to +0.07).
- **Correct history adds +0.01**, although 16,768 of the 115,414 visitors (14.5%) had bought before their first add-to-cart. Past purchases say little about whether this cart converts.

### B2 · Product facts known at the first add-to-cart

Group A's protocol; the reference row reproduces group A exactly. Added: the first item's price, whether it has a brand, distinct categories seen so far, removals so far.

| Features | LR: visitor / time | XGB: visitor / time |
|---|---|---|
| 13 features (group A) | 0.54 / 0.55 | 0.53 / 0.53 |
| 13 + product facts | 0.55 / 0.56 | 0.56 / 0.56 |

Gain +0.01 to +0.03 (intervals about ±0.01). The flower shop's product facts helped more (+0.04 to +0.09), mainly through its same-day vs scheduled split, which REES46 has no equivalent for.

## 6. Caveats
- November 2019 was run as a held-out month (`docs/paper2/robustness.md`): no automated days by the pre-fixed rule, and every October result repeats. Further months (Dec 2019 – Feb 2020) remain available.
- The clean window is a day-level cut. Real shoppers on Oct 1–8 are dropped along with the bursts.
- Temporal split: a single fixed cut (sd 0).
