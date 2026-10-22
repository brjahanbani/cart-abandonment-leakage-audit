# Paper 2 · Detection matrix

Built by `9-Paper2/build_detection_matrix.py` from the committed result files.
Datasets: Retail Rocket, flower shop, REES46 October, plus two opened **after every rule and
threshold was committed**: REES46 November (held-out month) and **Taobao (unrelated shop,
added 2026-10-21)**. Outputs: `5-Results/paper2__detection-matrix.csv`, `5-Results/paper2__burst-check.csv`.
Confidence intervals and the leak-size mechanism: `robustness.md`. False alarms: `negative_control.md`.

**Question:** for each leak mechanism and each dataset, is the leak present, does paper 1's
nine-question checklist (v1) flag it, and what does it cost?
**Effect:** XGBoost ROC-AUC, leaky configuration vs its fair comparator, as the range over the splits run.

## The matrix

| Mechanism | v1 | Retail Rocket | Flower shop | REES46 Oct | REES46 Nov (held out) | Taobao (unrelated, held out) |
|---|---|---|---|---|---|---|
| **L1** A count reconstructs the label | Q4 | exact · **flagged** · 1.00 vs 0.61–0.62 | absent · **correctly not flagged** | approx. · **partly** · 0.91 vs 0.72–0.73 | approx. · **partly** · 0.91–0.92 vs 0.74–0.75 | approx. · **partly** · 0.96 vs 0.77–0.78 |
| **L2** Completed-session timing | Q5, Q6 | **flagged** · 0.995 vs 0.61–0.62 | **flagged** · 0.76–0.78 vs 0.58–0.62 | **flagged** · 0.96 vs 0.72–0.73 | **flagged** · 0.965 vs 0.74–0.75 | **flagged** · 0.93 vs 0.77–0.78 |
| **L3** Label mixes purchase and abandonment | Q3 | **flagged** · 11.3% | **flagged** · 3 of 320 | **flagged** · 3.9% | **flagged** · 3.8% | **flagged** (no remove event) |
| **L4** History includes the scored purchase | — | **missed** · 0.97–0.98 vs 0.68–0.71 | **missed** · 0.93–0.94 vs 0.59–0.61 | **missed** · 0.89–0.90 vs 0.56 | **missed** · 0.87–0.89 vs 0.56–0.57 | **missed** · 0.74–0.76 vs 0.57–0.58 |
| **L5** History includes later purchases | — | weak · **missed** · +0.005 | **missed** · +0.080 | **missed** · +0.053 | **missed** · +0.079 | weak · **missed** · +0.014 |
| **L6** Checkout steps after the prediction moment | — | not logged | **missed** · 0.83–0.87 vs 0.58–0.62 | not logged | not logged | not logged |
| **L7** Automated traffic past the bot filter | — | absent | absent | **missed** · 0.59–0.66 vs 0.50–0.54 | absent | absent |

L5 is labelled "weak" when its mean gain over the fair history is under +0.03.

## What it shows

- **Checklist v1 catches the leaks it was written for on every dataset, including the two opened after the rules were fixed.** L2 and L3 are flagged on 5 of 5. L1 is flagged where it holds exactly, correctly not flagged where it doesn't hold, and only partly caught where it holds approximately (3 datasets).
- **It misses every leak outside its scope**, including the largest one: L4 (self-inclusion) inflates every dataset (0.74–0.98) with no question pointing at it.
- **Count over the 26 cases where a leak is present:** flagged 11, partly flagged 3, missed 12.
- **False positives:** none here. The negative control adds 14 correctly built configurations on all five datasets with 0 false alarms (`negative_control.md`).

## Burst check (L7 / proposed Q12)

Rule in `0-Shared/burst_check.py`; thresholds committed before November and Taobao were opened.

| | Flagged days | Purchase rate, flagged vs other | Night share (00–06), flagged vs other | Verdict |
|---|---|---|---|---|
| Retail Rocket | 1 | 31% vs 27% | 27% vs 29% | busy day, not automated |
| Flower shop | 6 (occasion days) | 30% vs 35% | 5% vs 6% | busy days, not automated |
| REES46 Oct | 5 (Oct 1, 2, 6, 7, 8) | **6% vs 16%** | **20% vs 10%** | **automated** |
| REES46 Nov | 2 (Nov 22, 28) | 18.5% vs 17.1% | 8.0% vs 9.5% | sale days, not automated |
| Taobao (China time) | 0 of 9 | — | 6.2% | nothing flagged |

## Proposed checklist v2 and the tests on data opened afterwards

| # | Question | Covers | REES46 Nov (same shop) | Taobao (unrelated shop) |
|---|---|---|---|---|
| Q4′ | Can the label event's count be reconstructed from other counts, **exactly or approximately**? | L1 near-identities | reappears | reappears (purchases + favourites; 0.96 vs 0.77–0.78) |
| Q10 | Does any feature record a step of the process that produces the label **after the prediction moment**? | L6 | not testable | not testable (no checkout steps) |
| Q11 | Is any aggregate computed over a window that **includes the scored case or anything after the prediction moment**? | L4, L5 | both reappear | L4 reappears (0.74–0.76 vs 0.57–0.58); L5 weak |
| Q12 | Do daily volumes and the hour-of-day profile show **traffic the bot filter can't see**? | L7 | correct negative on sale days | correct negative |
| Q13 | Did the **logging schema change** during the data period? | flower shop | not measured | not measured |

**Still open:** Q10 and Q13 rest on the flower shop alone. Q12 has one positive case (REES46 October) and four negatives.

## Caveats
- Flower-shop effects rest on 828 sessions (wide intervals; `robustness.md`).
- Taobao uses a fixed 10% user sample (users with ID divisible by 10); each sampled user's full history is kept.
- Retail Rocket's L4/L5 values come from the Oct 2 probe; its L4 gap is re-estimated with an interval in `robustness.md`.
