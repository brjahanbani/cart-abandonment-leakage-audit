# Paper 2 · Negative control: does the checklist stay quiet when there's no leak?

Run 2026-10-20. Script: `9-Paper2/negative_control.py` (one run per dataset).
Outputs: `5-Results/paper2__negative-control__{retail_rocket,flower_shop,rees46_oct,rees46_nov}.csv`.

**Why:** the detection matrix had only one true negative for checklist v1, so false alarms
couldn't be assessed. Here the checklist is applied to feature sets **built correctly**, where
every value is known at the first add-to-cart, on all four datasets.

| Configuration | Content |
|---|---|
| `prefix13` | paper 1's 13 point-in-time features |
| `prefix13_hist` | + item conversion rate and visitor purchase count, counted strictly before the first add |
| `prefix13_product` | + product facts known at the first add (flower shop, REES46) |

14 configurations in total (Retail Rocket 2, flower shop 3, REES46 October 3, REES46 November 3, Taobao 3; Taobao added 2026-10-21, with "product facts" = categories and favourites seen so far).

## 1. The checklist on the correctly built configurations

The questions are answered from how each feature is built, which is the same on every dataset.

| Question | Answer on all 14 configurations | Flag? |
|---|---|---|
| Q4 / Q4′ count reconstructs the label | No purchase event enters any feature; the code asserts that no prefix contains a purchase | no |
| Q5, Q6 duration or recency at a label-dependent endpoint | Every timing ends at the cutoff (the first add), the same rule for both classes | no |
| Q7 preprocessing fitted on train only | Yes (cap and scaler) | no |
| Q8 plausible after removing suspect groups | Scores 0.53–0.66 | no |
| Q10 process steps after the prediction moment | None used | no |
| Q11 aggregate window includes the scored case or later | History counted strictly before the first add | no |
| Q12 traffic the bot filter can't see | REES46 October uses the clean window; the other three datasets have no automated days (detection matrix) | no |
| Q13 logging-schema change | Flower shop: features that need the Aug 23 columns are only used from Aug 23 | no |
| Q3 label of the decision or the session | The label is "a purchase later in the session", not item-specific. The item-level check measures the gap (0.9–3.9% of purchased sessions) | **yes**, a real minor caveat, not a false alarm |
| Q1, Q2, Q9 (definitions, rerunnable) | Answered | — |

**False alarms: 0** across 14 configurations and the 10 leakage questions (Q3–Q8, Q10–Q13). Q3's flag is correct: the label caveat is real and measured (on Taobao it's large: 90.7% of purchased sessions; see `robustness.md`).

## 2. Empirical checks

| Dataset | Config | Real-label AUC | Shuffled-label AUC (range over 5 seeds) | Strongest single feature (AUC) |
|---|---|---|---|---|
| Retail Rocket | prefix13 | 0.622 | 0.478 (0.456–0.497) | seconds since first event (0.60) |
| | prefix13_hist | 0.662 | 0.475 (0.457–0.494) | seconds since first event (0.60) |
| Flower shop | prefix13 | 0.571 | 0.535 (0.473–0.577) | seconds since previous event (0.59) |
| | prefix13_hist | 0.599 | 0.538 (0.516–0.564) | seconds since previous event (0.59) |
| | prefix13_product | 0.642 | 0.506 (0.459–0.551) | lead time in days (0.66) |
| REES46 Oct | prefix13 | 0.533 | 0.498 (0.489–0.505) | repeated views of carted items (0.53) |
| | prefix13_hist | 0.538 | 0.499 (0.491–0.505) | repeated views of carted items (0.53) |
| | prefix13_product | 0.558 | 0.495 (0.487–0.501) | price of the first item (0.55) |
| REES46 Nov | prefix13 | 0.543 | 0.493 (0.491–0.495) | repeated views of carted items (0.53) |
| | prefix13_hist | 0.544 | 0.498 (0.496–0.501) | repeated views of carted items (0.53) |
| | prefix13_product | 0.564 | 0.494 (0.486–0.500) | price of the first item (0.54) |
| Taobao | prefix13 | 0.533 | 0.489 (0.479–0.497) | seconds since first event (0.54) |
| | prefix13_hist | 0.547 | 0.484 (0.476–0.492) | seconds since first event (0.54) |
| | prefix13_product | 0.543 | 0.488 (0.481–0.491) | seconds since first event (0.54) |

Protocol: group A's (visitor-grouped split, cap fitted on train, XGBoost, seeds 11–55). Shuffled labels: the training labels are permuted, and the test is scored on the true labels.

- **With shuffled labels every pipeline drops to chance**, so the split, the preprocessing and the features carry no hidden label information.
- **No single feature is strong** (≤ 0.66), unlike the leaky features in the matrix, where the two history aggregates including the scored purchase alone reach 0.73–0.98 (`*history-window*.csv`, rows `full, agg_only`: Taobao 0.73–0.75 … Retail Rocket 0.97–0.98).
- Retail Rocket's and Taobao's shuffled scores sit slightly below 0.5 (0.475–0.489). The deviation is small and far from any leak level; its cause isn't investigated.

## 3. What it adds to Paper 2

With the matrix, the checklist now has **26 positive cases** (11 flagged, 3 partly, 12 missed) and **14 negative configurations with 0 false alarms**. The false-alarm side is now evidenced, not just assumed.

## Caveats
- The questions were answered by the author, who also built the features. A second, independent rater would strengthen this.
- Flower shop: the product type (same-day vs scheduled) comes from the catalogue as exported, not as of each session. The owner describes product types as stable, but this hasn't been checked against history.
