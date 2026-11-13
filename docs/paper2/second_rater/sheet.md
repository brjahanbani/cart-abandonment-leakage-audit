# Second-rater sheet: does the checklist flag it?

Thank you for helping. This takes about 45 minutes and needs no data or code.

Your answers are compared with the first author's to measure how far two people agree when
they apply the checklist. There are no right or wrong answers: a disagreement is a result, not
a mistake.

## Rules

1. **Work alone**, ideally in one sitting. Do not discuss the items with anyone, including the
   authors, until you have returned the sheet.
2. **Do not use AI tools** (ChatGPT, Gemini, Claude or similar) or search for the answers.
3. **Do not read** the Paper 2 manuscript or the project's code repository before you finish.
   Reading Paper 1, where the checklist was published, is allowed.
4. **Do not ask** the authors what an item means. If an item is unclear, choose the rating that
   fits best and write your doubt in the Note column.
5. Rate every item. You may change an answer while you work; please do not change anything
   after you have returned the sheet.

## The task the models are meant for

At the moment a visitor **first adds an item to the cart**, predict whether the session will
end in a purchase. A shop would act at that moment (a reminder, an offer, help). Sessions end
after 30 minutes without activity.

## The nine questions

- Q1. What exact event or time defines the prediction moment?
- Q2. Which event creates the label?
- Q3. Does the label describe the outcome of the specific decision, or only of the session that contains it?
- Q4. Can that event count be reconstructed from other counts?
- Q5. Does any duration, recency or average use the final session event?
- Q6. Is the endpoint selected differently for positive and negative cases?
- Q7. Are clipping, scaling, feature selection and imputation fitted using training data only?
- Q8. Does performance remain plausible after suspicious feature groups are removed?
- Q9. Can the experiment be rerun from code and preserved outputs?

## How to rate each item

Answer from the description and, for Q8, from the score given with it. Each item that describes features ends with the ROC-AUC the pipeline reports for it (XGBoost, tested on visitors not seen in training, mean of five runs; 0.5 is chance, 1.0 is perfect). Items about the label have no score. Judge whether one of the nine questions, **read as written**, points at a problem in the part of the pipeline the item describes. Do not judge whether you personally see a problem.

| Rating | Meaning |
|---|---|
| **F** | Flagged: a question, read as written, identifies the problem. Name the question(s). |
| **P** | Partly: only an extended or generous reading of a question would identify it. Name the question(s). |
| **N** | Not flagged: no question identifies a problem (whether or not one exists). |

In every item the preprocessing (Q7) is fitted on training data only and the code is rerunnable
(Q9), so those two questions never apply. Unless an item says otherwise, the label is described
in the item about the label.

---

## Dataset A: Retail Rocket (public)

Events logged: view, add-to-cart, transaction (purchase). Timestamps in milliseconds.
Cleaning: per-visitor bot rules (over 500 events per hour, or very regular timing).

| Item | Description | Rating | Question(s) | Note |
|---|---|---|---|---|
| A1 | First add-to-cart table: 13 features computed from the events up to and including the first add-to-cart (counts, ratio, times since earlier events, mean interval, hour, day of week, repeated views of carted items). Reported ROC-AUC: 0.62. | | | |
| A2 | Completed-session table: event count, view count and add-to-cart count of the whole session. Label: the session contains a transaction. Reported ROC-AUC: 1.00. | | | |
| A3 | The label used by both tables: "the session contains a purchase" (completed table) or "a purchase later in the session" (first add-to-cart table). It is not tied to the carted item. | | | |
| A4 | The A1 features plus two history features joined at the first add-to-cart: the carted item's smoothed purchase rate and the visitor's purchase count, both counted over the whole data period, including the scored session. Reported ROC-AUC: 0.97. | | | |
| A5 | Completed-session table: session age (first to last event), mean interval between events, and time from the last add-to-cart to the session's last event, all over the whole session. Reported ROC-AUC: 0.87. | | | |
| A6 | The A1 features plus the same two history features counted strictly before the first add-to-cart. Reported ROC-AUC: 0.66. | | | |
| A7 | The A1 features plus the same two history features counted over the whole data period, excluding the scored session (later sessions included). Reported ROC-AUC: 0.68. | | | |

## Dataset B: flower shop (private)

Events logged: view, category view, search, add-to-cart, cart removal, and checkout steps
(cart, address, payment, final). The label event is the final checkout step (payment
succeeded). Timestamps in milliseconds. Cleaning: per-visitor bot rules, crawlers removed by
user agent, repeated front-end events within two seconds collapsed.

| Item | Description | Rating | Question(s) | Note |
|---|---|---|---|---|
| B1 | Completed-session table: event count, view count and add-to-cart count of the whole session. Label: the session contains a final checkout step. Reported ROC-AUC: 0.71. | | | |
| B2 | The 13 first add-to-cart features (as A1) plus facts known at that moment: whether the first carted item is a same-day or a scheduled-delivery product, its delivery lead time and price, the number of category views and searches so far, and whether the device is mobile. Reported ROC-AUC: 0.64. | | | |
| B3 | Completed-session table: session age, mean interval between events, and time from the last add-to-cart to the session's last event, all over the whole session. Reported ROC-AUC: 0.69. | | | |
| B4 | The 13 first add-to-cart features plus the two history features (as A4) counted over the whole data period, including the scored session. Reported ROC-AUC: 0.94. | | | |
| B5 | The 13 first add-to-cart features only. Reported ROC-AUC: 0.57. | | | |
| B6 | Completed-session counts (as B1) plus two checkout features of the whole session: the number of checkout-step events (cart, address, payment) and whether the payment step was reached. Reported ROC-AUC: 0.87. | | | |
| B7 | The label: as A3. | | | |
| B8 | The 13 first add-to-cart features plus the two history features counted over the whole period, excluding the scored session. Reported ROC-AUC: 0.69. | | | |
| B9 | The 13 first add-to-cart features plus the two history features counted strictly before the first add-to-cart. Reported ROC-AUC: 0.60. | | | |

## Dataset C: REES46 cosmetics shop, October 2019 (public)

Events logged: view, cart, remove_from_cart, purchase. Timestamps in whole seconds. Cleaning:
exact duplicate rows dropped; per-visitor bot rules. The models use all days of October. No
other traffic check was run (daily volumes and hour-of-day profiles were not examined).

| Item | Description | Rating | Question(s) | Note |
|---|---|---|---|---|
| C1 | The 13 first add-to-cart features plus the two history features counted strictly before the first add-to-cart. Reported ROC-AUC: 0.54. | | | |
| C2 | Completed-session table: session age, mean interval, and time from the last add-to-cart to the session's last event, over the whole session. Reported ROC-AUC: 0.86. | | | |
| C3 | The 13 first add-to-cart features, trained and tested on all days of October as cleaned above. Reported ROC-AUC: 0.65. | | | |
| C4 | Completed-session table: event count, view count and add-to-cart count of the whole session. Label: the session contains a purchase. Reported ROC-AUC: 0.91. | | | |
| C5 | The 13 first add-to-cart features plus the two history features over the whole month, including the scored session. Reported ROC-AUC: 0.89. | | | |
| C6 | The label: as A3. | | | |
| C7 | The 13 first add-to-cart features plus facts known at that moment: the first carted item's price and whether it has a brand, and the numbers of categories viewed and of cart removals so far. Reported ROC-AUC: 0.56. | | | |
| C8 | The 13 first add-to-cart features plus the two history features over the whole month, excluding the scored session. Reported ROC-AUC: 0.62. | | | |
| C9 | The 13 first add-to-cart features only, on the days from 9 October onward. Reported ROC-AUC: 0.53. | | | |

## Dataset D: REES46 cosmetics shop, November 2019 (public)

Same shop, events and cleaning as dataset C; all days of November are used.

| Item | Description | Rating | Question(s) | Note |
|---|---|---|---|---|
| D1 | The label: as A3. | | | |
| D2 | The 13 first add-to-cart features plus the two history features over the whole month, excluding the scored session. Reported ROC-AUC: 0.65. | | | |
| D3 | The 13 first add-to-cart features only. Reported ROC-AUC: 0.54. | | | |
| D4 | Completed-session table: event count, view count and add-to-cart count of the whole session. Label: the session contains a purchase. Reported ROC-AUC: 0.91. | | | |
| D5 | The 13 first add-to-cart features plus product facts (as C7). Reported ROC-AUC: 0.56. | | | |
| D6 | The 13 first add-to-cart features plus the two history features over the whole month, including the scored session. Reported ROC-AUC: 0.87. | | | |
| D7 | Completed-session table: session age, mean interval, and time from the last add-to-cart to the session's last event, over the whole session. Reported ROC-AUC: 0.87. | | | |
| D8 | The 13 first add-to-cart features plus the two history features counted strictly before the first add-to-cart. Reported ROC-AUC: 0.54. | | | |

## Dataset E: Taobao marketplace (public)

Events logged: page view, cart, favourite, buy. Timestamps in whole seconds. A fixed 10% sample
of users with each user's full history. Cleaning: per-visitor bot rules; rows with impossible
dates dropped.

| Item | Description | Rating | Question(s) | Note |
|---|---|---|---|---|
| E1 | Completed-session table: session age, mean interval, and time from the last add-to-cart to the session's last event, over the whole session. Reported ROC-AUC: 0.74. | | | |
| E2 | The 13 first add-to-cart features plus the two history features counted strictly before the first add-to-cart. Reported ROC-AUC: 0.55. | | | |
| E3 | The 13 first add-to-cart features plus the two history features over the whole period, including the scored session. Reported ROC-AUC: 0.74. | | | |
| E4 | The label: as A3. | | | |
| E5 | The 13 first add-to-cart features plus product facts known at that moment: categories and favourites seen so far. Reported ROC-AUC: 0.54. | | | |
| E6 | Completed-session table: event count, view count and add-to-cart count of the whole session. Label: the session contains a purchase. Reported ROC-AUC: 0.96. | | | |
| E7 | The 13 first add-to-cart features only. Reported ROC-AUC: 0.53. | | | |
| E8 | The 13 first add-to-cart features plus the two history features over the whole period, excluding the scored session. Reported ROC-AUC: 0.59. | | | |

---

**Rater:** ____________________  **Date:** ____________  **Time taken (minutes):** ______

**Background** (one line, for example "PhD student, 3 years of machine learning"): ____________________

**Read Paper 1 before rating?** yes / no  **Read any part of Paper 2?** yes / no

**Discussed any item with anyone, or used an AI tool?** yes / no

Please return the filled sheet to the first author.
