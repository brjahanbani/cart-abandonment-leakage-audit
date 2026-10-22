# Paper 2 · Related work (verified 2026-10-21)

Every entry was checked online; "in paper 1" means the BibTeX entry is already in
`paper/camera-ready/references_revised.bib`. New entries are in `references_paper2.bib`.

## 1. Leakage and evaluation integrity

| Reference | Key | What Paper 2 uses it for |
|---|---|---|
| Kaufman et al., *Leakage in data mining*, TKDD 2012 | `kaufman2012` (in paper 1) | Definition of leakage |
| Kapoor & Narayanan, *Leakage and the reproducibility crisis in ML-based science*, Patterns 2023 | `kapoor2023` (in paper 1) | Leakage taxonomy; leakage is widespread |
| Kapoor et al., *REFORMS*, Science Advances 10(18) eadk3452, 2024 | `reforms2024` (in paper 1) | A general, field-agnostic **32-item checklist**, built by expert consensus but not validated per domain |
| Yang, Brower-Sinning, Lewis, Kästner, *Data Leakage in Notebooks: Static Detection and Better Processes*, ASE 2022 | `yang2022` (new) | **Automated, code-level** leakage detection (over 100,000 notebooks); it catches preprocessing and split leaks, not semantic ones like a post-decision timestamp |
| Sculley et al., *Hidden technical debt in ML systems*, NeurIPS 2015 | `sculley2015` (in paper 1) | Undeclared data dependencies |

## 2. Leakage in recommender and clickstream evaluation

| Reference | Key | Use |
|---|---|---|
| Ji, Sun, Zhang, Li, *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, ACM TOIS 41(3), Art. 75, 2023 | `ji2023` (in paper 1) | **Split-level** temporal leakage (ignoring the global timeline) |
| Ferrari Dacrema et al., RecSys 2019 | `dacrema2019` (in paper 1) | Reported gains vanish under proper evaluation |
| Zhang, Peng, Wang, Liu, Wang, *Benchmark Leakage Trap: Can We Trust LLM-based Recommendation?*, arXiv 2602.13626, 2026 | `zhang2026` (new) | Leakage through **pre-training contamination** in LLM recommenders; a different mechanism from ours, cited to delimit scope |

## 3. Point-in-time correctness (feature stores)

| Reference | Key | Use |
|---|---|---|
| de la Rúa Martínez et al. (… Dowling), *The Hopsworks Feature Store for Machine Learning*, SIGMOD Companion 2024, doi 10.1145/3626246.3653389 | `hopsworks2024` (new) | Systems support for point-in-time queries. Paper 2 measures **what each violation costs** when such support is missing (L4/L5) |

## 4. Bots in clickstream data

| Reference | Key | Use |
|---|---|---|
| Kohavi, Mason, Parekh, Zheng, *Lessons and Challenges from Mining Retail E-Commerce Data*, Machine Learning 57:83–113, 2004, doi 10.1023/B:MACH.0000035473.11134.83 | `kohavi2004` (new) | Bot detection is a long-known clickstream problem. Paper 2 shows *rotating-identity* traffic that per-visitor rules miss (L7) |

## 5. Cart abandonment and purchase intent
Paper 1's set: `sakar2019`, `requena2020`, `rausch2022`, `esmeli2021`, `esmeli2025`, `waldmann2025`, `suleiman2026`, `alzghoul2026`, `fabra2020`.

## 6. Datasets
| Dataset | Key | Citation requirement |
|---|---|---|
| Retail Rocket | `retailrocket2015` (in paper 1) | as in paper 1 |
| REES46 cosmetics shop | `rees46` (new) | Link to the Kaggle page **and** to https://rees46.com (stated on the dataset page) |
| Taobao User Behavior | `taobao2017` (dataset) + `zhu2018` (paper) | Cite the paper that introduced the data: Zhu, Li, Zhang, Li, He, Li, Gai, *Learning Tree-based Deep Model for Recommender Systems*, KDD 2018, pp. 1079–1088, doi 10.1145/3219819.3219826 (verified on Crossref 2026-10-21; its UserBehavior footnote links Tianchi dataset 649). The Kaggle mirror requests no citation; the Tianchi page is rendered by script, so its own citation field could not be read. Also thank Alibaba in the Acknowledgments (done) |

## Positioning (draft for the paper)

Prior work offers **general checklists** (REFORMS: 32 items, expert consensus), **code-level
detectors** (Yang et al. 2022) and **split-level** analyses of temporal leakage in
recommenders (Ji et al. 2023). Feature stores provide the **mechanism** for point-in-time
correctness (Hopsworks 2024). None of them tests whether a **domain-specific leakage
checklist transfers to independent datasets**, measures which leaks it misses and what they
cost, or checks its false-alarm behaviour. Paper 2 does this for cart-abandonment prediction:
five datasets from four shops (one private), a detection matrix with 26 positive cases and
14 negative-control configurations, and five proposed questions, three of them tested on a
held-out month and an unrelated shop.
