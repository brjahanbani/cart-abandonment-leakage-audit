# Paper 2 · What was fixed before the late datasets were opened

REES46 November and Taobao were opened only after the rules below had been committed. The
commits are in the authors' private development repository, whose history also contains
material about the private flower-shop data and is therefore not published. The hashes and
times below identify them; reviewers can inspect the full history on request.

| Commit | Time (UTC+3) | What it fixed |
|---|---|---|
| `9ef426524ead92cad5cb0198e981ce8ad9be18ed` | 2026-10-20 23:02:21 | Cleaning rules (shared with the original audit); the checklist answers for each leak mechanism, recorded in `9-Paper2/build_detection_matrix.py`; the burst rule and its two thresholds (1.5 × median day, purchase rate below 60% of other days), run on Retail Rocket, the flower shop and REES46 October. At this point the matrix covered three datasets (7 of 16 present cases flagged, 1 partly, 8 missed). |
| `fafeaf1901cb5a0c04135ef82a373002e3ff147f` | 2026-10-20 23:30:31 | First results on REES46 November, using the rules above unchanged. |
| `62ce47b534b5a84decaa3b321a60e0951d3020bd` | 2026-10-21 00:38:22 | Taobao profiling script (event counts and dates only; no models). |
| `6a2a65c2437dfa508c0acfe9101e5adaeb0977d7` | 2026-10-21 00:57:05 | Taobao cleaning, and the leak-size prediction, committed before any Taobao model was trained (quoted below). |
| `c4b2c03` | 2026-10-21 01:27:47 | First Taobao model results (group A, group B, intervals, negative control, matrix). |

## The prediction, as committed in `6a2a65c`

> Prediction, committed before Taobao's group A result exists: the completed-session leak
> tracks the gap between abandoned and purchased visits ending on an add-to-cart (RR 58.5
> points -> +0.40, REES46 46.3 -> +0.24, flower shop 0.7 -> +0.12). Taobao's gap is 21.2
> points (36.4% vs 15.2%), so its completed-session leak should be smaller than REES46's and
> larger than the flower shop's.

Outcome (paper, Section 3.2): Taobao's leak is +0.155 [0.150, 0.160], below REES46's +0.243. The
second half, above the flower shop's +0.122 [0.021, 0.228], cannot be resolved because the
flower shop's interval contains Taobao's value.
