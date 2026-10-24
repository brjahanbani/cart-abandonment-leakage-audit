"""
second_rater_agreement.py
=========================

Paper 2 item 24: how far does a second, independent rater agree with the
first author when applying the nine-question checklist? The analysis is
fixed in docs/paper2/second_rater/PROTOCOL.md before any rating is seen.

Inputs (private, docs/paper2/second_rater/):
  key.csv              first author's ratings (item, mechanism, F/P/N, questions)
  ratings_<rater>.csv  second rater's sheet, transcribed: item,rating,questions,note
                       (questions separated by ";", e.g. "Q5;Q6")

Reported (all 41 items unless stated):
  agreement            share of items with the same rating (F, P or N)
  cohen_kappa          chance-corrected agreement
  gwet_ac1             chance-corrected agreement that stays stable when one
                       category dominates (27 of 41 key ratings are N)
  *_ci                 95% bootstrap interval over items (2,000 resamples, seed 42)
  binary_agreement     F and P merged into "flagged at all" versus N
  question_agreement   among items both raters flagged (F or P), the share
                       naming at least one common question
  tally_*              each rater's own detection matrix: on the 26 present
                       leaks flagged / partly / missed, and false alarms on
                       the 14 negative controls

Run:
  python 9-Paper2/second_rater_agreement.py <rater>
Output: 5-Results/paper2__second-rater-agreement__<rater>.csv, plus the
disagreeing items printed for the paper's discussion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RATER_DIR = ROOT / "docs" / "paper2" / "second_rater"
CATEGORIES = ["F", "P", "N"]
N_BOOT = 2000


def cohen_kappa(a, b):
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in CATEGORIES)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def gwet_ac1(a, b):
    po = np.mean(a == b)
    pi = [(np.mean(a == k) + np.mean(b == k)) / 2 for k in CATEGORIES]
    pe = sum(p * (1 - p) for p in pi) / (len(CATEGORIES) - 1)
    return (po - pe) / (1 - pe)


def bootstrap_ci(stat, a, b):
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    values = [stat(a[i], b[i]) for i in idx]
    return np.percentile(values, [2.5, 97.5])


def tally(df, col):
    present = df[~df["mechanism"].str.startswith("negative") & ~df["mechanism"].str.contains("absent")]
    negative = df[df["mechanism"].str.startswith("negative")]
    return {
        "present": len(present),
        "flagged": int((present[col] == "F").sum()),
        "partly": int((present[col] == "P").sum()),
        "missed": int((present[col] == "N").sum()),
        "negatives": len(negative),
        "false_alarms": int((negative[col] != "N").sum()),
    }


def main(rater):
    key = pd.read_csv(RATER_DIR / "key.csv", dtype=str).fillna("")
    second = pd.read_csv(RATER_DIR / f"ratings_{rater}.csv", dtype=str).fillna("")
    second["rating"] = second["rating"].str.strip().str.upper()
    df = key.merge(second[["item", "rating", "questions"]], on="item", how="left",
                   suffixes=("_1", "_2"))
    missing = df.loc[df["rating"].isna() | ~df["rating"].isin(CATEGORIES), "item"].tolist()
    if missing:
        sys.exit(f"Missing or invalid ratings for: {', '.join(missing)}")

    a = df["first_author_rating"].to_numpy()
    b = df["rating"].to_numpy()
    both = df[(df["first_author_rating"] != "N") & (df["rating"] != "N")]
    shared = [bool(set(q1.split(";")) & set(q2.replace(" ", "").split(";")))
              for q1, q2 in zip(both["questions_1"], both["questions_2"])]

    row = {
        "rater": rater,
        "items": len(df),
        "agreement": round(float(np.mean(a == b)), 3),
        "cohen_kappa": round(cohen_kappa(a, b), 3),
        "cohen_kappa_ci": "[{:.3f}, {:.3f}]".format(*bootstrap_ci(cohen_kappa, a, b)),
        "gwet_ac1": round(gwet_ac1(a, b), 3),
        "gwet_ac1_ci": "[{:.3f}, {:.3f}]".format(*bootstrap_ci(gwet_ac1, a, b)),
        "binary_agreement": round(float(np.mean((a == "N") == (b == "N"))), 3),
        "question_agreement": round(float(np.mean(shared)), 3) if shared else np.nan,
    }
    for who, col in [("first", "first_author_rating"), ("second", "rating")]:
        for k, v in tally(df, col).items():
            row[f"tally_{who}_{k}"] = v

    out = ROOT / "5-Results" / f"paper2__second-rater-agreement__{rater}.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(pd.Series(row).to_string())
    print("\nDisagreements:")
    diff = df[a != b][["item", "mechanism", "first_author_rating", "questions_1", "rating", "questions_2"]]
    print(diff.to_string(index=False) if len(diff) else "none")
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python 9-Paper2/second_rater_agreement.py <rater>")
    main(sys.argv[1])
