"""
second_rater_decomposition.py
=============================

Paper 2 item 24, POST-HOC (not in PROTOCOL.md §3; written 2026-11-05, after
both sheets and the fixed analysis were seen). Descriptive only: it does not
re-score any sheet and does not change the primary result. It asks where the
disagreement between the first author and each rater comes from.

Steps, applied cumulatively to a rater's answers as returned:
  S0  as returned (reproduces second_rater_agreement.py)
  S1  the paper's strict scoring rule, which the sheet did not give: Q1 (the
      prediction moment) and Q8 (plausibility of the score) name no column and
      do not count as flagging; a rating left with no other question becomes N
  S2  S1, without the four L3 items printed only as "The label: as A3."
      (B7, C6, D1, E4), a property of the sheet rather than of the questions
Also reported, separately: P read as N ("partly" counted as not flagged).

Also writes the by-mechanism counts (F/P/N per mechanism group and reader)
quoted in docs/paper2/second_rater/RESULTS.md.

Run:  python 9-Paper2/second_rater_decomposition.py A B
Output: 5-Results/paper2__second-rater-decomposition.csv
        5-Results/paper2__second-rater-by-mechanism.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from second_rater_agreement import (CATEGORIES, RATER_DIR, ROOT, bootstrap_ci,  # noqa: E402
                                    cohen_kappa, gwet_ac1, load_ratings, tally)

POINTER_ITEMS = {"B7", "C6", "D1", "E4"}
NOT_FLAGGING = {"Q1", "Q8"}


def strict(rating, questions):
    if rating == "N":
        return "N"
    left = [q for q in questions.replace(" ", "").split(";") if q and q not in NOT_FLAGGING]
    return rating if left else "N"


def group(mechanism):
    if mechanism.startswith("negative"):
        return "negative controls"
    return mechanism.split(" (")[0]


def measures(df, step, rater):
    a = df["first_author_rating"].to_numpy()
    b = df["r"].to_numpy()
    row = {"rater": rater, "step": step, "items": len(df),
           "agreement": round(float(np.mean(a == b)), 3),
           "cohen_kappa": round(cohen_kappa(a, b), 3),
           "cohen_kappa_ci": "[{:.3f}, {:.3f}]".format(*bootstrap_ci(cohen_kappa, a, b)),
           "gwet_ac1": round(gwet_ac1(a, b), 3),
           "gwet_ac1_ci": "[{:.3f}, {:.3f}]".format(*bootstrap_ci(gwet_ac1, a, b)),
           "binary_agreement": round(float(np.mean((a == "N") == (b == "N"))), 3)}
    t = tally(df.assign(rating=df["r"]), "rating")
    row.update({f"tally_{k}": v for k, v in t.items()})
    row["disagreeing_items"] = " ".join(df.loc[a != b, "item"])
    return row


def main(raters):
    key = pd.read_csv(RATER_DIR / "key.csv", dtype=str).fillna("")
    rows, mech = [], key.assign(group=key["mechanism"].map(group))[["item", "group", "first_author_rating"]]
    for rater in raters:
        df = key.merge(load_ratings(rater), on="item", how="left", suffixes=("_key", ""))
        assert df["rating"].isin(CATEGORIES).all(), rater
        mech[rater] = df["rating"].to_numpy()
        df["r"] = df["rating"]
        rows.append(measures(df, "S0 as returned", rater))
        df["r"] = [strict(r, q) for r, q in zip(df["rating"], df["questions"])]
        rows.append(measures(df, "S1 strict rule (Q1, Q8 do not flag)", rater))
        rows.append(measures(df[~df["item"].isin(POINTER_ITEMS)], "S2 S1 without pointer items", rater))
        df["r"] = df["rating"].replace({"P": "N"})
        rows.append(measures(df, "P read as N (separate)", rater))

    out = pd.DataFrame(rows)
    path = ROOT / "5-Results" / "paper2__second-rater-decomposition.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.drop(columns=["tally_present", "tally_negatives"]).to_string(index=False))

    order = list(dict.fromkeys(mech["group"]))
    counts = []
    for g in order:
        sub = mech[mech["group"] == g]
        row = {"group": g, "n": len(sub)}
        for col in ["first_author_rating"] + list(raters):
            row[col.replace("_rating", "")] = "/".join(str(int((sub[col] == k).sum())) for k in CATEGORIES)
        counts.append(row)
    counts = pd.DataFrame(counts)
    path2 = ROOT / "5-Results" / "paper2__second-rater-by-mechanism.csv"
    counts.to_csv(path2, index=False)
    print("\nBy mechanism (F/P/N):")
    print(counts.to_string(index=False))
    print(f"\nWrote {path.relative_to(ROOT)} and {path2.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python 9-Paper2/second_rater_decomposition.py <rater> [<rater> ...]")
    main(sys.argv[1:])
