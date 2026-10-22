"""
burst_check.py
==============

Proposed checklist question Q12: does the data contain traffic the bot
filter cannot see? Found on REES46 October 2019 (identity-rotating automated
traffic); the rule and its two thresholds were fixed and committed
(commit 9ef4265, 9-Paper2/build_detection_matrix.py) before any other month
was opened, so applying them unchanged to new data is a fair test.

Rule, on cart sessions:
  flagged day  more than BURST_FACTOR x the median daily number of cart
               sessions (by session start)
  present      the pooled purchase rate on flagged days is below
               PURCHASE_RATE_RATIO x the pooled rate on the other days
When present, every flagged day is treated as automated.
"""

import pandas as pd

BURST_FACTOR = 1.5
PURCHASE_RATE_RATIO = 0.6


def check(start_ms, bought):
    """start_ms: session start in ms; bought: bool per session (same index).
    Returns a dict summary and the list of flagged dates."""
    t = pd.to_datetime(pd.Series(start_ms), unit="ms")
    s = pd.DataFrame({"day": t.dt.date.values, "night": (t.dt.hour < 6).values,
                      "buy": pd.Series(bought).values})
    daily = s.groupby("day").size()
    flagged = list(daily.index[daily > BURST_FACTOR * daily.median()])
    f = s["day"].isin(flagged)
    rate_f = float(s.loc[f, "buy"].mean()) if f.any() else None
    rate_o = float(s.loc[~f, "buy"].mean())
    summary = {
        "days": len(daily), "median_cart_sessions_per_day": float(daily.median()),
        "flagged_days": len(flagged), "flagged_dates": ", ".join(map(str, flagged)),
        "purchase_rate_flagged": round(rate_f, 4) if rate_f is not None else None,
        "purchase_rate_other": round(rate_o, 4),
        "night_share_flagged": round(float(s.loc[f, "night"].mean()), 4) if f.any() else None,
        "night_share_other": round(float(s.loc[~f, "night"].mean()), 4),
        "automated": bool(flagged) and rate_f < PURCHASE_RATE_RATIO * rate_o,
    }
    return summary, flagged
