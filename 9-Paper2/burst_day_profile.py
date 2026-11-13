"""
burst_day_profile.py
====================

Day-level profile of REES46 October 2019 behind the suspected automated
traffic of Paper 2, Section "Suspected Automated Traffic that Evades
Per-Visitor Bot Rules". The burst rule itself (flagged day, verdict) is in
0-Shared/burst_check.py and its result in paper2__burst-check.csv; this file
only describes the days, so that every number in that paragraph traces to a
result file:

  cart_sessions        cart sessions starting that day
  ratio_to_median      cart_sessions / median day
  flagged              above 1.5 x the median day (the committed rule)
  distinct_visitors    distinct visitor ids among those sessions
  sessions_per_visitor cart_sessions / distinct_visitors
  median_session_s     median session length (last minus first event), s
  night_share          sessions starting 00:00-05:59
  hourly_cv            coefficient of variation of the 24 hourly session
                       counts (low = flat around the clock)
  purchase_rate        share of the day's cart sessions with a purchase

Input: the cleaned events (1-Data/rees46/clean/events_clean.parquet), i.e.
the whole of October, before the clean window from 9 October is applied.

Output: 5-Results/paper2__burst-days-profile.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "0-Shared"))
import burst_check  # noqa: E402  (BURST_FACTOR, the committed threshold)

D = ROOT / "1-Data"
R = ROOT / "5-Results"

ev = pd.read_parquet(D / "rees46" / "clean" / "events_clean.parquet",
                     columns=["timestamp", "visitorid", "event", "session_key"])
ev["event"] = ev["event"].astype(str)
key = ev["session_key"]
g = ev.groupby(key)
s = pd.DataFrame({
    "start": g["timestamp"].min(),
    "end": g["timestamp"].max(),
    "visitor": g["visitorid"].first(),
    "cart": ev["event"].eq("addtocart").groupby(key).any(),
    "buy": ev["event"].eq("transaction").groupby(key).any(),
})
s = s[s["cart"]].copy()
t = pd.to_datetime(s["start"], unit="ms")
s["day"] = t.dt.date
s["hour"] = t.dt.hour
s["length_s"] = (s["end"] - s["start"]) / 1000

median_day = s.groupby("day").size().median()
rows = []
for day, d in s.groupby("day"):
    hourly = d.groupby("hour").size().reindex(range(24), fill_value=0)
    n = len(d)
    rows.append({
        "day": str(day),
        "cart_sessions": n,
        "ratio_to_median": round(n / median_day, 2),
        "flagged": bool(n > burst_check.BURST_FACTOR * median_day),
        "distinct_visitors": int(d["visitor"].nunique()),
        "sessions_per_visitor": round(n / d["visitor"].nunique(), 3),
        "median_session_s": round(float(d["length_s"].median()), 1),
        "night_share": round(float((d["hour"] < 6).mean()), 4),
        "hourly_cv": round(float(hourly.std(ddof=0) / hourly.mean()), 3),
        "purchase_rate": round(float(d["buy"].mean()), 4),
    })

out = pd.DataFrame(rows)
out.to_csv(R / "paper2__burst-days-profile.csv", index=False)
print(f"median day: {median_day:.0f} cart sessions")
print(out.to_string(index=False))
