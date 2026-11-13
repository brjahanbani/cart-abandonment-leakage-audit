"""
taobao_sample_check.py
======================

Paper 2 analyses Taobao on a fixed 10% user sample (user id divisible by 10). Is the
sample representative of the full file? Compares, on the raw rows inside the data period
(25 Nov - 3 Dec 2017, China time), all users with the sampled users:

  users, rows, rows per user, the share of each behaviour (pv, cart, fav, buy),
  the share of users who ever add to cart and who ever buy, and buys per add-to-cart.

Output: 5-Results/paper2__taobao-sample-check.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "1-Data" / "taobao" / "raw" / "UserBehavior.csv"
T_FROM, T_TO = 1511539200, 1512316800  # 2017-11-25 00:00 to 2017-12-04 00:00, UTC+8

per_user = []  # per chunk: rows, carts and buys of each user
for chunk in pd.read_csv(RAW, header=None, names=["user", "item", "cat", "behavior", "ts"],
                         usecols=["user", "behavior", "ts"],
                         dtype={"user": "int64", "behavior": "category", "ts": "int64"},
                         chunksize=5_000_000):
    chunk = chunk[(chunk["ts"] >= T_FROM) & (chunk["ts"] < T_TO)]
    b = chunk["behavior"].astype(str)
    per_user.append(pd.DataFrame({"user": chunk["user"].values, "rows": 1,
                                  **{k: (b == k).values.astype("int32")
                                     for k in ("pv", "cart", "fav", "buy")}})
                    .groupby("user").sum())
u_all = pd.concat(per_user).groupby(level=0).sum()

rows = []
for part, u in (("all", u_all), ("sample", u_all[u_all.index % 10 == 0])):
    c, total = u[["pv", "cart", "fav", "buy"]].sum(), u["rows"].sum()
    u = u.assign(cart=u["cart"] > 0, buy=u["buy"] > 0)
    rows.append({"part": part, "users": len(u), "rows": total,
                 "rows_per_user": total / len(u),
                 **{f"share_{b}": c.get(b, 0) / total for b in ("pv", "cart", "fav", "buy")},
                 "users_ever_cart": u["cart"].mean(), "users_ever_buy": u["buy"].mean(),
                 "buys_per_cart": c.get("buy", 0) / c.get("cart", 1)})
out = pd.DataFrame(rows)
out.round(4).to_csv(ROOT / "5-Results" / "paper2__taobao-sample-check.csv", index=False)
print(out.round(4).T.to_string())
