"""
run_all.py
==========

Reproduces every Paper 2 result in order, one command:

    python 9-Paper2/run_all.py               # all steps
    python 9-Paper2/run_all.py --from 7      # resume at step 7
    python 9-Paper2/run_all.py --list        # show the steps

Each step is the script that produced a committed result file; the runner
stops at the first failure. Inputs it needs (see docs/paper2/REPRODUCE.md):
the Retail Rocket artifacts of paper 1, the private flower-shop exports in
1-Data/own/raw/, REES46 2019-Oct.csv / 2019-Nov.csv in 1-Data/rees46/raw/, and
Taobao UserBehavior.csv in 1-Data/taobao/raw/. A full run takes about an hour.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    ("flower shop: clean + sessions", "7-Own-Data/build_own_events.py", []),
    ("flower shop: orders, products, labels", "7-Own-Data/build_own_tables.py", []),
    ("flower shop: group A (paper 1 repeated)", "7-Own-Data/replicate_paper1.py", []),
    ("flower shop: group B checkout steps + product facts", "7-Own-Data/extras_funnel_product.py", []),
    ("flower shop: group B history window", "7-Own-Data/history_window_test.py", []),
    ("REES46 Oct: clean + sessions", "8-REES46/build_rees46_events.py", []),
    ("REES46 Nov: clean + sessions", "8-REES46/build_rees46_events.py", ["--month", "2019-Nov"]),
    ("REES46 Oct: group A, all October", "8-REES46/replicate_paper1_rees46.py", []),
    ("REES46 Oct: group A, clean window", "8-REES46/replicate_paper1_rees46.py", ["--from", "2019-10-09"]),
    ("REES46 Nov: group A (held out)", "8-REES46/replicate_paper1_rees46.py", ["--month", "2019-Nov"]),
    ("REES46 Oct: group B", "8-REES46/group_b_rees46.py", []),
    ("REES46 Nov: group B", "8-REES46/group_b_rees46.py", ["--month", "2019-Nov"]),
    ("Taobao: clean + sessions (10% users)", "10-Taobao/build_taobao_events.py", []),
    ("Taobao: group A", "8-REES46/replicate_paper1_rees46.py", ["--dataset", "taobao"]),
    ("Taobao: group B", "8-REES46/group_b_rees46.py", ["--dataset", "taobao"]),
    ("detection matrix + burst check", "9-Paper2/build_detection_matrix.py", []),
    ("headline confidence intervals", "9-Paper2/headline_cis.py", []),
    ("headline confidence intervals: Taobao", "9-Paper2/headline_cis.py", ["taobao"]),
    ("headline confidence intervals: REES46 Nov", "9-Paper2/headline_cis.py", ["rees46_nov"]),
    ("leak-size mechanism", "9-Paper2/leak_size_mechanism.py", []),
    ("negative control: Retail Rocket", "9-Paper2/negative_control.py", ["retail_rocket"]),
    ("negative control: flower shop", "9-Paper2/negative_control.py", ["flower_shop"]),
    ("negative control: REES46 Oct", "9-Paper2/negative_control.py", ["rees46_oct"]),
    ("negative control: REES46 Nov", "9-Paper2/negative_control.py", ["rees46_nov"]),
    ("negative control: Taobao", "9-Paper2/negative_control.py", ["taobao"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", type=int, default=1)
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

for i, (name, script, extra) in enumerate(STEPS, 1):
    if args.list:
        print(f"{i:2d}. {name}: {script} {' '.join(extra)}")
        continue
    if i < args.start:
        continue
    t0 = time.time()
    print(f"[{i:2d}/{len(STEPS)}] {name} ...", flush=True)
    path = ROOT / script
    done = subprocess.run([sys.executable, "-X", "utf8", path.name, *extra], cwd=path.parent,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if done.returncode:
        print(done.stderr[-3000:])
        sys.exit(f"step {i} failed: {script} {' '.join(extra)}")
    print(f"        done in {time.time() - t0:.0f} s", flush=True)
