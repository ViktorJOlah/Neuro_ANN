"""Fit the Standard (PR 11 recipe + PR 13 timing refit) to every cell in cells.csv, a few cells at a time.

Run from "Isolated L2_3 circuit/":
    python experiments/population/run_population.py --workers 2 [--cells ...]

Per cell, each step is skipped when its output already exists (so the run can be restarted):
  1. real_cells/code/build_dataset.py      -> real_cells/data/<cell>/dataset.json
  2. real_cells/code/fit_standard.py       -> real_cells/results/<cell>/standard/         (PR 11, ~30-40 min)
  3. spike_timing/fit_branches.py          -> real_cells/results/<cell>/standard_branches/ (PR 13, ~3 min)
  4. spike_timing/fit_spikes.py --mode smooth --start standard_branches --evals 150
                                           -> real_cells/results/<cell>/standard_timing/   (PR 13, ~4 min)
The NWB must already be in --nwb-dir (fetch_allen.py). A cell whose step fails is logged and skipped.
Logs: population/logs/<cell>.log; status: population/logs/status.tsv. With --mirror, each finished cell's
data and results folders are also copied there (a checkpoint outside the container).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RC = os.path.join(ROOT, "experiments", "real_cells")
ST = os.path.join(ROOT, "experiments", "spike_timing")
PY = sys.executable


def steps(cell, nwb):
    res = os.path.join(RC, "results", cell)
    return [
        ("build", os.path.join(RC, "data", cell, "dataset.json"),
         [PY, os.path.join(RC, "code", "build_dataset.py"), "--nwb-dir", nwb, "--cells", cell]),
        ("standard", os.path.join(res, "standard", "params.json"),
         [PY, os.path.join(RC, "code", "fit_standard.py"), "--cell", cell]),
        ("branches", os.path.join(res, "standard_branches", "params.json"),
         [PY, os.path.join(ST, "fit_branches.py"), "--cell", cell]),
        ("timing", os.path.join(res, "standard_timing", "params.json"),
         [PY, os.path.join(ST, "fit_spikes.py"), "--cell", cell, "--start", "standard_branches", "--mode", "smooth",
          "--evals", "150", "--tag", "standard_timing"]),
    ]


def status(cell, step, state, t):
    with open(os.path.join(HERE, "logs", "status.tsv"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{cell}\t{step}\t{state}\t{t:.0f}\n")


def run_cell(cell, a):
    if not os.path.exists(os.path.join(a.nwb_dir, f"{cell}_ephys.nwb")):
        status(cell, "nwb", "missing", 0)
        return
    log = open(os.path.join(HERE, "logs", f"{cell}.log"), "a")
    for name, out, cmd in steps(cell, a.nwb_dir):
        if os.path.exists(out):
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and os.path.exists(out)
        status(cell, name, "ok" if ok else f"fail{r.returncode}", time.time() - t0)
        if not ok:
            return
    status(cell, "all", "ok", 0)
    if a.mirror:
        for sub in ("data", "results"):
            src = os.path.join(RC, sub, cell)
            shutil.copytree(src, os.path.join(a.mirror, sub, cell), dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nwb-dir", default="/mnt/project-files/real_cells_nwb")
    ap.add_argument("--mirror", default="")
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    cells = a.cells or [r["cell"] for r in csv.DictReader(open(os.path.join(HERE, "cells.csv")))]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda c: run_cell(c, a), cells))
    print("POPULATION_DONE", flush=True)


if __name__ == "__main__":
    main()
