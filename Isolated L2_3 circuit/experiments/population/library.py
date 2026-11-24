"""Load cells from the population library (population/library/, built by build_library.py).

    import sys; sys.path.insert(0, "experiments/population")
    import library
    idx = library.index()                       # list of dicts from index.csv (depth_norm, flags, metrics, ...)
    cell = library.load_cell("318808427")       # a Standard StandardCell with this cell's fitted parameters
    cells, rows = library.load_population()     # every use == 1 cell; use_only=False for all fitted cells

Each StandardCell takes the raw injected current in pA (the per-cell input rescaling is folded into its
front-end weights, g and I0) and is run like the Standard (`simulate`, or step-by-step as in circuit/hybrid.py).
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "library")
STANDARD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Standard")


def _sm():
    if STANDARD not in sys.path:
        sys.path.insert(0, STANDARD)
    import standard_model as sm
    return sm


def index():
    return list(csv.DictReader(open(os.path.join(LIB, "index.csv"))))


def load_cell(cell):
    sm = _sm()
    p = json.load(open(os.path.join(LIB, str(cell), "params.json")))
    net = sm.build_front_end()
    w = np.load(os.path.join(LIB, str(cell), "front_end.npz"))
    net.set_weights([w[f"w{i}"] for i in range(len(w.files))])
    return sm.StandardCell(p["izhikevich"], p["branches"], net)


def load_population(use_only=True):
    rows = [r for r in index() if not use_only or r["use"] == "1"]
    return [load_cell(r["cell"]) for r in rows], rows
