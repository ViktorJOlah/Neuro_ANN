"""Hybrid prep with homogeneous interneurons: every interneuron of a class gets that class's most common GLIF
model (Pvalb 341442651, Sst 324266189, Htr3a 475622793). Cells, positions, edges, weights and inputs are
unchanged; only the GLIF parameters of the interneurons that had a minority model change (5,857 of 9,892).

The new prep links every large file of the original prep and only rewrites the interneuron node file and the
NEST config.

    python make_homogeneous_interneurons.py --prep PREP --out PREP_HOMOG_IN
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import h5py
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    src, out = os.path.abspath(a.prep), os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "nest_network"), exist_ok=True)
    for f in os.listdir(src):
        if f not in ("nest_network", "nest_output", "config_nest_part.json") and not os.path.exists(os.path.join(out, f)):
            os.symlink(os.path.join(src, f), os.path.join(out, f))
    for f in os.listdir(os.path.join(src, "nest_network")):
        d = os.path.join(out, "nest_network", f)
        if f == "l23i_nodes.h5":
            shutil.copy(os.path.join(src, "nest_network", f), d)
        elif not os.path.exists(d):
            os.symlink(os.path.join(src, "nest_network", f), d)
    types = pd.read_csv(os.path.join(src, "nest_network", "l23i_node_types.csv"), sep=" ")
    pop = dict(zip(types["node_type_id"], types["pop_name"]))
    with h5py.File(os.path.join(out, "nest_network", "l23i_nodes.h5"), "r+") as f:
        tid = f["nodes/l23i/node_type_id"][()]
        cls = np.array([pop[t] for t in tid])
        new = tid.copy()
        info = {}
        for c in np.unique(cls):
            m = cls == c
            u, n = np.unique(tid[m], return_counts=True)
            keep = int(u[np.argmax(n)])
            new[m] = keep
            info[c] = {"kept_model": keep, "n_cells": int(m.sum()), "n_changed": int((tid[m] != keep).sum()),
                       "models_before": dict(zip(map(str, u.tolist()), n.tolist()))}
        f["nodes/l23i/node_type_id"][...] = new
    cfg = json.load(open(os.path.join(src, "config_nest_part.json")))
    s = json.dumps(cfg).replace(src + "/", out + "/")
    cfg = json.loads(s)
    cfg["output"]["output_dir"] = os.path.join(out, "nest_output")
    json.dump(cfg, open(os.path.join(out, "config_nest_part.json"), "w"), indent=2)
    json.dump(info, open(os.path.join(out, "homogeneous_interneurons.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
