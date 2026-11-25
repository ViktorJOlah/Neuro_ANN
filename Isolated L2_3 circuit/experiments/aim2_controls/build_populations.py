"""Population files for the Aim 2 controls (one model per PC; see circuit/population.py).

  homogeneous      every PC is the Standard (the existing hybrid run; written for completeness)
  shuffled         the 8 fitted models in equal shares, placed at random
  depth            the same shares, placed by depth: PCs sorted by soma depth, the shallowest eighth gets the
                   model of the shallowest recorded cell, and so on (rank match of network depth to the
                   recorded cells' normalized depth)
  rheo_matched     the shuffled placement, with each PC's synaptic input multiplied by its model's
                   rheobase / the Standard's rheobase. This removes the excitability (rheobase) offset
                   between models and keeps everything else (f-I gain, adaptation, subthreshold dynamics).

    python build_populations.py --prep PREP --extract-dir X --out populations
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from circuit import sonata_io as sio  # noqa: E402


def pc_depth(prep, extract_dir):
    with h5py.File(os.path.join(prep, "ids.h5")) as f:
        pc_to_l23 = f["pc_to_l23"][()]
    net = os.path.join(extract_dir, "network")
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    y = nodes["y"].to_numpy(float)[pc_to_l23]
    # Billeh coordinates: y is the cortical axis (more negative = deeper)
    return -y, nodes[["x", "y", "z"]].to_numpy(float)[pc_to_l23]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", required=True); p.add_argument("--extract-dir", required=True)
    p.add_argument("--out", required=True); p.add_argument("--models", default=os.path.join(HERE, "models.json"))
    p.add_argument("--excitability", default=os.path.join(HERE, "results", "models", "excitability.json"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = json.load(open(a.models))["models"]
    exc = json.load(open(a.excitability))["models"]
    K = len(spec)
    depth, xyz = pc_depth(a.prep, a.extract_dir)
    n = len(depth)
    mdirs = [{"name": m["name"], "dir": os.path.relpath(os.path.join(HERE, m["dir"]), a.out)} for m in spec]

    # depth placement: equal shares, shallow PCs to the shallowest recorded cell
    order_models = np.argsort([m["normalized_depth"] for m in spec])
    shares = np.array_split(np.argsort(depth, kind="stable"), K)
    depth_assign = np.zeros(n, np.int64)
    for rank, pcs in enumerate(shares):
        depth_assign[pcs] = order_models[rank]
    rng = np.random.default_rng(a.seed)
    shuffled = rng.permutation(depth_assign)  # same counts, random place
    rheo = np.array([exc[m["name"]]["rheobase_pa"] for m in spec], float)
    scale = (rheo / rheo[0])[shuffled].astype(np.float32)   # model 0 is the Standard

    np.save(os.path.join(a.out, "assign_homogeneous.npy"), np.zeros(n, np.int64))
    np.save(os.path.join(a.out, "assign_depth.npy"), depth_assign)
    np.save(os.path.join(a.out, "assign_shuffled.npy"), shuffled)
    np.save(os.path.join(a.out, "input_scale_rheo.npy"), scale)
    np.save(os.path.join(a.out, "pc_depth_um.npy"), depth.astype(np.float32))
    for name, asg, sc in (("homogeneous", "assign_homogeneous.npy", None), ("shuffled", "assign_shuffled.npy", None),
                          ("depth", "assign_depth.npy", None), ("rheo_matched", "assign_shuffled.npy", "input_scale_rheo.npy")):
        d = {"models": mdirs, "assignment": asg}
        if sc:
            d["input_scale"] = sc
        json.dump(d, open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
    lo = [float(depth[s].min()) for s in shares]; hi = [float(depth[s].max()) for s in shares]
    info = {"n_pc": n, "depth_range_um": [float(depth.min()), float(depth.max())],
            "depth_bins": [{"model": spec[order_models[r]]["name"], "normalized_depth": spec[order_models[r]]["normalized_depth"],
                            "network_depth_um": [lo[r], hi[r]]} for r in range(K)],
            "rheobase_pa": dict(zip([m["name"] for m in spec], rheo.tolist())),
            "rheo_input_scale": dict(zip([m["name"] for m in spec], (rheo / rheo[0]).tolist()))}
    json.dump(info, open(os.path.join(a.out, "populations.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
