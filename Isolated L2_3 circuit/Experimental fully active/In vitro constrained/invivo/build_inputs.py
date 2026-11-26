"""In vivo-like input streams for one L2/3 PC, taken from the Allen V1 model (Billeh et al. 2020).

For a chosen L2/3 E cell of the GLIF network (pop e23Cux2) this collects every synapse onto it
(V1 -> V1 edges, LGN edges, background edge) with its source class, sign, weight (pA, glif_psc) and
delay, and the spike times of each source during the drifting-grating run used for the isolated L2/3
circuit (ori 315, contrast 80%, LGN/bkg trial 0, 3 s). V1 spike times come from the replay streams in
the project folder (isolated_l23/gratings_ori315_trial0: L2/3 reference spikes + v1ext streams), so the
inputs are exactly what that cell received in the full network.

Run:
  python invivo/build_inputs.py --v1-point /tmp/v1dl --streams /mnt/project-files/isolated_l23 \
      --out invivo/inputs --pick median low high
Writes <out>/cell_<v1id>.npz (arrays per edge: src_class, ei, weight_pA, delay_ms, spike times as a
flat array + offsets) and <out>/cells.json.
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd


def node_table(v1_point):
    nt = pd.read_csv(os.path.join(v1_point, "network", "v1_node_types.csv"), sep=" ")
    with h5py.File(os.path.join(v1_point, "network", "v1_nodes.h5")) as f:
        g = f["nodes/v1"]
        df = pd.DataFrame(dict(node_id=g["node_id"][:], node_type_id=g["node_type_id"][:],
                               x=g["0/x"][:], y=g["0/y"][:], z=g["0/z"][:], tuning=g["0/tuning_angle"][:]))
    df = df.merge(nt[["node_type_id", "pop_name", "ei"]], on="node_type_id", how="left")
    return df.sort_values("node_id").reset_index(drop=True)


def src_class(pop):
    """e23Cux2 -> 'L2/3 E', i23Pvalb -> 'L2/3 Pvalb', i1Htr3a -> 'L1 Htr3a', e4... -> 'L4 E'."""
    layer = {"1": "L1", "23": "L2/3", "4": "L4", "5": "L5", "6": "L6"}
    p = pop[1:]
    lay = next(k for k in ("23", "1", "4", "5", "6") if p.startswith(k))
    if pop[0] == "e":
        return f"{layer[lay]} E"
    for t in ("Pvalb", "Sst", "Htr3a"):
        if t in pop:
            return f"{layer[lay]} {t}"
    return f"{layer[lay]} I"


def edges_onto(h5path, pop, target):
    """All edges onto one target, read slice by slice from the target_to_source index."""
    with h5py.File(h5path) as f:
        g = f["edges"][pop]
        r = g["indices/target_to_source/node_id_to_range"][target]
        src, w, et = [], [], []
        for a, b in g["indices/target_to_source/range_to_edge_id"][int(r[0]):int(r[1])]:
            a, b = int(a), int(b)
            src.append(g["source_node_id"][a:b]); et.append(g["edge_type_id"][a:b])
            gi = g["edge_group_index"][a:b]
            w.append(g["0/syn_weight"][int(gi.min()):int(gi.max()) + 1][gi - gi.min()])
    cat = lambda x, d: np.concatenate(x).astype(d) if x else np.zeros(0, d)
    return cat(src, int), cat(w, float), cat(et, int)


def edge_delays(csv):
    t = pd.read_csv(csv, sep=" ")
    return dict(zip(t.edge_type_id, t.delay))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-point", required=True)
    ap.add_argument("--streams", required=True, help="isolated_l23 folder (id_maps.h5, gratings_ori315_trial0/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pick", nargs="+", default=["median", "low", "high"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    nodes = node_table(a.v1_point)
    run = os.path.join(a.streams, "gratings_ori315_trial0")
    with h5py.File(os.path.join(a.streams, "id_maps.h5")) as f:
        l23_to_v1 = f["l23_to_v1"][:]; ext_to_v1 = f["v1ext_to_v1"][:]
    spikes = {}
    for fn, grp, m in (("l23_reference_spikes.h5", "l23", l23_to_v1), ("v1ext_spikes.h5", "v1ext", ext_to_v1)):
        with h5py.File(os.path.join(run, fn)) as f:
            ids = m[f[f"spikes/{grp}/node_ids"][:].astype(int)]; ts = f[f"spikes/{grp}/timestamps"][:]
        o = np.argsort(ids, kind="stable")
        ids, ts = ids[o], ts[o]
        u, st = np.unique(ids, return_index=True)
        for i, s, e in zip(u, st, np.append(st[1:], len(ids))):
            spikes[int(i)] = np.sort(ts[s:e])
    rates = pd.read_csv(os.path.join(run, "rates_per_cell.csv"))
    rates["v1_id"] = l23_to_v1[rates.l23_id.values]
    e = rates[rates["class"] == "L2/3 E"].merge(nodes[["node_id", "x", "z", "y"]], left_on="v1_id", right_on="node_id")
    e = e[(np.hypot(e.x, e.z) < 200)]                           # central column
    act = e[e.rate_full_hz > 0]
    choice = {"median": act.rate_full_hz.median(), "low": act.rate_full_hz.quantile(0.2),
              "high": act.rate_full_hz.quantile(0.95)}
    lgn_sp = pd.read_csv(os.path.join(a.v1_point, "lgn_stimulus",
                                      "full3_production_3.0sec_SF0.04_TF2.0_ori315.0_c80.0_gs0.5",
                                      "spikes.trial_0.csv"), sep=" ")
    bkg_sp = pd.read_csv(os.path.join(a.v1_point, "bkg_inputs", "bkg_spikes_n1_fr1000_dt0.25", "spikes.trial_0.csv"), sep=" ")
    lgn_by = {k: np.sort(v.timestamps.values) for k, v in lgn_sp.groupby("node_ids")}
    bkg_t = np.sort(bkg_sp.timestamps.values)
    net = os.path.join(a.v1_point, "network")
    d_v1 = edge_delays(os.path.join(net, "v1_v1_edge_types.csv"))
    d_lgn = edge_delays(os.path.join(net, "lgn_v1_edge_types.csv"))
    d_bkg = edge_delays(os.path.join(net, "bkg_v1_edge_types.csv"))
    pops = nodes.pop_name.values; eis = nodes.ei.values
    meta = []
    for name in a.pick:
        target_rate = choice[name]
        cand = act.iloc[(act.rate_full_hz - target_rate).abs().argsort()[:20]]
        row = cand.iloc[rng.integers(len(cand))]
        tid = int(row.v1_id)
        src, w, et = edges_onto(os.path.join(net, "v1_v1_edges.h5"), "v1_to_v1", tid)
        cls = [src_class(pops[s]) for s in src]
        ei = [eis[s] for s in src]
        dl = [d_v1[t] for t in et]
        sp = [spikes.get(int(s), np.zeros(0)) for s in src]
        lsrc, lw, let = edges_onto(os.path.join(net, "lgn_v1_edges.h5"), "lgn_to_v1", tid)
        cls += ["LGN"] * len(lsrc); ei += ["e"] * len(lsrc); w = np.concatenate([w, lw])
        dl += [d_lgn[t] for t in let]; sp += [lgn_by.get(int(s), np.zeros(0)) for s in lsrc]
        bsrc, bw, bet = edges_onto(os.path.join(net, "bkg_v1_edges.h5"), "bkg_to_v1", tid)
        cls += ["bkg"] * len(bsrc); ei += ["e"] * len(bsrc); w = np.concatenate([w, bw])
        dl += [d_bkg[t] for t in bet]; sp += [bkg_t for _ in bsrc]
        off = np.cumsum([0] + [len(s) for s in sp])
        np.savez_compressed(os.path.join(a.out, f"cell_{tid}.npz"), src_class=np.array(cls), ei=np.array(ei),
                            weight_pA=w, delay_ms=np.array(dl, float), spikes=np.concatenate(sp), offsets=off)
        summ = pd.Series(cls).value_counts().to_dict()
        meta.append(dict(pick=name, v1_id=tid, glif_rate_hz=float(row.rate_full_hz), n_edges=len(cls),
                         edges_by_class=summ, x=float(row.x), y=float(row.y), z=float(row.z)))
        print(meta[-1])
    json.dump(meta, open(os.path.join(a.out, "cells.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
