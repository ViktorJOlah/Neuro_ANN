"""Drive the Experimental fully active L2/3 PC with in vivo-like synaptic input from the Allen V1 model.

Inputs (invivo/build_inputs.py): every synapse a GLIF L2/3 E cell received in the full V1 network during
3 s of drifting gratings, with the presynaptic spike trains. Each Billeh edge (weight w in pA) becomes
K_PER_EDGE dendritic synapses, each with peak AMPA conductance g_scale * |w| / K (uS per pA) and the
model's NMDA:AMPA ratio for its location; inhibitory edges become GABA-A synapses with gi_scale * |w| / K.
The 1 kHz background source is split into N_BKG independent streams spread over the whole arbor.

Placement by source (bottom-up proximal vs top-down distal, as in Olah et al. 2025 and the review):
  LGN, L4 E, L5 E    -> basal + apical dendrite within 100 um of the soma (not tuft)
  L2/3 E             -> basal + apical dendrite outside the tuft
  bkg                -> anywhere on the dendrites (length-weighted)
  top-down (added)   -> tuft only; N_TD Poisson inputs at --td-rate Hz (the Billeh model has no
                        long-range cortical feedback, so this stream is synthetic and swept)
  Pvalb              -> soma and basal/apical < 50 um
  Sst                -> apical > 50 um (incl. tuft)
  L1 Htr3a           -> tuft; other Htr3a -> dendrites > 50 um

Run:  python invivo/run_invivo.py --cell 82141 --cond control --td-rate 0 --out results/invivo
Saves npz with soma Vm, spike times, per-segment Vm (1 ms) and [Ca]i (1 ms) and segment metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from model.l23_cell import L23Cell, load_params  # noqa: E402
from neuron import h  # noqa: E402

K_PER_EDGE = 3
N_BKG = 200
N_TD = 100
T_STOP = 3000.0


def zone_pools(cell):
    segs = []
    for sec in cell.dends:
        kind = "apical" if sec in cell.apical else "basal"
        for seg in sec:
            d = h.distance(seg.x, sec=sec)
            segs.append(dict(sec=sec, x=seg.x, d=d, kind=kind, tuft=cell._is_tuft(sec, seg.x),
                             w=sec.L / sec.nseg))
    soma = [dict(sec=cell.soma[0], x=0.5, d=0.0, kind="soma", tuft=False, w=50.0)]
    Z = {
        "proximal": [s for s in segs if s["d"] <= 100 and not s["tuft"]],
        "nontuft": [s for s in segs if not s["tuft"]],
        "all": segs,
        "tuft": [s for s in segs if s["tuft"]],
        "perisomatic": soma + [s for s in segs if s["d"] <= 50],
        "distal_apical": [s for s in segs if s["kind"] == "apical" and s["d"] > 50],
        "dend_gt50": [s for s in segs if s["d"] > 50],
    }
    return Z


def zone_for(cls):
    if cls in ("LGN", "L4 E", "L5 E", "L6 E"):
        return "proximal"
    if cls == "L2/3 E":
        return "nontuft"
    if cls in ("bkg",):
        return "all"
    if cls == "topdown":
        return "tuft"
    if "Pvalb" in cls:
        return "perisomatic"
    if "Sst" in cls:
        return "distal_apical"
    if cls == "L1 Htr3a":
        return "tuft"
    return "dend_gt50"


def build(cell, inputs, g_scale, gi_scale, td_rate, td_w, seed, nmda_ratio=None):
    rng = np.random.default_rng(seed)
    Z = zone_pools(cell)
    pick = {}
    for k, pool in Z.items():
        w = np.array([p["w"] for p in pool]); pick[k] = (pool, w / w.sum())
    keep = []                         # (netcons, event times incl. delay)
    cls, ei, wts, dly = inputs["src_class"], inputs["ei"], inputs["weight_pA"], inputs["delay_ms"]
    sp, off = inputs["spikes"], inputs["offsets"]
    trains = []
    for i in range(len(cls)):
        t = sp[off[i]:off[i + 1]]
        if cls[i] == "bkg":
            lab = rng.integers(N_BKG, size=len(t))
            for j in range(N_BKG):
                trains.append(("bkg", "e", wts[i], dly[i], t[lab == j], 1))
        else:
            trains.append((cls[i], ei[i], wts[i], dly[i], t, K_PER_EDGE))
    for j in range(N_TD if td_rate > 0 else 0):
        n = rng.poisson(td_rate * T_STOP / 1000.0)
        trains.append(("topdown", "e", td_w, 2.0, np.sort(rng.uniform(0, T_STOP, n)), K_PER_EDGE))
    counts = {}
    for c, e, w, d, t, k in trains:
        if len(t) == 0:
            continue
        pool, p = pick[zone_for(c)]
        for _ in range(k):
            s = pool[rng.choice(len(pool), p=p)]
            if e == "e":
                syn = cell.add_synapse(s["sec"], s["x"], g_ampa=g_scale * abs(w) / k, nmda_ratio=nmda_ratio)
            else:
                syn = cell.add_synapse(s["sec"], s["x"], g_ampa=gi_scale * abs(w) / k, kind="inh")
            syn["src"] = c
            ncs = cell.drive(syn, None)
            keep.append((ncs, np.asarray(t, float) + d))
            counts[c] = counts.get(c, 0) + 1
    return keep, counts


def run(cell_id, cond, td_rate, g_scale, gi_scale, td_w, seed, inputs_dir, record_dend=True, params=None):
    cell = L23Cell(params or load_params())
    inputs = dict(np.load(os.path.join(inputs_dir, f"cell_{cell_id}.npz"), allow_pickle=False))
    keep, counts = build(cell, inputs, g_scale, gi_scale, td_rate, td_w, seed)
    for b in cond.split("+"):
        if b != "control":
            cell.block(b)
    vs = h.Vector(); vs.record(cell.soma[0](0.5)._ref_v, 0.1)
    meta, vrec, carec = [], [], []
    if record_dend:
        for sec in cell.dends + cell.soma:
            for seg in sec:
                kind = "soma" if sec in cell.soma else ("apical" if sec in cell.apical else "basal")
                meta.append((sec.name(), seg.x, h.distance(seg.x, sec=sec), kind, int(cell._is_tuft(sec, seg.x)),
                             cell.height(sec, seg.x), sec in cell.trunk))
                a = h.Vector(); a.record(seg._ref_v, 1.0); vrec.append(a)
                b = h.Vector(); b.record(seg._ref_cai, 1.0); carec.append(b)
    nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0]); nc.threshold = -10
    spk = h.Vector(); nc.record(spk)
    t0 = time.time()
    cell.init()
    for ncs, times in keep:          # queue every presynaptic spike (after init, t = 0)
        for tt in times:
            for nc in ncs:
                nc.event(tt)
    h.continuerun(T_STOP)
    out = dict(vsoma=np.array(vs, np.float32), spikes=np.array(spk), runtime_s=time.time() - t0, counts=counts)
    if record_dend:
        out.update(vdend=np.array([np.array(v) for v in vrec], np.float32),
                   cadend=np.array([np.array(c) for c in carec], np.float32), seg_meta=meta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="82141")
    ap.add_argument("--cond", default="control")
    ap.add_argument("--td-rate", type=float, default=0.0)
    ap.add_argument("--td-w", type=float, default=2.0, help="top-down weight in Billeh pA units")
    ap.add_argument("--g-scale", type=float, default=None)
    ap.add_argument("--gi-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", default=os.path.join(HERE, "inputs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "invivo"))
    ap.add_argument("--no-dend", action="store_true")
    a = ap.parse_args()
    cal = json.load(open(os.path.join(HERE, "calibration.json"))) if os.path.exists(os.path.join(HERE, "calibration.json")) else {}
    g = a.g_scale if a.g_scale is not None else cal.get("g_scale", 2e-4)
    gi = a.gi_scale if a.gi_scale is not None else cal.get("gi_scale", 3 * g)
    r = run(a.cell, a.cond, a.td_rate, g, gi, a.td_w, a.seed, a.inputs, record_dend=not a.no_dend)
    os.makedirs(a.out, exist_ok=True)
    tag = f"cell{a.cell}_{a.cond.replace('+', '-')}_td{a.td_rate:g}_s{a.seed}"
    meta = r.pop("seg_meta", None)
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), **{k: v for k, v in r.items() if k != "counts"},
                        seg_meta=np.array(meta, dtype=object) if meta else np.zeros(0))
    print(json.dumps(dict(tag=tag, rate_hz=len(r["spikes"]) / (T_STOP / 1000), runtime_s=round(r["runtime_s"], 1),
                          g_scale=g, gi_scale=gi, n_syn=r["counts"])))


if __name__ == "__main__":
    main()
