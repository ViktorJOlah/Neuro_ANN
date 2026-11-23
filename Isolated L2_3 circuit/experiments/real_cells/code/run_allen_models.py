"""Run the published Allen models of a cell (all GLIF levels and the all-active NEURON model) on its test sweeps.

Run from "Isolated L2_3 circuit/":
    python experiments/real_cells/code/run_allen_models.py --cell 513531471 --nwb-dir /mnt/project-files/real_cells_nwb
        [--which glif|allactive|both] [--dt 0.025]

Both models get the full-resolution NWB stimulus of each test sweep, cropped exactly as in build_dataset.py.
  GLIF: Allen GLIF code (vendored in code/glif/, unchanged apart from imports) at the model's own dt (50 us),
        stimulus resampled to it; Vm = model voltage + El_reference; spike times = interpolated spike times.
  All-active: Nandi et al. 2022 hall-of-fame #0 model built with simulations/additional_cells/code/allactive_cell.py
        (stub axon, 34 C, steady-state start, junction potential -14 mV as Allen); dt 0.025 ms, IClamp at the
        soma; spike times = upward crossings of -20 mV (recording convention).
Both are reduced to 1 ms with the same rule as the recordings (bin start, spike bins get the bin maximum; GLIF
spike bins are set to +20 mV since GLIF has no spike shape).

Output: results/<cell>/<model>/predictions.npz with s<sweep>_vm (1 ms) and s<sweep>_spk (spike times, ms).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(RC))
THR = -20.0
GLIF_NAMES = {1: "glif1_LIF", 2: "glif2_LIF-R", 3: "glif3_LIF-ASC", 4: "glif4_LIF-R-ASC", 5: "glif5_LIF-R-ASC-A"}


def test_sweeps(cell, nwb_dir):
    meta = json.load(open(os.path.join(RC, "data", cell, "dataset.json")))
    f = h5py.File(os.path.join(nwb_dir, f"{cell}_ephys.nwb"), "r")
    for r in meta["sweeps"]:
        if r["split"] != "test":
            continue
        g = f["stimulus/presentation"][f"Sweep_{r['sweep']}"]
        rate = float(g["starting_time"].attrs["rate"])
        a0 = int(round(r["crop_start_s"] * rate))
        n = int(round(r["T_ms"] * 1e-3 * rate))
        I = g["data"][a0:a0 + n] * float(g["data"].attrs.get("conversion", 1.0)) * 1e12     # pA
        yield r, I, rate


def reduce_1ms(v_fine, dt_ms, spk_ms, T, spike_value=None):
    n = int(round(1.0 / dt_ms))
    v = np.asarray(v_fine)[: T * n]
    if len(v) < T * n:
        v = np.concatenate([v, np.full(T * n - len(v), v[-1])])
    out = v[::n].copy()
    for s in spk_ms:
        b = int(s)
        if b < T:
            out[b] = spike_value if spike_value is not None else np.nanmax(v[b * n:(b + 1) * n])
    return out.astype(np.float32)


def run_glif(cell, nwb_dir):
    sys.path.insert(0, os.path.join(HERE, "glif"))
    from glif_neuron import GlifNeuron
    for cfg in sorted(glob.glob(os.path.join(RC, "data", cell, "glif", "model_*_neuron_config.json"))):
        level = int(re.search(r"model_(\d)_", os.path.basename(cfg)).group(1))
        name = GLIF_NAMES[level]
        out = os.path.join(RC, "results", cell, name)
        os.makedirs(out, exist_ok=True)
        conf = json.load(open(cfg))
        nrn = GlifNeuron.from_dict(conf)
        dt = nrn.dt                                       # s
        preds, t0 = {}, time.time()
        for r, I, rate in test_sweeps(cell, nwb_dir):
            t_in = np.arange(len(I)) / rate
            t_g = np.arange(0, len(I) / rate, dt)
            stim = np.interp(t_g, t_in, I) * 1e-12        # A
            o = nrn.run(stim)
            v = (o["voltage"] + conf["El_reference"]) * 1e3   # mV, NaN during the spike cut
            spk = np.asarray(o["interpolated_spike_times"], float) * 1e3
            v = np.where(np.isnan(v), np.nan, v)
            # fill spike-cut NaNs with the reset value that follows (for plotting / RMSE outside spikes)
            idx = np.where(~np.isnan(v), np.arange(len(v)), len(v) - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            v = v[np.minimum(idx, len(v) - 1)]
            preds[f"s{r['sweep']}_vm"] = reduce_1ms(v, dt * 1e3, spk, r["T_ms"], spike_value=20.0)
            preds[f"s{r['sweep']}_spk"] = spk
        np.savez_compressed(os.path.join(out, "predictions.npz"), **preds)
        json.dump(dict(model=name, config=os.path.basename(cfg), dt_s=dt, wall_s=time.time() - t0),
                  open(os.path.join(out, "info.json"), "w"), indent=1)
        print(cell, name, f"{time.time()-t0:.0f}s", flush=True)


def run_allactive(cell, nwb_dir, dt):
    sys.path.insert(0, os.path.join(ROOT, "simulations", "additional_cells", "code"))
    from neuron import h
    import allactive_cell as ac
    mech = os.path.join(ROOT, "simulations", "additional_cells", "models")
    out = os.path.join(RC, "results", cell, "allactive")
    os.makedirs(out, exist_ok=True)
    R = ac.Runner(os.path.join(RC, "data", cell, "allactive"), mech, dt=dt)
    jp = R.info["junction_potential"]
    R.vrec.record(h.soma[0](0.5)._ref_v, 0.1)             # finer sampling for the spike-peak rule
    tspk = h.Vector()
    nc = h.NetCon(h.soma[0](0.5)._ref_v, None, sec=h.soma[0])
    nc.threshold = THR + jp                               # reported = sim - jp
    nc.record(tspk)
    preds, t0 = {}, time.time()
    for r, I, rate in test_sweeps(cell, nwb_dir):
        step = max(1, int(round(rate * 1e-4)))           # play the current at 0.1 ms
        Ic = I[: len(I) // step * step].reshape(-1, step).mean(1)
        tspk.resize(0)
        v = R.run(Ic, 0.1)                                # recording convention, sampled at 0.1 ms
        spk = np.array(tspk, float)
        preds[f"s{r['sweep']}_vm"] = reduce_1ms(v, 0.1, spk, r["T_ms"])
        preds[f"s{r['sweep']}_spk"] = spk
        print(cell, "allactive sweep", r["sweep"], r["stimulus"], f"spikes {r['n_spikes']} -> {len(spk)}",
              f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(out, "predictions.npz"), **preds)
    json.dump(dict(model="allactive", source=json.load(open(os.path.join(RC, "data", cell, "allactive", "source.json")))
                   if os.path.exists(os.path.join(RC, "data", cell, "allactive", "source.json")) else None,
                   dt_ms=dt, info=R.info, rest_mV=R.rest_raw - jp, wall_s=time.time() - t0),
              open(os.path.join(out, "info.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--which", default="both", choices=("glif", "allactive", "both"))
    ap.add_argument("--dt", type=float, default=0.025)
    a = ap.parse_args()
    if a.which in ("glif", "both"):
        run_glif(a.cell, a.nwb_dir)
    if a.which in ("allactive", "both"):
        run_allactive(a.cell, a.nwb_dir, a.dt)


if __name__ == "__main__":
    main()
