"""Turn each cell's Allen NWB recording into 1 ms sweeps for fitting and comparison.

Run from "Isolated L2_3 circuit/":
    python experiments/real_cells/code/build_dataset.py --nwb-dir /mnt/project-files/real_cells_nwb

Input: <nwb-dir>/<cell>_ephys.nwb (Allen Cell Types Database, 50 kHz) and data/<cell>/sweeps.json.
Output: data/<cell>/sweeps_1ms.npz and data/<cell>/dataset.json.

Protocols (the Druckmann-style square / noise / ramp used everywhere in this project):
  square = "Long Square" (1 s steps) + "Square - 2s Suprathreshold"
  noise  = "Noise 1" + "Noise 2"
  ramp   = "Ramp"
Each sweep is cropped to [stimulus onset - 500 ms, stimulus offset + 500 ms] (the 10 ms test pulse at the
start of every Allen sweep is dropped), or to the end of the recorded response if it stops earlier (ramps
are stopped by the rig after the cell fires). The current is the NWB stimulus (it does not include the small
bias current, as in Allen's own GLIF and biophysical fits); 1 ms bins: current = bin mean, Vm = the first
sample of each bin except bins holding a spike peak, which get the bin maximum (so 1 ms Vm keeps every spike).
Spike times are also kept at full resolution (upward crossing of -20 mV, 50 kHz).

Split (the same for every model):
  test  = every "Noise 2" sweep (Allen's own held-out GLIF test stimulus), every third "Long Square" sweep in
          order of amplitude (starting with the second), and the last "Ramp" sweep.
  train = everything else. Only the Standard ANN is trained here; the Allen models are used as published.
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CELLS = ("513531471", "485468180", "502269786", "318808427", "521409057", "382982932")
PROTO = {"Long Square": "square", "Square - 2s Suprathreshold": "square",
         "Noise 1": "noise", "Noise 2": "noise", "Ramp": "ramp"}
PAD_S = 0.5
THR = -20.0


def spikes_fine(v, rate):
    up = np.flatnonzero((v[1:] >= THR) & (v[:-1] < THR)) + 1
    keep, last = [], -1e9
    for i in up:                                   # 1 ms refractory against noise re-crossings
        if i - last > rate * 1e-3:
            keep.append(i)
            last = i
    return np.array(keep, int)


def to_1ms(I, V, rate, spk):
    n = int(rate * 1e-3)
    T = len(I) // n
    Ib = I[:T * n].reshape(T, n).mean(1)
    Vb = V[:T * n:n].copy()
    for s in spk:
        b = s // n
        if b < T:
            Vb[b] = V[b * n:(b + 1) * n].max()
    return Ib.astype(np.float32), Vb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--cells", nargs="*", default=CELLS)
    a = ap.parse_args()
    for cell in a.cells:
        meta = {s["sweep_number"]: s for s in json.load(open(os.path.join(DATA, cell, "sweeps.json")))}
        f = h5py.File(os.path.join(a.nwb_dir, f"{cell}_ephys.nwb"), "r")
        arrays, rows = {}, []
        for key in f["stimulus/presentation"]:
            sw = int(key.split("_")[1])
            g, r = f["stimulus/presentation"][key], f["acquisition/timeseries"][key]
            name = g["aibs_stimulus_name"][()].decode()
            if name not in PROTO:
                continue
            rate = float(g["starting_time"].attrs["rate"])
            I = g["data"][()] * float(g["data"].attrs.get("conversion", 1.0)) * 1e12          # pA
            V = r["data"][()] * float(r["data"].attrs.get("conversion", 1.0)) * 1e3           # mV
            valid = np.flatnonzero(V != 0)
            if len(valid) < rate:
                continue
            end_rec = valid[-1] + 1
            nz = np.flatnonzero(np.abs(I[int(0.1 * rate):]) > 1e-3) + int(0.1 * rate)
            if not len(nz):
                continue
            a0 = max(0, nz[0] - int(PAD_S * rate))
            a1 = min(end_rec, nz[-1] + 1 + int(PAD_S * rate))
            I, V = I[a0:a1], V[a0:a1]
            if np.any(~np.isfinite(V)) or len(V) < 1.2 * rate:
                continue
            spk = spikes_fine(V, rate)
            Ib, Vb = to_1ms(I, V, rate, spk)
            k = f"s{sw}"
            arrays[k + "_I"], arrays[k + "_V"] = Ib, Vb
            arrays[k + "_spk_ms"] = (spk / rate * 1e3).astype(np.float64)
            m = meta.get(sw, {})
            amp = float(m.get("stimulus_absolute_amplitude") or np.round(np.median(Ib[Ib != 0]) if (Ib != 0).any() else 0))
            rows.append(dict(sweep=sw, stimulus=name, protocol=PROTO[name], amp_pA=amp, T_ms=len(Ib),
                             n_spikes=int(len(spk)), stim_on_ms=int(round((nz[0] - a0) / rate * 1e3)),
                             stim_off_ms=int(round((min(nz[-1], a1 - 1) - a0) / rate * 1e3)),
                             truncated=bool(a1 < nz[-1] + 1 + int(PAD_S * rate)),
                             leak_pA=m.get("leak_pa"), crop_start_s=a0 / rate))
        rows.sort(key=lambda r: r["sweep"])
        # split
        test = set()
        ls = sorted([r for r in rows if r["stimulus"] == "Long Square"], key=lambda r: (r["amp_pA"], r["sweep"]))
        test |= {r["sweep"] for r in ls[1::3]}
        test |= {r["sweep"] for r in rows if r["stimulus"] == "Noise 2"}
        ramps = [r for r in rows if r["stimulus"] == "Ramp"]
        if len(ramps) > 1:
            test.add(ramps[-1]["sweep"])
        for r in rows:
            r["split"] = "test" if r["sweep"] in test else "train"
        np.savez_compressed(os.path.join(DATA, cell, "sweeps_1ms.npz"), **arrays)
        summ = {}
        for sp in ("train", "test"):
            for p in ("square", "noise", "ramp"):
                rr = [r for r in rows if r["split"] == sp and r["protocol"] == p]
                summ[f"{sp}_{p}"] = dict(n=len(rr), minutes=round(sum(r["T_ms"] for r in rr) / 6e4, 2),
                                         spikes=sum(r["n_spikes"] for r in rr))
        json.dump(dict(cell=cell, sweeps=rows, summary=summ), open(os.path.join(DATA, cell, "dataset.json"), "w"), indent=1)
        print(cell, {k: (v["n"], v["minutes"], v["spikes"]) for k, v in summ.items()})


if __name__ == "__main__":
    main()
