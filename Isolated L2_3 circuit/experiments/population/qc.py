"""Recording-quality checks per cell (run-down, drift, too few spikes), from the 1 ms sweeps.

Run from "Isolated L2_3 circuit/":
    python experiments/population/qc.py

For every cell with real_cells/data/<cell>/dataset.json, writes population/qc.csv:
  noise_repeat_ratio  min over Noise 1 / Noise 2 of (spikes in the last repeat / spikes in the first repeat),
                      in recording order. The repeats are the identical stimulus, so a ratio well below 1 means
                      the cell fires less and less (run-down), as in 513531471 (21 -> 8 spikes, ratio 0.38).
  baseline_drift_mV   pre-stimulus Vm (Allen's pre_vm_mv) of the last minus the first analysed sweep.
  peak_drop_mV        first-spike peak (1 ms bins) on Long Square steps: median of the first 3 spiking steps minus
                      the median of the last 3, in recording order.
  noise_spikes_train, noise_spikes_test, train_minutes, rheobase_train_pA
Flags (semicolon list, empty = clean):
  rundown        noise_repeat_ratio < 0.6 (first repeat >= 8 spikes), or peak_drop_mV > 10
  drift          |baseline_drift_mV| > 5
  sparse_noise   fewer than 10 spikes in the test Noise 2 sweeps (spike timing cannot be scored well)
  no_noise_test  no Noise 2 sweeps
  no_rheobase    no spiking Long Square step in the training sweeps (fit_standard.py cannot rescale)
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.join(os.path.dirname(HERE), "real_cells")


def qc(cell):
    d = json.load(open(os.path.join(RC, "data", cell, "dataset.json")))
    meta = {s["sweep_number"]: s for s in json.load(open(os.path.join(RC, "data", cell, "sweeps.json")))}
    z = np.load(os.path.join(RC, "data", cell, "sweeps_1ms.npz"))
    rows = sorted(d["sweeps"], key=lambda r: r["sweep"])
    out = dict(cell=cell, train_minutes=round(sum(r["T_ms"] for r in rows if r["split"] == "train") / 6e4, 2))
    ratios = []
    for stim in ("Noise 1", "Noise 2"):
        rr = [r for r in rows if r["stimulus"] == stim]
        amps = {}
        for r in rr:
            amps.setdefault(round(r["amp_pA"]), []).append(r)
        for g in amps.values():
            if len(g) >= 2 and g[0]["n_spikes"] >= 8:
                ratios.append(g[-1]["n_spikes"] / g[0]["n_spikes"])
    out["noise_repeat_ratio"] = round(min(ratios), 3) if ratios else ""
    pre = [meta[r["sweep"]].get("pre_vm_mv") for r in rows if meta.get(r["sweep"], {}).get("pre_vm_mv") is not None]
    out["baseline_drift_mV"] = round(pre[-1] - pre[0], 2) if len(pre) >= 2 else ""
    peaks = []                            # peak of the first spike of each spiking Long Square step
    for r in rows:
        t = z[f"s{r['sweep']}_spk_ms"]
        if len(t) and r["stimulus"] == "Long Square":
            V = z[f"s{r['sweep']}_V"]
            peaks.append(float(V[min(int(t[0]), len(V) - 1)]))
    k = min(3, len(peaks) // 2)
    out["peak_drop_mV"] = round(float(np.median(peaks[:k]) - np.median(peaks[-k:])), 2) if k else ""
    out["noise_spikes_train"] = sum(r["n_spikes"] for r in rows if r["protocol"] == "noise" and r["split"] == "train")
    out["noise_spikes_test"] = sum(r["n_spikes"] for r in rows if r["stimulus"] == "Noise 2")
    ls = [r for r in rows if r["split"] == "train" and r["stimulus"] == "Long Square" and r["amp_pA"] > 0 and r["n_spikes"] > 0]
    out["rheobase_train_pA"] = min(r["amp_pA"] for r in ls) if ls else ""
    f = []
    if (ratios and min(ratios) < 0.6) or (out["peak_drop_mV"] != "" and out["peak_drop_mV"] > 10):
        f.append("rundown")
    if out["baseline_drift_mV"] != "" and abs(out["baseline_drift_mV"]) > 5:
        f.append("drift")
    if not any(r["stimulus"] == "Noise 2" for r in rows):
        f.append("no_noise_test")
    elif out["noise_spikes_test"] < 10:
        f.append("sparse_noise")
    if not ls:
        f.append("no_rheobase")
    out["flags"] = ";".join(f)
    return out


def main():
    cells = [r["cell"] for r in csv.DictReader(open(os.path.join(HERE, "cells.csv")))]
    res = [qc(c) for c in cells if os.path.exists(os.path.join(RC, "data", c, "dataset.json"))]
    with open(os.path.join(HERE, "qc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
