"""Score every population fit on its held-out test sweeps and flag bad fits.

Run from "Isolated L2_3 circuit/":
    python experiments/population/score.py [--models standard_timing standard]

Uses the metric code of real_cells/code/compare.py unchanged (same test split, scored from 450 ms; Γ ±4 ms).
Writes population/fit_quality.csv, one row per (cell, model):
  noise_gamma, noise_gamma_repeat (the cell's own repeat-to-repeat Γ), noise_gamma_norm, noise_spike_ev_ratio,
  noise_count_bias, noise_count_abs_err, noise_n_data (mean spikes per Noise 2 sweep), noise_rmse_sub_mV,
  noise_vm_ev_sub, square_rmse_sub_mV, square_vm_ev_sub, square_ss_err_mV, square_count_abs_err,
  square_latency_err_ms, test_spikes_data, test_spikes_model, prestim_spikes (model spikes between 450 ms and
  stimulus onset, summed over test sweeps)
and fit_flags (semicolon list, empty = usable):
  silent        the model fires < 25 % of the recorded test spikes
  overfire      the model fires > 2x the recorded test spikes
  prestim       the model fires > 5 spikes before the stimulus starts (fires with no input, as 485468180 did)
  poor_vm       square Vm RMSE away from spikes > 5 mV
  missing       no predictions for this model
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "real_cells", "code"))
import compare as cmp  # noqa: E402


def score(cell, models):
    meta, test, data, preds = cmp.load_cell(cell)
    rows = []
    for m in models:
        out = dict(cell=cell, model=m)
        if m not in preds:
            out["fit_flags"] = "missing"
            rows.append(out)
            continue
        P = preds[m]
        per = []
        for r in test:
            d = data[r["sweep"]]
            per.append((r, cmp.sweep_metrics(r, d["V"], d["spk"], P[r["sweep"]]["vm"], P[r["sweep"]]["spk"])))
        _, nm = cmp.noise_metrics(test, data, {m: P})
        intr, _ = cmp.noise_metrics(test, data, {})
        mean = lambda xs: float(np.nanmean(xs)) if len(xs) else np.nan
        for proto, key in (("noise", "Noise 2"), ("square", "Long Square")):
            pp = [x for r, x in per if r["stimulus"] == key]
            out[f"{proto}_gamma"] = mean([x["gamma"] for x in pp if x["n_data"] or x["n_model"]])
            out[f"{proto}_count_bias"] = mean([x["count_err"] for x in pp])
            out[f"{proto}_count_abs_err"] = mean([abs(x["count_err"]) for x in pp])
            out[f"{proto}_n_data"] = mean([x["n_data"] for x in pp])
            out[f"{proto}_rmse_sub_mV"] = mean([x["rmse_sub_mV"] for x in pp if "rmse_sub_mV" in x])
            out[f"{proto}_vm_ev_sub"] = mean([x["vm_ev_sub"] for x in pp if "vm_ev_sub" in x])
        out["square_ss_err_mV"] = mean([x["ss_err_mV"] for _, x in per if "ss_err_mV" in x])
        out["square_latency_err_ms"] = mean([x["latency_err_ms"] for _, x in per if "latency_err_ms" in x])
        out["noise_gamma_repeat"] = intr.get("gamma_repeat", np.nan) if intr else np.nan
        out["noise_gamma_norm"] = nm.get(m, {}).get("gamma_norm", np.nan)
        out["noise_spike_ev_ratio"] = nm.get(m, {}).get("spike_ev_ratio", np.nan)
        nd = sum(len(data[r["sweep"]]["spk"][data[r["sweep"]]["spk"] >= cmp.SEED]) for r in test)
        nmod = sum(int(np.sum(P[r["sweep"]]["spk"] >= cmp.SEED)) for r in test)
        pre = sum(int(np.sum((P[r["sweep"]]["spk"] >= cmp.SEED) & (P[r["sweep"]]["spk"] < r["stim_on_ms"]))) for r in test)
        out.update(test_spikes_data=nd, test_spikes_model=nmod, prestim_spikes=pre)
        f = []
        if nd and nmod < 0.25 * nd:
            f.append("silent")
        if nmod > 2 * max(nd, 5):
            f.append("overfire")
        if pre > 5:
            f.append("prestim")
        if np.isfinite(out["square_rmse_sub_mV"]) and out["square_rmse_sub_mV"] > 5:
            f.append("poor_vm")
        out["fit_flags"] = ";".join(f)
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["standard_timing", "standard"])
    a = ap.parse_args()
    cells = [r["cell"] for r in csv.DictReader(open(os.path.join(HERE, "cells.csv")))]
    rows = []
    for c in cells:
        if os.path.exists(os.path.join(cmp.RES, c)):
            rows += score(c, a.models)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(HERE, "fit_quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(len(rows), "rows ->", os.path.join(HERE, "fit_quality.csv"))


if __name__ == "__main__":
    main()
