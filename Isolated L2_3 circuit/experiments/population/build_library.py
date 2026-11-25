"""Collect the population fits into a per-cell parameter library for the hybrid L2/3 circuit.

Run from "Isolated L2_3 circuit/" after run_population.py, qc.py and score.py:
    python experiments/population/build_library.py [--model standard_timing]

Writes population/library/:
  <cell>/params.json    Izhikevich + branch parameters (training history removed) and the input rescaling k
                        (already folded into the front-end weights, g and I0: the cell takes raw current in pA)
  <cell>/front_end.npz  fitted CNN-LSTM weights (keys w0..wN, as real_cells fit_standard.py writes them)
  index.csv             one row per fitted cell: soma depth, Cre line, Allen ephys features, recording QC flags,
                        fit-quality metrics, fit flags, key fitted parameters, and `use`:
                          use = 1  recording clean (qc flags empty or only no_noise_test) and fit flags empty
                          use = 0  otherwise (kept for sensitivity analyses, e.g. including run-down cells)
Load with population/library.py (load_cell, load_population).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "real_cells", "results")
LIB = os.path.join(HERE, "library")
KEEP_CELLS = ("cell", "line", "reporter", "depth_norm", "ccf_x", "ccf_y", "ccf_z", "rheobase_pA", "ri_MOhm",
              "tau_ms", "vrest_mV", "fi_slope", "adaptation", "glif", "biophys_all_active")
KEEP_FIT = ("noise_gamma", "noise_gamma_repeat", "noise_gamma_norm", "noise_count_bias", "noise_n_data",
            "noise_rmse_sub_mV", "square_rmse_sub_mV", "square_ss_err_mV", "square_count_abs_err",
            "square_latency_err_ms", "test_spikes_data", "test_spikes_model", "prestim_spikes")
KEY_PARAMS = ("V_th", "c_mV", "d", "ln_s_sk", "ln_s_ca", "ln_s_kp")


def rd(name):
    return {r["cell"]: r for r in csv.DictReader(open(os.path.join(HERE, name)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="standard_timing")
    a = ap.parse_args()
    cells, qc = rd("cells.csv"), rd("qc.csv")
    fq = {r["cell"]: r for r in csv.DictReader(open(os.path.join(HERE, "fit_quality.csv"))) if r["model"] == a.model}
    os.makedirs(LIB, exist_ok=True)
    rows = []
    for c, info in cells.items():
        src = os.path.join(RES, c, a.model)
        if not os.path.exists(os.path.join(src, "params.json")):
            continue
        p = json.load(open(os.path.join(src, "params.json")))
        p.pop("history", None)
        p["source"] = f"experiments/real_cells/results/{c}/{a.model}"
        os.makedirs(os.path.join(LIB, c), exist_ok=True)
        json.dump(p, open(os.path.join(LIB, c, "params.json"), "w"), indent=1)
        shutil.copy(os.path.join(src, "front_end.npz"), os.path.join(LIB, c, "front_end.npz"))
        q, f = qc.get(c, {}), fq.get(c, {})
        qflags = [x for x in q.get("flags", "").split(";") if x]
        fflags = [x for x in f.get("fit_flags", "missing").split(";") if x]
        row = {k: info.get(k, "") for k in KEEP_CELLS}
        row.update(qc_flags=";".join(qflags), noise_repeat_ratio=q.get("noise_repeat_ratio", ""),
                   train_minutes=q.get("train_minutes", ""))
        row.update({k: f.get(k, "") for k in KEEP_FIT})
        row["fit_flags"] = ";".join(fflags)
        row["input_rescaling_k"] = round(p.get("input_rescaling_k", float("nan")), 4)
        row.update({k: round(p["branches"][k], 4) for k in KEY_PARAMS})
        row["use"] = int(not fflags and all(x == "no_noise_test" for x in qflags))
        rows.append(row)
    with open(os.path.join(LIB, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cells in the library, {sum(r['use'] for r in rows)} with use = 1 -> {LIB}")


if __name__ == "__main__":
    main()
