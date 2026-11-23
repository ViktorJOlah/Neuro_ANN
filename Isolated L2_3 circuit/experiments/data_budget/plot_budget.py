"""Learning curves and examples for the data-budget fits (fit_budget.py).

Run from "Isolated L2_3 circuit/":
    python experiments/data_budget/plot_budget.py [--results experiments/data_budget/results]

Writes into the results folder:
  learning_curves.png   Γ, spike-count error and Vm RMSE vs training minutes, per protocol, one line per cell
  noise_cost.png        the same scored against clean vs noisy held-out traces (Vm RMSE)
  examples_<cell>.png   held-out test traces: clean NEURON vs the model at 0 min (Standard), 5 min and full
  parameters.png        Izhikevich and spike parameters vs training minutes
  summary.csv           one row per cell x budget x protocol
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "simulations", "additional_cells", "data")
PROTOS = ("square", "noise", "ramp")
BUDGETS = ("0min", "5min", "10min", "20min", "30min", "full")
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


SEED_MS = 450


def spikes(v, thr=-20.0):
    return np.flatnonzero((v[1:] >= thr) & (v[:-1] < thr)) + 1


def interspike_rmse(vm, ref):
    """Vm RMSE away from spikes: after the seed, excluding 5 ms before to 30 ms after every spike
    of either the model or the reference trace. Works for traces that spike (noise, ramp)."""
    err, n = 0.0, 0
    for a, b in zip(vm, ref):
        keep = np.ones(len(b), bool)
        keep[:SEED_MS] = False
        for s in np.concatenate([spikes(a), spikes(b)]):
            keep[max(0, s - 5):s + 30] = False
        err += float(np.sum((a[keep] - b[keep]) ** 2))
        n += int(keep.sum())
    return float(np.sqrt(err / n)) if n else np.nan


def add_interspike(res, R):
    """Adds eval[p][scored]['interspike_rmse_mV'] from the saved test predictions."""
    cache = {}
    for (c, b), m in R.items():
        z = np.load(os.path.join(res, c, b, "test_predictions.npz"))
        for p in PROTOS:
            if (c, p) not in cache:
                cache[(c, p)] = (np.load(os.path.join(DATA, c, f"{p}_noisy.npy"), mmap_mode="r"),
                                 np.load(os.path.join(DATA, c, f"{p}.npy"), mmap_mode="r"))
            noisy, clean = cache[(c, p)]
            idx = z[f"{p}_idx"]
            for scored, arr in (("noisy", noisy), ("clean", clean)):
                m["eval"][p][scored]["interspike_rmse_mV"] = interspike_rmse(z[f"{p}_vm"], np.asarray(arr[idx, :, 1]))


def load(res):
    out = {}
    for cell in sorted(os.listdir(res)):
        if not os.path.isdir(os.path.join(res, cell)):
            continue
        for b in BUDGETS:
            f = os.path.join(res, cell, b, "metrics.json")
            if os.path.exists(f):
                out[(cell, b)] = json.load(open(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    R = load(a.results)
    cells = sorted({c for c, _ in R})
    add_interspike(a.results, R)

    # summary table
    rows = []
    for (c, b), m in sorted(R.items(), key=lambda kv: (kv[0][0], BUDGETS.index(kv[0][1]))):
        for p in PROTOS:
            for scored in ("noisy", "clean"):
                e = m["eval"][p][scored]
                rows.append(dict(cell=c, budget=b, train_minutes=round(m["train_minutes"], 2), protocol=p, scored_on=scored,
                                 gamma=e["gamma"], count_abs_err=e["count_abs_err"], count_bias=e["count_bias"],
                                 count_r=e["count_r"], spike_free_rmse_mV=e["spike_free_rmse_mV"],
                                 clipped_rmse_mV=e["clipped_rmse_mV"], interspike_rmse_mV=e["interspike_rmse_mV"]))
    with open(os.path.join(a.results, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def series(c, p, key, scored="noisy"):
        xs, ys = [], []
        for b in BUDGETS:
            if (c, b) in R:
                xs.append(R[(c, b)]["train_minutes"])
                ys.append(R[(c, b)]["eval"][p][scored][key])
        return np.array(xs), np.array(ys, float)

    # learning curves: rows = protocols, cols = metrics
    metrics = (("gamma", "Γ (±4 ms)"), ("count_abs_err", "|spike-count error| per trace"),
               ("clipped_rmse_mV", "Vm RMSE, clipped at -20 mV (mV)"), ("interspike_rmse_mV", "Vm RMSE away from spikes (mV)"))
    fig, ax = plt.subplots(3, 4, figsize=(17, 10.5))
    for r, p in enumerate(PROTOS):
        for col, (key, lab) in enumerate(metrics):
            A = ax[r, col]
            for k, c in enumerate(cells):
                x, y = series(c, p, key)
                A.plot(x, y, "-o", color=COLORS[k], ms=4, label=c)
                if len(x) and x[-1] > 0:
                    A.plot(x[-1], y[-1], "s", color=COLORS[k], ms=7, mfc="none")
            for v in (5, 10, 20, 30):
                A.axvline(v, color="0.88", lw=0.8, zorder=0)
            A.set_title(f"{p}: {lab}", fontsize=9)
            if key in ("clipped_rmse_mV", "interspike_rmse_mV"):
                A.set_yscale("log")
            elif key == "count_abs_err":
                A.set_yscale("symlog", linthresh=1)
            if r == 2:
                A.set_xlabel("training data (min; 0 = Standard unfitted, square = full pool)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Standard model fitted to 5 Allen L2/3 PCs on growing data budgets (noisy training data, 45 held-out test traces per cell)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "learning_curves.png"), dpi=120)
    plt.close(fig)

    # noise cost: clean vs noisy scoring of the Vm RMSE
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for col, p in enumerate(PROTOS):
        for k, c in enumerate(cells):
            for scored, ls in (("noisy", "-"), ("clean", "--")):
                x, y = series(c, p, "clipped_rmse_mV", scored)
                ax[col].plot(x, y, ls, marker="o", ms=3, color=COLORS[k], label=f"{c} ({scored})" if col == 0 else None)
        ax[col].set_yscale("log")
        ax[col].set_title(f"{p}: Vm RMSE vs noisy (solid) and clean (dashed) test traces", fontsize=9)
        ax[col].set_xlabel("training data (min)")
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "noise_cost.png"), dpi=120)
    plt.close(fig)

    # parameters
    keys = (("izhikevich", "a"), ("izhikevich", "b"), ("izhikevich", "time_scale"), ("izhikevich", "g"),
            ("izhikevich", "I0"), ("izhikevich", "V0"), ("izhikevich", "Vs"),
            ("branches", "V_th"), ("branches", "c_mV"), ("branches", "d"), ("branches", "b_na"), ("branches", "ln_s_sk"))
    fig, ax = plt.subplots(3, 4, figsize=(16, 9))
    for j, (grp, key) in enumerate(keys):
        A = ax.flat[j]
        for k, c in enumerate(cells):
            xs = [R[(c, b)]["train_minutes"] for b in BUDGETS if (c, b) in R]
            ys = [R[(c, b)][grp][key] for b in BUDGETS if (c, b) in R]
            A.plot(xs, ys, "-o", ms=3, color=COLORS[k], label=c)
        A.set_title(key, fontsize=9)
    ax.flat[0].legend(fontsize=7)
    for A in ax[-1]:
        A.set_xlabel("training data (min)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "parameters.png"), dpi=110)
    plt.close(fig)

    # examples per cell
    show = [b for b in ("0min", "5min", "full")]
    for c in cells:
        have = [b for b in show if (c, b) in R]
        if not have:
            continue
        fig, ax = plt.subplots(len(have), 3, figsize=(17, 2.8 * len(have)), squeeze=False)
        for col, p in enumerate(PROTOS):
            clean = np.load(os.path.join(DATA, c, f"{p}.npy"), mmap_mode="r")
            for r, b in enumerate(have):
                z = np.load(os.path.join(a.results, c, b, "test_predictions.npz"))
                idx = z[f"{p}_idx"]
                n_true = np.array(R[(c, b)]["eval"][p]["clean"]["n_true"])
                j = int(np.argmin(np.abs(n_true - np.percentile(n_true, 70))))
                t = np.arange(clean.shape[1]) / 1000.0
                A = ax[r, col]
                A.plot(t, clean[idx[j], :, 1], "k", lw=0.6, label="NEURON (clean)")
                A.plot(t, z[f"{p}_vm"][j], color="#d95f02", lw=0.6, alpha=0.85, label="model")
                A.axvline(0.45, color="0.6", lw=0.8, ls=":")
                e = R[(c, b)]["eval"][p]["clean"]
                g = e["gamma_per_trace"][j]
                A.set_title(f"{b} ({R[(c, b)]['train_minutes']:.1f} min)  {p} test #{idx[j]}  spikes {n_true[j]} → "
                            f"{e['n_model'][j]}  Γ {'–' if g is None else f'{g:.2f}'}", fontsize=8)
                A.set_ylim(-95, 45)
        ax[0, 0].legend(fontsize=7, loc="upper left")
        for A in ax[-1]:
            A.set_xlabel("time (s); dotted line = end of the recorded-Vm seed")
        fig.suptitle(f"cell {c}: held-out test traces", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.results, f"examples_{c}.png"), dpi=110)
        plt.close(fig)
    print("wrote plots for", cells)


if __name__ == "__main__":
    main()
