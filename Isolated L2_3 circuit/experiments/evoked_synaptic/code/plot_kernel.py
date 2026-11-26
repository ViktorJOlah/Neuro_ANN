"""Synaptic kernel layer figure -> ../results_ampa/kernel_layer.png and ../results_ampa/kernel_layer.json"""
import json
import os
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import shape_metrics as SM  # noqa: E402

R, DD = os.path.join(HERE, "..", "results_ampa"), os.path.join(HERE, "..", "data_ampa")
CELLS = ("487661754", "521409057")
MODELS = [("tcn64_all_sub1_ew5_os400", "Standard + tcn64 D (current recipe)", "#e07b39", "-"),
          ("kernel_clean_k2s", "kernel layer, clean", "#2a6fdb", "-"),
          ("kernel_soft_k2s", "kernel layer, soft (ANN event gain)", "#2a9d5c", "--")]
ROUND1 = [("kernel_clean", "kernel, clean, 1 component"), ("kernel_soft", "kernel, soft, 1 component")]
INK, GREY = "#222222", "#9a9a9a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def mean_epsp(V, stim, test, vm, j):
    g, m = [], []
    for k, i, jj, t in SM.iso_events(V, stim, test):
        if jj == j:
            g.append(V[i, t - 10:t + 140] - V[i, t - 10:t].mean())
            m.append(np.minimum(vm[k, t - 10:t + 140], -20) - vm[k, t - 10:t].mean())
    return np.mean(g, 0), np.mean(m, 0)


fig = plt.figure(figsize=(16, 8.4))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.6, 1.15], wspace=0.42, hspace=0.5)
summary = {}
tt = np.arange(-10, 140)
for r, cell in enumerate(CELLS):
    z = np.load(os.path.join(DD, cell, "evoked.npz"))
    V, st = z["V"], z["stim"]
    P = {tag: np.load(os.path.join(R, cell, tag, "test_predictions.npz")) for tag, *_ in MODELS}
    for col, (j, lab) in enumerate(((1, "L2/3"), (0, "L1"))):
        ax = fig.add_subplot(gs[r, col])
        for n, (tag, name, c, ls) in enumerate(MODELS):
            g, m = mean_epsp(V, st, P[tag]["test"], P[tag]["vm"], j)
            if n == 0:
                ax.plot(tt, g, color=INK, lw=2, label="NEURON")
            ax.plot(tt, m, color=c, lw=1.3, ls=ls, label=name)
        ax.axvline(64, color=GREY, lw=0.7, ls=":")
        ax.set_xlabel("ms after pulse"); ax.set_ylabel("mean isolated EPSP (mV)")
        ax.set_title(f"{'ab'[col] if r == 0 else 'de'[col]}  {lab} pulse, cell {cell[-3:]}" + (" (Standard)" if r == 0 else ""), loc="left")
        if r == 0 and col == 0:
            ax.legend(frameon=False, fontsize=7, loc="upper right")
    # close-up of a Poisson stretch: where the jagged look shows
    ax = fig.add_subplot(gs[r, 2])
    test = P[MODELS[0][0]]["test"]
    k = [n for n, i in enumerate(test) if z["kind"][i] == "poisson" and z["dc"][i] == 0][0]
    i = test[k]
    a, b = 1300, 1900
    x = np.arange(a, b)
    ax.plot(x, V[i, a:b], color=INK, lw=1.8, label="NEURON")
    for tag, name, c, ls in MODELS:
        ax.plot(x, np.minimum(P[tag]["vm"][k, a:b], -20), color=c, lw=1.1, ls=ls)
    lo = V[i, a:b].min() - 0.8
    for j, c in ((0, GREY), (1, INK)):
        for t in np.nonzero(st[i, a:b, j])[0]:
            ax.plot([a + t, a + t], [lo, lo + 0.25 + 0.5 * st[i, a + t, j]], color=c, lw=1)
    ax.set_xlabel("ms (held-out trace; ticks: L2/3 pulses black, L1 grey)"); ax.set_ylabel("Vm (mV)")
    ax.set_title(f"{'c' if r == 0 else 'f'}  Poisson stimulation, cell {cell[-3:]}", loc="left")
    summary[cell] = {}
    for tag in [m[0] for m in MODELS] + [m[0] for m in ROUND1]:
        m = json.load(open(os.path.join(R, cell, tag, "metrics.json")))["eval"]
        s = json.load(open(os.path.join(R, cell, tag, "shape.json")))["summary"]
        summary[cell][tag] = dict(rmse_away=m["rmse_away_mV"], rmse_L1=m["rmse_L1_mV"], rmse_L23=m["rmse_L23_mV"],
                                  gamma=m["gamma"], n_true=m["n_true"], n_model=m["n_model"], count_abs_err=m["count_abs_err"],
                                  **{f"{p}_{q}": s[p][q] for p in ("L1", "L23") for q in ("ratio", "tpk", "tpk_model", "shape_rmse", "shape_rmse_norm", "extra_peaks", "pre_dip")},
                                  spurious_extrema_per_s=s["trace"]["spurious_extrema_per_s"], rough_ratio=s["trace"]["rough_ratio"])
# metric panel
ax = fig.add_subplot(gs[:, 3]); ax.axis("off")
rows = [("Vm RMSE away from spikes (mV)", "rmse_away", 2), ("L2/3 peak, model/NEURON", "L23_ratio", 2),
        ("L1 peak, model/NEURON", "L1_ratio", 2), ("L2/3 shape error, peak-normalised", "L23_shape_rmse_norm", 3),
        ("L1 shape error, peak-normalised", "L1_shape_rmse_norm", 3), ("dip before EPSP, L2/3 (mV)", "L23_pre_dip", 2),
        ("dip before EPSP, L1 (mV)", "L1_pre_dip", 2), ("spurious wiggles per s", "spurious_extrema_per_s", 1),
        ("spikes, model (NEURON)", None, 0)]
txt = "g  Held-out evoked traces\n\n"
for cell in CELLS:
    txt += f"cell {cell[-3:]}            tcn64 D   clean   soft\n"
    for lab, key, nd in rows:
        if key is None:
            vals = [f"{summary[cell][t]['n_model']}" for t, *_ in MODELS]
            txt += f"  {lab:<33s}" + "".join(f"{v:>8s}" for v in vals) + f"  ({summary[cell][MODELS[0][0]]['n_true']})\n"
        else:
            txt += f"  {lab:<33s}" + "".join(f"{summary[cell][t][key]:8.{nd}f}" for t, *_ in MODELS) + "\n"
    txt += "\n"
ax.text(0, 1, txt, family="monospace", fontsize=6.9, va="top")
fig.savefig(os.path.join(R, "kernel_layer.png"), dpi=150, bbox_inches="tight")
json.dump(summary, open(os.path.join(R, "kernel_layer.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
