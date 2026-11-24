"""AMPA-only rerun figure -> ../results_ampa/summary_ampa.png"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results_ampa"); D = os.path.join(HERE, "..", "data_ampa")
P, A = "487661754", "521409057"
C1, C2, GREY, INK = "#2a6fdb", "#e07b39", "#9a9a9a", "#222222"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
met = lambda c, t: json.load(open(os.path.join(R, c, t, "metrics.json")))

def evoked(cell, V, stim, j, src):
    out = []
    for k, i in enumerate(src[0]):
        for t in np.nonzero(stim[i, :, j] >= 0.4)[0]:
            if t < 400 or t + 150 > V.shape[1]: continue
            w = stim[i, t - 100:t + 150].copy(); w[100, j] = 0
            if w.sum() == 0 and not (V[i, t - 20:t + 150] > -20).any():
                x = src[1][k] if src[1] is not None else V[i]
                out.append(x[t:t + 150] - x[t - 10:t].mean())
    return np.mean(out, 0)

fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw=dict(wspace=0.55))
for col, cell in enumerate((P, A)):
    ax = axs[col]
    z = np.load(os.path.join(D, cell, "evoked.npz")); V, stim = z["V"], z["stim"]
    for m, ls in (("tcn64_all_sub", "--"), ("tcn256_all_sub", ":")):
        p = np.load(os.path.join(R, cell, m, "test_predictions.npz"))
        for j, c in ((0, C1), (1, C2)):
            if m == "tcn64_all_sub":
                ax.plot(evoked(cell, V, stim, j, (p["test"], None)), "-", color=c, lw=1.6, label=f"NEURON, {'L1' if j == 0 else 'L2/3'} pulse")
            ax.plot(evoked(cell, V, stim, j, (p["test"], p["vm"])), ls, color=c, lw=1.1, label=f"model, CNN {m[3:-8]} ms" if j == 0 else None)
    ax.axvline(64, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("ms after pulse"); ax.set_ylabel("somatic EPSP (mV)")
    ax.set_title(f"{'ab'[col]}  Mean held-out EPSP, cell {cell[-3:]}" + (" (Standard)" if cell == P else ""), loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
ax = axs[2]
rows = [(c, lab, met(c, t)["eval"]) for c in (P, A) for t, lab in (("linear_all", "linear 256 ms"), ("tcn64_all_sub", "CNN 64 ms"), ("tcn256_all_sub", "CNN 256 ms")) if os.path.exists(os.path.join(R, c, t, "metrics.json"))]
y = np.arange(len(rows)) + np.array([0 if r[0] == P else 0.6 for r in rows])
ax.barh(y - 0.19, [r[2]["rmse_L1_mV"] for r in rows], 0.36, color=C1, label="after L1 pulses")
ax.barh(y + 0.19, [r[2]["rmse_L23_mV"] for r in rows], 0.36, color=C2, label="after L2/3 pulses")
ax.set_yticks(y, [f"{r[1]} ({r[0][-3:]})" for r in rows]); ax.invert_yaxis()
for c, ls in ((P, "--"), (A, "-.")):
    e = met(c, "tcn64_all_sub")["eval"]
    ax.axvline(e["no_synaptic_input"]["rmse_away_mV"], color=GREY, ls=ls, lw=1, label=f"no synaptic input ({c[-3:]})")
    ax.axvline(e["repeat_ceiling"]["rmse_away_mV"], color=INK, ls=ls, lw=1, label=f"release noise ({c[-3:]})")
ax.set_xlabel("Vm RMSE, held-out traces (mV)"); ax.set_title("c  64 vs 256 ms (AMPA only)", loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.savefig(os.path.join(R, "summary_ampa.png"), dpi=150, bbox_inches="tight"); print("saved")
