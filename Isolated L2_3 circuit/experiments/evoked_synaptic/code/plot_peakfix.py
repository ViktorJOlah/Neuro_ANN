"""EPSP-peak fix figure -> ../results_ampa/peak_fix.png and ../results_ampa/peak_fix.json"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results_ampa"); DD = os.path.join(HERE, "..", "data_ampa")
P, A = "487661754", "521409057"
OLD, NEW = "tcn64_all_sub", "tcn64_all_sub1_ew5_os400"
INK, GREY, C_OLD, C_NEW = "#222222", "#9a9a9a", "#e07b39", "#2a6fdb"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def mean_epsp(V, stim, test, vm, j):
    out_t, out_m = [], []
    for k, i in enumerate(test):
        for t in np.nonzero(stim[i, :, j] > 0)[0]:
            if t < 400 or t + 150 > V.shape[1]:
                continue
            w = stim[i, t - 100:t + 150].copy(); w[100, j] = 0
            if w.sum() == 0 and not (V[i, t - 20:t + 150] > -20).any():
                out_t.append(V[i, t - 10:t + 100] - V[i, t - 10:t].mean())
                out_m.append(vm[k, t - 10:t + 100] - vm[k, t - 10:t].mean())
    return np.mean(out_t, 0), np.mean(out_m, 0)


ABL = [("tcn64_all_sub", "PR 12 recipe (75% spike-free batches)"), ("tcn64_all_sub_ew5", "+ 5x loss weight 0-25 ms after pulses"),
       ("tcn64k_all_sub_ew5", "+ fast current-kernel path in D"), ("tcn64_all_sub_cont1600", "3x more steps (2400)"),
       ("tcn64_all_sub1_ew5", "100% spike-free batches + weight"), ("tcn64_all_sub_ew5_os400", "one-step warm-up (400) + weight"),
       ("tcn64_all_sub1_ew5_os400", "all three (new recipe)"), ("tcn64_all_sub1_ew5_os800", "all three, 800 warm-up + 400"),
       ("tcn64_all_sub1_ew20_os400", "all three, 20x weight")]
fig = plt.figure(figsize=(15, 6.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], wspace=0.95, hspace=0.55)
summary = {}
for r, cell in enumerate((P, A)):
    z = np.load(os.path.join(DD, cell, "evoked.npz"))
    for j, lab in ((1, "L2/3"), (0, "L1")):
        ax = fig.add_subplot(gs[r, 1 - j])
        tt = np.arange(-10, 100)
        for tag, c, name in ((OLD, C_OLD, "before"), (NEW, C_NEW, "after")):
            p = np.load(os.path.join(R, cell, tag, "test_predictions.npz"))
            gt, m = mean_epsp(z["V"], z["stim"], p["test"], p["vm"], j)
            if tag == OLD:
                ax.plot(tt, gt, color=INK, lw=1.8, label="NEURON")
            ax.plot(tt, m, color=c, lw=1.3, ls="--" if tag == OLD else "-", label=f"model, {name}")
        ax.axvline(64, color=GREY, lw=0.7, ls=":")
        ax.set_xlabel("ms after pulse"); ax.set_ylabel("mean EPSP (mV)")
        ax.set_title(f"{'abcd'[2 * r + 1 - j]}  {lab} pulse, cell {cell[-3:]}" + (" (Standard)" if cell == P else ""), loc="left")
        if r == 0 and j == 1:
            ax.legend(frameon=False, fontsize=7)
    summary[cell] = {t: json.load(open(os.path.join(R, cell, t, "peaks.json")))["summary"] for t in (OLD, NEW)}
ax = fig.add_subplot(gs[:, 2])
rows = [(lab, json.load(open(os.path.join(R, P, t, "peaks.json")))["summary"]) for t, lab in ABL]
rows += [(f"057: {lab}", json.load(open(os.path.join(R, A, t, "peaks.json")))["summary"]) for t, lab in ((OLD, "PR 12 recipe"), (NEW, "new recipe"))]
y = np.arange(len(rows)); y[-2:] += 1
ax.barh(y - 0.19, [s["L23"]["ratio"] for _, s in rows], 0.36, color=C_NEW, label="L2/3 pulses")
ax.barh(y + 0.19, [s["L1"]["ratio"] for _, s in rows], 0.36, color=GREY, label="L1 pulses")
ax.axvline(1, color=INK, lw=0.8)
ax.set_yticks(y, [l for l, _ in rows], fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 1.15)
ax.set_xlabel("mean EPSP peak, model / NEURON (held-out isolated pulses)")
ax.set_title("e  What closes the peak gap (cell 754 unless noted)", loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.08), ncol=2)
fig.savefig(os.path.join(R, "peak_fix.png"), dpi=150, bbox_inches="tight")
json.dump(summary, open(os.path.join(R, "peak_fix.json"), "w"), indent=1)
print("saved")
