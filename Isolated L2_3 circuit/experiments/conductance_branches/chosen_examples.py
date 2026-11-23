"""Example continuous-prediction traces of the Chosen One (CNN-LSTM + conductance branches) on the
three protocols, NEURON vs the base fit (results_slow_k) and the first-ISI penalty fit (results_chosen_isi).
Run from "Isolated L2_3 circuit/":  python experiments/conductance_branches/chosen_examples.py
Writes to $CE_OUT (default /mnt/project-files/plots/chosen_one/)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb
sp, fs, base = sb.sp, sb.fs, sb.base
OUT = os.environ.get("CE_OUT", "/mnt/project-files/plots/chosen_one/")
os.makedirs(OUT, exist_ok=True)
FITS = [("results_slow_k", "Chosen One, before", "#95a5a6"), ("results_chosen_isi", "Chosen One, first-ISI penalty", "#27ae60")]
sets = fs.load_split()
test = {pr: sets[(pr, "test")] for pr in sp.PROTOS}
P = {d: np.load(os.path.join(HERE, d, "cnnlstm", "predictions.npz")) for d, _, _ in FITS}


def pick(pr, q):
    s = test[pr]; i = np.flatnonzero(s["n_spikes"] > 0)
    return int(i[np.argsort(s["n_spikes"][i])[int(q * (len(i) - 1))]])


sq = test["square"]; amp = sq["I"][:, 1000]
rows = [("square", pick("square", 0.5), None), ("square", int(np.flatnonzero(amp >= 180)[np.argsort(amp[amp >= 180])[len(np.flatnonzero(amp >= 180)) // 2]]), None),
        ("ramp", pick("ramp", 0.5), None), ("noise", pick("noise", 0.5), None), ("noise", pick("noise", 0.85), None)]


def gamma_one(gt, m, start):
    return sb.fs.spike_metrics(gt[None, start:], m[None, start:], 0)["gamma"]


fig, axes = plt.subplots(len(rows), 1, figsize=(13, 2.6 * len(rows)))
for ax, (pr, i, _) in zip(axes, rows):
    s = test[pr]; V = s["V"][i]; t = np.arange(len(V)) / 1000
    ax.axvspan(0, base.SEED_MS / 1000, color="#eeeeee", zorder=0)
    ax.plot(t, V, color="k", lw=1.3, label="NEURON")
    gt_s = sp.gt_masks(V[None])[0][0] > 0.5
    txt = []
    for d, lab, col in FITS:
        vm = P[d][f"{pr}_cont_vm"][i]; spk = P[d][f"{pr}_cont_spikes"][i]
        ax.plot(t, vm, color=col, lw=0.9, alpha=0.9, label=lab)
        txt.append(f"{lab.split(', ')[1]}: {int(spk[base.SEED_MS:].sum())} spikes")
    I = s["I"][i]
    lbl = f"{I[1000]:.0f} pA step" if pr == "square" else pr
    ax.set_title(f"{lbl} (test #{s['ids'][i]}): NEURON {int(gt_s[base.SEED_MS:].sum())} spikes | " + " | ".join(txt), fontsize=10, loc="left")
    ax.set_ylabel("Vm (mV)"); ax.spines[["top", "right"]].set_visible(False)
    if pr == "square": ax.set_xlim(0.3, 1.8)
axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
axes[-1].set_xlabel("time (s)")
fig.suptitle("Continuous prediction: NEURON seed for 450 ms (grey), then the model runs on its own. Model spikes drawn at +30 mV", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "examples_continuous.png"), dpi=150)

# onset close-ups of 4 strong steps
strong = np.flatnonzero(amp >= 180); sel = strong[np.argsort(amp[strong])][[0, len(strong) // 3, 2 * len(strong) // 3, -1]]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
for ax, i in zip(axes, sel):
    t = np.arange(sq["V"].shape[1])
    ax.plot(t, sq["V"][i], color="k", lw=1.3, label="NEURON")
    for d, lab, col in FITS:
        ax.plot(t, P[d]["square_cont_vm"][i], color=col, lw=1.0, label=lab)
    ax.set_xlim(480, 700); ax.set_title(f"{amp[i]:.0f} pA step onset", fontsize=10); ax.set_xlabel("time (ms)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Vm (mV)"); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Strong-step onsets: first ISI (NEURON ~52 ms)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "strong_step_onsets.png"), dpi=150)

# noise rasters, 10 traces
nz = test["noise"]; idx = np.flatnonzero(nz["n_spikes"] > 0)[:10]
fig, ax = plt.subplots(figsize=(13, 4.2))
for r, i in enumerate(idx):
    gt = np.flatnonzero(sp.gt_masks(nz["V"][i][None])[0][0] > 0.5)
    ax.vlines(gt / 1000, r * 3 + 0.05, r * 3 + 0.95, color="k", lw=1)
    for k, (d, lab, col) in enumerate(FITS):
        m = np.flatnonzero(P[d]["noise_cont_spikes"][i])
        ax.vlines(m / 1000, r * 3 + 1.05 + k, r * 3 + 1.95 + k, color=col, lw=1)
ax.axvspan(0, base.SEED_MS / 1000, color="#eeeeee", zorder=0)
ax.set_yticks([r * 3 + 1.5 for r in range(len(idx))]); ax.set_yticklabels([f"#{nz['ids'][i]}" for i in idx], fontsize=8)
ax.set_xlabel("time (s)"); ax.set_title("Noise test traces, continuous: NEURON (black), before (grey), first-ISI penalty (green)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "noise_raster.png"), dpi=150)
print("wrote", OUT)
