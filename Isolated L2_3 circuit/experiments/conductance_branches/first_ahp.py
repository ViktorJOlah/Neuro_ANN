"""First-AHP check: deepest Vm in the 60 ms after the first spike in continuous prediction vs ground truth,
for each result set (original fit, SK pinned, fine-tuned). Writes first_ahp.png, first_ahp_summary.png and
first_ahp.json into results_first_ahp/."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb

MODELS = [("dense3", "3×32 dense"), ("cnnlstm", "CNN-LSTM"), ("dense_cnnlstm", "dense + CNN-LSTM")]
COL = {"dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}
SETS = [("results", "original fit"), ("results_sk_pinned", "SK pinned"), ("results_finetune", "SK pinned + ANN fine-tune")]
SETS = [(d, l) for d, l in SETS if all(os.path.exists(os.path.join(HERE, d, m, "predictions.npz")) for m, _ in MODELS)]
OUT = os.path.join(HERE, "results_first_ahp"); os.makedirs(OUT, exist_ok=True)
PRE, POST = 10, 150

WIN = 60
def trough(v, spk, f, nxt):
    """min Vm in the WIN ms after the first spike, skipping spike samples (catches the AHP after an onset doublet)"""
    end = f + WIN
    w = v[f + 2:end][~spk[f + 2:end]]
    return float(w.min()) if w.size else np.nan

def first_two(onsets):
    k = np.flatnonzero(onsets)
    return (k[0] if k.size else None), (k[1] if k.size > 1 else None)

summary = {}
for pr in ("square", "ramp", "noise"):
    s = sb.fs.load_split()[(pr, "test")]
    V = s["V"]; S, M = sb.sp.gt_masks(V); S, M = S.astype(bool), M.astype(bool)
    rows = np.flatnonzero(S.any(1))
    gt = {}
    for i in rows:
        f, n = first_two(S[i]); gt[i] = (f, trough(V[i], M[i] | S[i], f, n))
    summary[pr] = {"gt_trough_mean": float(np.nanmean([g[1] for g in gt.values()]))}
    for d, lab in SETS:
        for m, _ in MODELS:
            z = np.load(os.path.join(HERE, d, m, "predictions.npz"))
            vm, sp_ = z[f"{pr}_cont_vm"], z[f"{pr}_cont_spikes"].astype(bool)
            err, tr = [], []
            for i in rows:
                f, n = first_two(sp_[i])
                if f is None: continue
                t = trough(vm[i], sp_[i], f, n); tr.append(t); err.append(t - gt[i][1])
            summary[pr][f"{d}/{m}"] = {"trough_mean": float(np.nanmean(tr)), "trough_minus_gt": float(np.nanmean(err)),
                                       "abs_err": float(np.nanmean(np.abs(err))), "n": len(err)}
json.dump(summary, open(os.path.join(OUT, "first_ahp.json"), "w"), indent=1)

# close-up on square steps of 4 amplitudes, aligned on each trace's own first spike
s = sb.fs.load_split()[("square", "test")]
V = s["V"]; S, M = sb.sp.gt_masks(V); S = S.astype(bool)
amp = s["I"][:, 1000]; idx = np.flatnonzero(S.any(1)); idx = idx[np.argsort(amp[idx])]
pick = idx[[int(q * (len(idx) - 1)) for q in (0.1, 0.4, 0.7, 1.0)]]
t = np.arange(-PRE, POST)
fig, ax = plt.subplots(len(pick), len(SETS), figsize=(5.2 * len(SETS), 2.6 * len(pick)), sharey=True, squeeze=False)
for c, (d, lab) in enumerate(SETS):
    Z = {m: np.load(os.path.join(HERE, d, m, "predictions.npz")) for m, _ in MODELS}
    for r, i in enumerate(pick):
        a = ax[r, c]; f = np.flatnonzero(S[i])[0]
        a.plot(t, V[i, f - PRE:f + POST], "k", lw=1.6, label="NEURON (GT)")
        for m, ml in MODELS:
            sp_ = Z[m]["square_cont_spikes"][i].astype(bool)
            if not sp_.any(): continue
            g = np.flatnonzero(sp_)[0]; seg = Z[m]["square_cont_vm"][i, g - PRE:g + POST]
            a.plot(t[:len(seg)], seg, color=COL[m], lw=1.1, label=ml)
        a.axhline(-50, color="grey", ls=":", lw=0.8)
        a.set_ylim(-75, 20); a.grid(alpha=0.25)
        if r == 0: a.set_title(lab)
        if c == 0: a.set_ylabel(f"{amp[i]:.0f} pA step\nVm (mV)")
        if r == len(pick) - 1: a.set_xlabel("ms from first spike (each trace aligned on its own)")
ax[0, 0].legend(fontsize=7, loc="upper right")
fig.suptitle("First AHP, continuous prediction, square test steps")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "first_ahp.png"), dpi=130)

fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
w = 0.8 / len(MODELS)
for j, pr in enumerate(("square", "ramp", "noise")):
    for k, (m, ml) in enumerate(MODELS):
        y = [summary[pr][f"{d}/{m}"]["trough_minus_gt"] for d, _ in SETS]
        ax[j].bar(np.arange(len(SETS)) + (k - 1) * w, y, w, color=COL[m], label=ml)
    ax[j].axhline(0, color="k", lw=0.8); ax[j].set_xticks(range(len(SETS))); ax[j].set_xticklabels([l for _, l in SETS], fontsize=8)
    ax[j].set_title(f"{pr}: GT trough {summary[pr]['gt_trough_mean']:.1f} mV"); ax[j].grid(alpha=0.25, axis="y")
ax[0].set_ylabel("first-AHP trough, model − GT (mV)\n(negative = too deep)"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "first_ahp_summary.png"), dpi=130)
print(json.dumps(summary, indent=1))
