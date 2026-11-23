"""Plots for the in-silico SK-blocker test: NEURON with gbar_SK = 0 vs the fitted model with its
SK edge zeroed (no retraining). Writes results/sk_block_counts.png and sk_block_traces.png."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb

MODELS = [("dense3", "3×32 dense → Izh"), ("cnnlstm", "CNN-LSTM → Izh"), ("dense_cnnlstm", "dense + CNN-LSTM → Izh")]
COL = {"dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}
PR = sb.sp.PROTOS
st = sb.base.SEED_MS
OUT = os.environ.get("SB_OUT", sb.OUT)
R = {m: json.load(open(os.path.join(OUT, m, "metrics.json")))["sk_block"] for m, _ in MODELS}
Z = {m: np.load(os.path.join(OUT, m, "predictions.npz")) for m, _ in MODELS}

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for j, pr in enumerate(PR):
    a = ax[0, j]
    xs = np.arange(len(MODELS) + 1)
    nc, nb = R["cnnlstm"][pr]["neuron_count_control"], R["cnnlstm"][pr]["neuron_count_block"]
    a.bar(xs[0] - 0.18, nc, 0.36, color="#444", label="control"); a.bar(xs[0] + 0.18, nb, 0.36, color="#bbb", label="SK removed")
    for i, (m, lab) in enumerate(MODELS):
        a.bar(xs[i + 1] - 0.18, R[m][pr]["model_count_control"], 0.36, color=COL[m])
        a.bar(xs[i + 1] + 0.18, R[m][pr]["model_count_block"], 0.36, color=COL[m], alpha=0.4)
    a.set_xticks(xs); a.set_xticklabels(["NEURON"] + [l.replace(" → Izh", "") for _, l in MODELS], fontsize=9)
    a.set_ylabel("spikes per test trace (after 450 ms)"); a.set_title(f"{pr}: solid = control, light = SK removed")
    if j == 0: a.legend(fontsize=8)
    b = ax[1, j]
    for m, lab in MODELS:
        S = {c: Z[m][f"skblock_{pr}_{c}_{k}_spikes"][:, st:].sum(1) for c in ("control", "sk_block") for k in ("model", "neuron")} if False else None
        dn = Z[m][f"skblock_{pr}_sk_block_neuron_spikes"][:, st:].sum(1) - Z[m][f"skblock_{pr}_control_neuron_spikes"][:, st:].sum(1)
        dm = Z[m][f"skblock_{pr}_sk_block_model_spikes"][:, st:].sum(1) - Z[m][f"skblock_{pr}_control_model_spikes"][:, st:].sum(1)
        b.scatter(dn, dm, s=10, color=COL[m], alpha=0.6, label=f"{lab} (r = {R[m][pr]['delta_count_corr']:.2f})")
    lim = [0, max(b.get_xlim()[1], b.get_ylim()[1])]
    b.plot(lim, lim, "k--", lw=0.8); b.set_xlabel("NEURON: extra spikes with SK removed"); b.set_ylabel("model: extra spikes with SK edge zeroed")
    b.set_title(f"{pr}: per-trace effect of removing SK"); b.legend(fontsize=7)
fig.suptitle("In-silico SK blocker: model predictions with the SK edge zeroed, no retraining", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sk_block_counts.png"), dpi=130)

# example traces: NEURON control / blocked, CNN-LSTM model control / blocked
z = np.load(os.path.join(sb.NREF, "test_square.npz"))
m = "cnnlstm"
nc = Z[m]["skblock_square_control_neuron_spikes"][:, st:].sum(1)
i = int(np.argsort(nc)[len(nc) // 2])
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
t = np.arange(z["control_v"].shape[1]) / 1000
for a, c, lab in ((ax[0], "control", "control"), (ax[1], "sk_block", "SK removed")):
    a.plot(t, z[f"{c}_v"][i], "k", lw=0.6, label=f"NEURON, {lab}")
    for tt in np.flatnonzero(Z[m][f"skblock_square_{c}_model_spikes"][i]) / 1000:
        a.axvline(tt, ymin=0.92, ymax=1.0, color=COL[m], lw=1.2)
    a.set_ylabel("Vm (mV)"); a.set_title(f"{lab}: NEURON trace (black), CNN-LSTM → Izh spike times (green ticks)")
ax[1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sk_block_traces.png"), dpi=130)
