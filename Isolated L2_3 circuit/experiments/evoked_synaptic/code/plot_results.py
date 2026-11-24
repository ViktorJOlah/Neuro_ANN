"""Summary figure for the evoked-synaptic feasibility test -> ../results/summary.png"""
SET = ('_' + __import__('os').environ['EVOKED_SET']) if __import__('os').environ.get('EVOKED_SET') else ''   # e.g. EVOKED_SET=ampa -> data_ampa/, results_ampa/
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results" + SET)
D = os.path.join(HERE, "..", "data" + SET)
A, B = "521409057", "513531471"
C1, C2, GREY, INK = "#2a6fdb", "#e07b39", "#9a9a9a", "#222222"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def met(cell, tag):
    f = os.path.join(R, cell, tag, "metrics.json")
    return json.load(open(f)) if os.path.exists(f) else None


fig = plt.figure(figsize=(14, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)

# (a) mean isolated evoked responses (strong pulses, no DC) per pathway and cell
ax = fig.add_subplot(gs[0, 0])
for cell, ls in ((A, "-"), (B, "--")):
    z = np.load(os.path.join(D, cell, "evoked.npz"))
    V, stim, dc = z["V"], z["stim"], z["dc"]
    for j, col, name in ((0, C1, "L1 (tuft)"), (1, C2, "L2/3 (proximal)")):
        rs = []
        for i in np.nonzero(dc == 0)[0]:
            for t in np.nonzero(stim[i, :, j] >= 0.4)[0]:
                if t < 300 or t + 400 > V.shape[1]:
                    continue
                w = stim[i, t - 300:t + 400].copy(); w[300, j] = 0
                if w.sum() == 0 and not (V[i, t - 20:t + 400] > -20).any():
                    rs.append(V[i, t:t + 400] - V[i, t - 20:t].mean())
        if rs:
            m = np.mean(rs, 0)
            ax.plot(m / m.max(), ls, color=col, lw=1.4, label=f"{name}, cell {cell[-3:]} (n={len(rs)})")
ax.axvline(64, color=GREY, lw=0.8, ls=":"); ax.text(66, 1.02, "64 ms", color=GREY)
ax.set_xlabel("ms after pulse"); ax.set_ylabel("somatic EPSP / peak")
ax.set_title("a  Evoked responses (NEURON, no DC)", loc="left")
ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.85)); ax.set_xlim(0, 520)

# (b) input-transform variants (cell A: all; cell B: 64 vs 256 ms)
ax = fig.add_subplot(gs[0, 1])
spec = [(A, "linear", "linear 256 ms"), (A, "tcn64", "CNN 64 ms"), (A, "tcn256", "CNN 256 ms"), (A, "gru", "GRU (recurrent)"),
        (B, "tcn64", "CNN 64 ms"), (B, "tcn256", "CNN 256 ms")]
rows = [(c, l, met(c, f"{k}_all")) for c, k, l in spec]
rows = [r for r in rows if r[2]]
y = np.arange(len(rows)) + np.array([0 if r[0] == A else 0.6 for r in rows])
ax.barh(y - 0.19, [r[2]["eval"]["rmse_L1_mV"] for r in rows], 0.36, color=C1, label="after L1 pulses")
ax.barh(y + 0.19, [r[2]["eval"]["rmse_L23_mV"] for r in rows], 0.36, color=C2, label="after L2/3 pulses")
ax.set_yticks(y, [f"{r[1]}  ({r[0][-3:]})" for r in rows]); ax.invert_yaxis()
for c, ls in ((A, "--"), (B, "-.")):
    m = met(c, "tcn256_all")
    if m:
        ax.axvline(m["eval"]["no_synaptic_input"]["rmse_away_mV"], color=GREY, ls=ls, lw=1, label=f"no synaptic input ({c[-3:]})")
o = met(A, "oracle")
if o:
    ax.axvline(o["eval"]["rmse_away_mV"], color=INK, ls=":", lw=1, label="best possible current into S (057)")
rc = met(A, "tcn256_all")["eval"].get("repeat_ceiling")
if rc:
    ax.axvline(rc["rmse_away_mV"], color=INK, lw=1, label="trial-to-trial (release noise, 057)")
ax.set_xlabel("Vm RMSE on held-out traces (mV)")
ax.set_title("b  Input transform D -> frozen somatic model S", loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

# (c) data size
ax = fig.add_subplot(gs[0, 2])
pts = [(m["train_minutes"], m["eval"]) for m in (met(A, "tcn256_2min"), met(A, "tcn256_5min"), met(A, "tcn256_all")) if m]
xs = [p[0] for p in pts]
ax.plot(xs, [p[1]["rmse_L1_mV"] for p in pts], "o-", color=C1, label="after L1 pulses")
ax.plot(xs, [p[1]["rmse_L23_mV"] for p in pts], "o-", color=C2, label="after L2/3 pulses")
ax.plot(xs, [p[1]["rmse_away_mV"] for p in pts], "o-", color=INK, label="all, away from spikes")
ax.set_ylim(0, None); ax.set_xlabel("evoked recording used for training (min)"); ax.set_ylabel("Vm RMSE (mV)")
ax.set_title("c  How much evoked data (CNN 256 ms)", loc="left"); ax.legend(frameon=False, fontsize=7)

# (d) swap test
sw = os.path.join(R, "swap_tcn256.json")
if os.path.exists(sw):
    s = json.load(open(sw))
    for k, (tag, d) in enumerate(s["chimeras"].items()):
        ax = fig.add_subplot(gs[1, k])
        keys = ["DA_SA", "DA_SB", "DB_SA", "DB_SB"]
        M = np.array([[d["scores"]["DA_SA"]["rmse_away_mV"], d["scores"]["DA_SB"]["rmse_away_mV"]],
                      [d["scores"]["DB_SA"]["rmse_away_mV"], d["scores"]["DB_SB"]["rmse_away_mV"]]])
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=8)
        exp = d["expected_best"]
        for i in range(2):
            for j in range(2):
                kk = keys[2 * i + j]
                ax.text(j, i, f"{M[i, j]:.1f} mV" + ("\n(expected best)" if kk == exp else ""),
                        ha="center", va="center", color="white" if M[i, j] > 4.5 else INK, fontsize=8)
        ax.set_xticks([0, 1], [f"S from {A[-3:]}", f"S from {B[-3:]}"]); ax.set_yticks([0, 1], [f"D from {A[-3:]}", f"D from {B[-3:]}"])
        ax.set_title(f"{'de'[k]}  Chimera: {d['dendrites_from'][-3:]} morphology+dendrites,\n    {d['soma_from'][-3:]} soma+axon channels", loc="left")

# (f) example prediction
ax = fig.add_subplot(gs[1, 2])
z = np.load(os.path.join(D, A, "evoked.npz"))
pl = {k: np.load(os.path.join(R, A, f"{k}_all", "test_predictions.npz")) for k in ("linear", "tcn256") if os.path.exists(os.path.join(R, A, f"{k}_all", "test_predictions.npz"))}
if pl:
    test = pl["tcn256"]["test"]; k = int(np.nonzero(test == 21)[0][0]) if 21 in test else 0
    i = test[k]; t = np.arange(z["V"].shape[1])
    sl = slice(300, 2300)
    ax.plot(t[sl], z["V"][i, sl], color=INK, lw=1, label="NEURON")
    ax.plot(t[sl], pl["linear"]["vm"][k, sl], color=GREY, lw=1, label="linear D")
    ax.plot(t[sl], pl["tcn256"]["vm"][k, sl], color=C1, lw=1, label="CNN 256 ms D")
    for j, col in ((0, C1), (1, C2)):
        ts = np.nonzero(z["stim"][i, sl, j])[0] + sl.start
        ax.plot(ts, np.full(len(ts), z["V"][i, sl].max() + 2 + 2 * j), "|", color=col, ms=6)
    ax.set_xlabel("ms"); ax.set_ylabel("Vm (mV)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title(f"f  Held-out Poisson trace, no DC (cell {A[-3:]})", loc="left")
fig.savefig(os.path.join(R, "summary.png"), dpi=150, bbox_inches="tight")
print("saved")
