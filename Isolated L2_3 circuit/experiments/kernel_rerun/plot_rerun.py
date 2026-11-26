"""Figures for the kernel-layer rerun: step 0 (membrane correction from S alone), step 2 (rates vs in vivo),
step 4 (evoked data amount). -> results/fig_step0_calibration.png, fig_step2_invivo.png, fig_step4_data_amount.png"""
import glob
import json
import os

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
R = os.path.join(HERE, "results"); CAL = os.path.join(EXP, "kernel_calibration", "results")
EV = os.path.join(EXP, "evoked_synaptic", "results_ampa")
INK, GREY, BLUE, ORANGE, GREEN = "#222222", "#9a9a9a", "#2a6fdb", "#e07b39", "#2a9d5c"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- step 0
cal = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in sorted(glob.glob(os.path.join(CAL, "*", "calibration.json")))}
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), gridspec_kw=dict(width_ratios=[2.2, 1, 1, 1]), constrained_layout=True)
z = np.load(os.path.join(CAL, "487661754", "test_traces.npz"))
t = np.arange(300, 1200)
ax[0].plot(t, z["V_target"][0, 300:], color=INK, lw=1.6, label="S, current through front end (target)")
ax[0].plot(t, np.minimum(z["vm_none"][0, 300:], -20), color=ORANGE, lw=1, label="kernel path, no correction")
ax[0].plot(t, np.minimum(z["vm_fit"][0, 300:], -20), color=BLUE, lw=1, label="kernel path, calibrated correction")
ax[0].set_xlabel("ms (held-out PSC train, cell 754)"); ax[0].set_ylabel("Vm (mV)"); ax[0].legend(frameon=False, fontsize=7)
ax[0].set_title("a  Calibration target vs bypass path", loc="left")
a = np.array([c["test_no_correction"]["rmse_mV"] for c in cal.values()]); b = np.array([c["test"]["rmse_mV"] for c in cal.values()])
ax[1].scatter(a, b, s=12, color=BLUE); m = max(a.max(), b.max()) * 1.05
ax[1].plot([0, m], [0, m], color=GREY, lw=0.8, ls=":"); ax[1].set_xlabel("RMSE, no correction (mV)"); ax[1].set_ylabel("RMSE, calibrated (mV)")
ax[1].set_title(f"b  {len(cal)} models, median {np.median(a):.2f} → {np.median(b):.2f}", loc="left")
ks = ["k0", "c1", "c2"]
ax[2].boxplot([[c["correction"][k] for c in cal.values()] for k in ks], tick_labels=ks)
ax[2].axhline(0, color=GREY, lw=0.6); ax[2].set_title("c  correction weights", loc="left")
ax[3].boxplot([[c["correction"][k] for c in cal.values()] for k in ("tau1", "tau2")], tick_labels=["τ1", "τ2"])
ax[3].set_ylabel("ms"); ax[3].set_title("d  low-pass time constants", loc="left")
fig.savefig(os.path.join(R, "fig_step0_calibration.png"), dpi=150)
json.dump(dict(n=len(cal), median_rmse_none=float(np.median(a)), median_rmse_fit=float(np.median(b)), n_improved=int((b < a - 1e-6).sum()),
               n_identity=int(sum(c["best_iter"] < 0 for c in cal.values())),
               params={k: np.percentile([c["correction"][k] for c in cal.values()], [10, 50, 90]).tolist() for k in ("k0", "c1", "tau1", "c2", "tau2")}),
          open(os.path.join(R, "step0_summary.json"), "w"), indent=1)

# ---- step 2
zr = np.load(os.path.join(R, "step2_invivo", "rates.npz")); st = json.load(open(os.path.join(R, "step2_invivo", "invivo_stats.json")))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
cols = {"in_vivo": INK, "GLIF_network": GREY, "ANN_PCs,_current_path": ORANGE, "ANN_PCs,_kernel_layer": BLUE}
mu = zr["in_vivo"].mean()
for k, c in cols.items():
    r = np.sort(zr[k] * (mu / zr[k].mean()))[::-1]
    ax[0].plot(np.arange(1, len(r) + 1) / len(r), r, color=c, lw=2 if k == "in_vivo" else 1.2, label=k.replace("_", " "))
    cs = np.cumsum(r) / r.sum(); ax[1].plot(np.arange(1, len(r) + 1) / len(r), cs, color=c, lw=2 if k == "in_vivo" else 1.2)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("rank fraction"); ax[0].set_ylabel("rate (Hz, matched mean)")
ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("a  Rank-ordered rates, L2/3 PCs", loc="left")
ax[1].axvline(0.1, color=GREY, lw=0.6, ls=":"); ax[1].set_xlabel("fraction of cells (most active first)"); ax[1].set_ylabel("share of spikes")
ax[1].set_title("b  Top-10% share: in vivo {:.2f}, kernel ANN {:.2f}".format(st["in vivo (Neuropixels L2/3 RS)"]["top10_share"], st["ANN PCs, kernel layer"]["top10_share"]), loc="left")
fig.savefig(os.path.join(R, "fig_step2_invivo.png"), dpi=150)

# ---- step 4
tags = [("kernel_soft_corr_2min", "2 min"), ("kernel_soft_corr_5min", "5 min"), ("kernel_soft_corr_noinit", "full")]
out = {}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
for cell, c in (("487661754", BLUE), ("521409057", ORANGE)):
    rows = []
    for tg, lab in tags:
        m = json.load(open(os.path.join(EV, cell, tg, "metrics.json")))["eval"]; s = json.load(open(os.path.join(EV, cell, tg, "shape.json")))["summary"]
        rows.append(dict(budget=lab, rmse_away=m["rmse_away_mV"], L23_ratio=s["L23"]["ratio"], L1_ratio=s["L1"]["ratio"],
                         L23_shape=s["L23"]["shape_rmse_norm"], L1_shape=s["L1"]["shape_rmse_norm"], spurious=s["trace"]["spurious_extrema_per_s"]))
    out[cell] = rows
    x = np.arange(len(tags))
    ax[0].plot(x, [r["rmse_away"] for r in rows], "o-", color=c, label=f"cell {cell[-3:]}")
    ax[1].plot(x, [r["L23_ratio"] for r in rows], "o-", color=c); ax[1].plot(x, [r["L1_ratio"] for r in rows], "s--", color=c)
    ax[2].plot(x, [r["L23_shape"] for r in rows], "o-", color=c); ax[2].plot(x, [r["L1_shape"] for r in rows], "s--", color=c)
for a_, t_ in zip(ax, ("a  Vm RMSE away from spikes (mV)", "b  peak model/NEURON (● L2/3, ■ L1)", "c  peak-normalised shape error")):
    a_.set_xticks(range(len(tags))); a_.set_xticklabels([t[1] for t in tags]); a_.set_xlabel("evoked training data"); a_.set_title(t_, loc="left")
ax[1].axhline(1, color=GREY, lw=0.6); ax[0].legend(frameon=False)
fig.savefig(os.path.join(R, "fig_step4_data_amount.png"), dpi=150)
json.dump(out, open(os.path.join(R, "step4_summary.json"), "w"), indent=1)
print("ok")
