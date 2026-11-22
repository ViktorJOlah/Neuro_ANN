"""Comparison plots for the spiking fit (threshold, c and d on the frozen subthreshold models).

Run from "Isolated L2_3 circuit/" after spiking.py has run for all four models:
    python experiments/spiking/compare.py

Writes to results/:
  comparison_continuous.png, comparison_one_step.png  traces vs ground truth (+ clipped error)
  error_bars.png       Vm RMSE with both traces clipped at -40 mV, one-step and continuous
  spike_metrics.png    coincidence factor Gamma (+-4 ms), Allen explained variance (10 ms
                       Gaussian), spike-count error and bias, continuous and one-step
  raster_continuous.png  spike rasters of 8 noise test traces, ground truth vs each model
  scatter_500_points.png  500 random test points, Vm clipped at -40 mV
  summary.json
All numbers are on held-out test traces (square, noise and ramp), spiking or not.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking as sp  # noqa: E402

fs, base = sp.fs, sp.base
OUT = os.environ.get("SPIKING_OUT", sp.OUT)
MODELS = [("izh", "Izhikevich alone"), ("dense3", "3×32 dense → Izh"), ("cnnlstm", "CNN-LSTM → Izh"),
          ("dense_cnnlstm", "dense + CNN-LSTM → Izh")]
COLORS = {"izh": "#e67e22", "dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}
PL = {"square": "square", "noise": "noise", "ramp": "ramp"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = fs.load_split()
    test = {pr: sets[(pr, "test")] for pr in sp.PROTOS}
    preds, res = {}, {}
    for key, _ in MODELS:
        z = np.load(os.path.join(OUT, key, "predictions.npz"))
        preds[key] = {pr: {k: dict(vm=z[f"{pr}_{k}_vm"], spikes=z[f"{pr}_{k}_spikes"]) for k in ("one", "cont")}
                      for pr in sp.PROTOS}
        res[key] = json.load(open(os.path.join(OUT, key, "metrics.json")))
    summary = {k: {"trained": res[k]["trained"], "eval": res[k]["eval"]} for k, _ in MODELS}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- trace overlays
    sq = test["square"]
    amp = sq["I"][:, 1000]
    spk_sq = np.flatnonzero(sq["n_spikes"] > 0)
    mid = spk_sq[np.argmin(np.abs(sq["n_spikes"][spk_sq] - np.median(sq["n_spikes"][spk_sq])))]
    hi = spk_sq[np.argmax(amp[spk_sq])]
    nz = test["noise"]
    nz_i = np.flatnonzero(nz["n_spikes"] > 0)[np.random.default_rng(3).integers((nz["n_spikes"] > 0).sum())]
    rp = test["ramp"]
    rp_i = int(np.argsort(rp["n_spikes"])[len(rp["n_spikes"]) // 2])
    picks = [("square", mid, f"{amp[mid]:.0f} pA step"), ("square", hi, f"{amp[hi]:.0f} pA step (strongest)"),
             ("noise", nz_i, "noise"), ("ramp", rp_i, "ramp")]
    for kind, fname, title in (
            ("one", "comparison_one_step.png", "One-step prediction: each step starts from ground-truth Vm"),
            ("cont", "comparison_continuous.png",
             "Continuous prediction: ground-truth seed for 450 ms (grey), then each model runs on its own state")):
        fig, axes = plt.subplots(2, len(picks), figsize=(5.2 * len(picks), 6.6), sharex="col",
                                 gridspec_kw={"height_ratios": [3, 1.4]})
        for c, (pr, i, name) in enumerate(picks):
            s = test[pr]
            V = s["V"][i]
            t = np.arange(len(V)) / 1000
            top, bot = axes[0, c], axes[1, c]
            if kind == "cont":
                for a in (top, bot):
                    a.axvspan(0, base.SEED_MS / 1000, color="#eeeeee", zorder=0)
            top.plot(t, V, color="k", lw=1.6, label="ground truth")
            for key, lab in MODELS:
                vm = preds[key][pr][kind]["vm"][i]
                top.plot(t, vm, color=COLORS[key], lw=0.9, alpha=0.85, label=lab)
                bot.plot(t, np.minimum(vm, sp.V_CLIP) - np.minimum(V, sp.V_CLIP), color=COLORS[key], lw=0.8)
            bot.axhline(0, color="k", lw=0.6)
            top.set_title(f"{name} (test {pr} #{s['ids'][i]}, {s['n_spikes'][i]} spikes)", fontsize=10)
            bot.set_xlabel("time (s)")
            for a in (top, bot):
                a.spines[["top", "right"]].set_visible(False)
        axes[0, 0].set_ylabel("Vm (mV)")
        axes[1, 0].set_ylabel("error, clipped at -40 mV (mV)")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(title + ". Model spikes drawn at +30 mV", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    # ---- bars
    w = 0.2
    xg = np.arange(len(sp.PROTOS))

    def bars(ax, kind, metric, title, ylab):
        for j, (key, lab) in enumerate(MODELS):
            vals = [res[key]["eval"][pr][kind][metric] for pr in sp.PROTOS]
            b = ax.bar(xg + (j - 1.5) * w, vals, w, color=COLORS[key], label=lab)
            ax.bar_label(b, fmt="%.2f", fontsize=7)
        ax.set_xticks(xg, [PL[p] for p in sp.PROTOS])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bars(axes[0], "one", "vm_rmse_clipped_mV", "One-step prediction", "Vm RMSE, clipped at -40 mV (mV)")
    bars(axes[1], "cont", "vm_rmse_clipped_mV", "Continuous prediction (after 450 ms seed)", "Vm RMSE, clipped at -40 mV (mV)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Vm error on spiking test sets (spike peaks excluded by clipping both traces at -40 mV)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_bars.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.4))
    for r, kind in enumerate(("cont", "one")):
        kl = "continuous" if kind == "cont" else "one-step"
        bars(axes[r, 0], kind, "gamma", f"Coincidence factor Γ (±4 ms), {kl}", "Γ (1 = perfect, 0 = chance)")
        bars(axes[r, 1], kind, "ev", f"Explained variance, 10 ms Gaussian, {kl}", "EV (Allen GLIF measure)")
        bars(axes[r, 2], kind, "count_abs_err", f"Spike-count error per trace, {kl}", "mean |N_model - N_true|")
        bars(axes[r, 3], kind, "count_bias", f"Spike-count bias per trace, {kl}", "mean (N_model - N_true)")
        axes[r, 3].axhline(0, color="k", lw=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Spike fidelity on test traces. Continuous: after a 450 ms ground-truth seed. "
                 "One-step: each step from ground-truth Vm", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spike_metrics.png"), dpi=150)
    plt.close(fig)

    # ---- raster
    idx = np.flatnonzero(nz["n_spikes"] > 0)[:8]
    fig, ax = plt.subplots(figsize=(14, 6.5))
    row = 0
    gts = fs.gt_spikes(nz["V"])
    yt, yl = [], []
    for i in idx:
        ax.vlines(np.flatnonzero(gts[i]) / 1000, row, row + 0.8, color="k", lw=1.2)
        yt.append(row + 0.4); yl.append(f"#{nz['ids'][i]} truth")
        row += 1
        for key, lab in MODELS:
            ax.vlines(np.flatnonzero(preds[key]["noise"]["cont"]["spikes"][i]) / 1000, row, row + 0.8, color=COLORS[key], lw=1.2)
            row += 1
        row += 0.8
    ax.axvspan(0, base.SEED_MS / 1000, color="#eeeeee", zorder=0)
    ax.set_yticks(yt, yl, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color="k")] + [Line2D([], [], color=COLORS[k]) for k, _ in MODELS],
              ["ground truth"] + [l for _, l in MODELS], frameon=False, fontsize=8, loc="upper right", ncol=5)
    ax.set_title("Continuous prediction, 8 spiking noise test traces: ground truth (black) and each model below it", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "raster_continuous.png"), dpi=150)
    plt.close(fig)

    # ---- scatter (clipped)
    rng = np.random.default_rng(0)
    pts = []
    for pr in sp.PROTOS:
        n, T = test[pr]["V"].shape
        pts += [(pr, i, t) for i, t in zip(rng.integers(n, size=2000), rng.integers(base.SEED_MS, T, size=2000))]
    pick = [pts[k] for k in rng.choice(len(pts), 500, replace=False)]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    lo = min(test[pr]["V"].min() for pr in sp.PROTOS) - 3
    for c, (key, lab) in enumerate(MODELS):
        for r, kind in enumerate(("one", "cont")):
            ax = axes[r, c]
            gt = np.array([min(test[pr]["V"][i, t], sp.V_CLIP) for pr, i, t in pick])
            pm = np.array([min(preds[key][pr][kind]["vm"][i, t], sp.V_CLIP) for pr, i, t in pick])
            ax.scatter(gt, pm, s=5, color=COLORS[key], alpha=0.6)
            ax.plot([lo, -35], [lo, -35], "--", color="#7f8c8d", lw=1)
            ax.set_xlim(lo, -35); ax.set_ylim(lo, -35)
            kl = "one-step" if kind == "one" else "continuous"
            ax.set_title(f"{lab}, {kl}\nRMSE {np.sqrt(np.mean((pm - gt) ** 2)):.2f} mV", fontsize=9)
            ax.set_xlabel("ground truth Vm (mV, clipped at -40)", fontsize=8)
            ax.set_ylabel("predicted Vm (mV, clipped at -40)", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("500 random test points (square + noise + ramp, after the 450 ms seed), Vm clipped at -40 mV", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scatter_500_points.png"), dpi=150)
    plt.close(fig)
    print(json.dumps({k: {pr: {kk: {q: round(v, 3) for q, v in d.items() if q in ("vm_rmse_clipped_mV", "gamma", "ev", "count_bias")}
                              for kk, d in r["eval"][pr].items()} for pr in r["eval"]} for k, r in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
