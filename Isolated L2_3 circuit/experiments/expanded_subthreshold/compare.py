"""Comparison plots after continued training on the expanded subthreshold set (square + noise + ramp start).

Run from "Isolated L2_3 circuit/" after continue_training.py has run for all four models:
    python experiments/expanded_subthreshold/compare.py

Writes results/comparison_one_step.png, comparison_continuous.png, error_bars.png and
rmse_summary.json. Square and noise errors are on held-out test traces. The ramp error is
in-sample, because only one unique spike-free ramp segment exists. In the bar plots, black
ticks mark each model's error before the continued training.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import continue_training as ct  # noqa: E402

base = ct.base
OUT = os.path.join(HERE, "results")
MODELS = [
    ("izh", "Izhikevich alone"),
    ("dense3", "3×32 dense → Izh"),
    ("cnnlstm", "CNN-LSTM → Izh"),
    ("dense_cnnlstm", "dense + CNN-LSTM → Izh"),
]
COLORS = {"izh": "#e67e22", "dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}
SPLIT = {"square": "test", "noise": "test", "ramp": "eval"}


def rmse(preds, sets, key, kind, protos):
    start = base.SKIP_MS if kind == "one" else base.SEED_MS
    e = np.concatenate([(preds[key][pr][kind] - sets[(pr, SPLIT[pr])]["V"])[:, start:].ravel() for pr in protos])
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets, cut = ct.load_expanded()
    preds, before = {}, {}
    for key, _ in MODELS:
        z = np.load(os.path.join(OUT, key, "predictions.npz"))
        preds[key] = {pr: dict(one=z[f"{pr}_one"], cont=z[f"{pr}_cont"]) for pr in SPLIT}
        before[key] = json.load(open(os.path.join(OUT, key, "metrics.json")))["before"]

    groups = [("square", ("square",)), ("noise", ("noise",)), ("ramp*", ("ramp",)), ("all test", ("square", "noise"))]
    res = {key: {kind: {g: rmse(preds, sets, key, kind, prs) for g, prs in groups} for kind in ("one", "cont")}
           for key, _ in MODELS}
    with open(os.path.join(OUT, "rmse_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    # ---- trace overlays
    sq = sets[("square", "test")]
    amp = sq["I"][:, 1000]
    rng = np.random.default_rng(3)
    picks = [("square", int(np.argmin(np.abs(amp + 60))), f"{amp[np.argmin(np.abs(amp + 60))]:.0f} pA step"),
             ("square", int(np.argmax(amp)), f"{amp.max():.0f} pA step (near threshold)"),
             ("noise", int(rng.integers(len(sets[("noise", "test")]["ids"]))), "noise"),
             ("ramp", 0, f"ramp, 0 to {cut} ms (in training set)")]
    for kind, fname, title in (
            ("one", "comparison_one_step.png", "One-step (native) prediction: each step starts from ground-truth Vm"),
            ("cont", "comparison_continuous.png",
             "Continuous prediction: ground-truth seed for 450 ms (grey), then each model runs on its own state")):
        fig, axes = plt.subplots(2, len(picks), figsize=(5.0 * len(picks), 6.4), sharex="col",
                                 gridspec_kw={"height_ratios": [3, 1.4]})
        for c, (proto, i, name) in enumerate(picks):
            s = sets[(proto, SPLIT[proto])]
            t = np.arange(s["V"].shape[1]) / 1000
            top, bot = axes[0, c], axes[1, c]
            if kind == "cont":
                for a in (top, bot):
                    a.axvspan(0, base.SEED_MS / 1000, color="#eeeeee", lw=0)
            top.plot(t, s["V"][i], color="k", lw=2.2, label="ground truth")
            for key, lab in MODELS:
                pr = preds[key][proto][kind][i]
                top.plot(t, pr, color=COLORS[key], lw=1.0, label=lab)
                bot.plot(t, pr - s["V"][i], color=COLORS[key], lw=0.9)
            bot.axhline(0, color="k", lw=0.6)
            top.set_title(name if proto == "ramp" else f"{name} (test {proto} #{s['ids'][i]})", fontsize=10)
            bot.set_xlabel("time (s)")
            if c == 0:
                top.set_ylabel("Vm (mV)")
                bot.set_ylabel("error (mV)")
            for a in (top, bot):
                a.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(title + " (after training on square + noise + ramp start)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    # ---- error bars (black tick = before continued training)
    bkey = {"square": "square", "noise": "noise", "ramp*": "ramp"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.2
    for ax, kind, mkey, title in ((axes[0], "one", "one_step", "One-step (native) prediction"),
                                  (axes[1], "cont", "continuous_after_seed", "Continuous prediction (after 450 ms seed)")):
        for j, (key, lab) in enumerate(MODELS):
            vals = [res[key][kind][g] for g, _ in groups]
            xs = np.arange(len(groups)) + (j - 1.5) * w
            ax.bar(xs, vals, w, color=COLORS[key], label=lab)
            for x_, v, (g, _) in zip(xs, vals, groups):
                ax.text(x_, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
                if g in bkey:
                    ax.plot([x_ - w * 0.4, x_ + w * 0.4], [before[key][bkey[g]][mkey]["rmse_mV"]] * 2, color="k", lw=1.4)
        ax.set_xticks(np.arange(len(groups)), [g for g, _ in groups])
        ax.set_ylabel("RMSE (mV)")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Error after continued training on square + noise + ramp start. Black tick: before. "
                 "*ramp is in-sample (one unique trace).", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_bars.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
