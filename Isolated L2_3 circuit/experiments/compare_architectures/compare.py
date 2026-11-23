"""Compare the four Izhikevich top-layer architectures on the subthreshold test traces.

Run from "Isolated L2_3 circuit/" after the four experiments have been trained:
    python experiments/compare_architectures/compare.py

Models (same data, split, training budget and evaluation):
  1. Izhikevich alone                      experiments/izhikevich_subthreshold (spiking variant)
  2. 3x32 dense -> Izhikevich              experiments/dense3_izhikevich_subthreshold
  3. CNN-LSTM -> Izhikevich                experiments/cnnlstm_izhikevich_subthreshold --arch cnnlstm
  4. 3x32 dense + CNN-LSTM -> Izhikevich    ... --arch dense_cnnlstm

One-step ("native") prediction: each 1 ms step starts from the ground-truth Vm of the previous
step. Continuous: Vm clamped to ground truth for the first 450 ms, then free-running.
Errors: RMSE over test traces, from 200 ms (one-step) or 450 ms (continuous) to the end.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "results")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("izh_base", os.path.join(EXP, "izhikevich_subthreshold", "run.py"))
sys.modules["run"] = base  # the other experiments import it as "run"
dense3 = load_module("dense3", os.path.join(EXP, "dense3_izhikevich_subthreshold", "run.py"))

MODELS = [
    ("izh", "Izhikevich alone"),
    ("dense3", "3×32 dense → Izh"),
    ("cnnlstm", "CNN-LSTM → Izh"),
    ("dense_cnnlstm", "dense + CNN-LSTM → Izh"),
]
COLORS = {"izh": "#e67e22", "dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}


def predictions(sets):
    preds = {}
    # 1. Izhikevich alone
    p = json.load(open(os.path.join(EXP, "izhikevich_subthreshold", "results", "spiking", "metrics.json")))["params"]
    preds["izh"] = {proto: dict(one=base.simulate_np(sets[(proto, "test")]["I"], p, sets[(proto, "test")]["V"], one_step=True),
                                cont=base.simulate_np(sets[(proto, "test")]["I"], p, sets[(proto, "test")]["V"],
                                                      clamp_until=base.SEED_MS))
                    for proto in ("square", "noise")}
    # 2. dense3
    d3 = os.path.join(EXP, "dense3_izhikevich_subthreshold", "results")
    init = json.load(open(dense3.BASE_FIT))["params"]
    m = dense3.DenseIzhikevich(init)
    m(tf.zeros((1, 10)))
    m.load_weights(os.path.join(d3, "model.weights.h5"))
    p = dense3.params_of(m)
    preds["dense3"] = {}
    for proto in ("square", "noise"):
        s = sets[(proto, "test")]
        x = m.current(tf.constant(s["I"])).numpy()[..., 0].astype(np.float64)
        preds["dense3"][proto] = dict(one=base.simulate_np(s["I"], p, s["V"], one_step=True, x=x),
                                      cont=base.simulate_np(s["I"], p, s["V"], clamp_until=base.SEED_MS, x=x))
    # 3, 4. CNN-LSTM variants (saved by their run)
    for arch in ("cnnlstm", "dense_cnnlstm"):
        z = np.load(os.path.join(EXP, "cnnlstm_izhikevich_subthreshold", "results", arch, "test_predictions.npz"))
        preds[arch] = {proto: dict(one=z[f"{proto}_one"], cont=z[f"{proto}_cont"]) for proto in ("square", "noise")}
    return preds


def rmse(preds, sets, key, kind, protos=("square", "noise")):
    start = base.SKIP_MS if kind == "one" else base.SEED_MS
    e = np.concatenate([(preds[key][pr][kind] - sets[(pr, "test")]["V"])[:, start:].ravel() for pr in protos])
    return float(np.sqrt(np.mean(e ** 2)))


def trace_panels(sets, preds, kind, picks, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(picks), figsize=(5.2 * len(picks), 6.4), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.4]})
    for c, (proto, i, name) in enumerate(picks):
        s = sets[(proto, "test")]
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
        top.set_title(f"{name} (test {proto} #{s['ids'][i]})", fontsize=10)
        bot.set_xlabel("time (s)")
        if c == 0:
            top.set_ylabel("Vm (mV)")
            bot.set_ylabel("error (mV)")
        for a in (top, bot):
            a.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def bars(sets, preds, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    groups = [("square", ("square",)), ("noise", ("noise",)), ("all", ("square", "noise"))]
    w = 0.2
    for ax, kind, title in ((axes[0], "one", "One-step (native) prediction"),
                            (axes[1], "cont", "Continuous prediction (after 450 ms seed)")):
        for j, (key, lab) in enumerate(MODELS):
            vals = [res[key][kind][g] for g, _ in groups]
            xs = np.arange(len(groups)) + (j - 1.5) * w
            ax.bar(xs, vals, w, color=COLORS[key], label=lab)
            for x_, v in zip(xs, vals):
                ax.text(x_, v, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(np.arange(len(groups)), [g for g, _ in groups])
        ax.set_ylabel("RMSE (mV)")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Test-set error, subthreshold Allen L2/3 traces", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_bars.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    sets = base.load()
    preds = predictions(sets)
    res = {key: {kind: {"square": rmse(preds, sets, key, kind, ("square",)),
                        "noise": rmse(preds, sets, key, kind, ("noise",)),
                        "all": rmse(preds, sets, key, kind)} for kind in ("one", "cont")} for key, _ in MODELS}
    with open(os.path.join(OUT, "rmse_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

    sq = sets[("square", "test")]
    amp = sq["I"][:, 1000]
    rng = np.random.default_rng(3)
    picks = [("square", int(np.argmin(np.abs(amp + 60))), f"{amp[np.argmin(np.abs(amp + 60))]:.0f} pA step"),
             ("square", int(np.argmax(amp)), f"{amp.max():.0f} pA step (near threshold)"),
             ("noise", int(rng.integers(len(sets[("noise", "test")]["ids"]))), "noise")]
    trace_panels(sets, preds, "one", picks, "comparison_one_step.png",
                 "One-step (native) prediction: each step starts from ground-truth Vm")
    trace_panels(sets, preds, "cont", picks, "comparison_continuous.png",
                 "Continuous prediction: ground-truth seed for 450 ms (grey), then each model runs on its own state")
    bars(sets, preds, res)


if __name__ == "__main__":
    main()
