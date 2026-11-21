"""Izhikevich layer alone: injected current -> somatic Vm, subthreshold Allen L2/3 traces.

Run from "Isolated L2_3 circuit/":
    python experiments/izhikevich_subthreshold/run.py [--iters 600]

Data: every simulation with 0 spikes in simulations/data (square and noise; no ramp trace is
spike-free). Split by trace (70/15/15, stratified by protocol, seed 0).

Model: I_izh = g * (I_pA + I0)  ->  IzhikevichLayer (published 2003 equations)  ->  Vm = Vs * v + V0.
The affine maps on current, voltage (Vs, V0) and time (time_scale, a factor on dt) make this the
general "simple model" (Izhikevich 2007: C, k, vr, vt free); with Vs = time_scale = g = 1 and
I0 = V0 = 0 it is the 2003 cell exactly. c and d do nothing without spikes.

Two variants (--variant):
  spiking: rheobase pinned to the NEURON cell's (83.8 pA) by freezing g, I0 and b after the
           steady-state initialization; V0, Vs, a and time_scale are trained.
  free:    everything trained. The fit is free to move the rheobase anywhere.

Initialization: least squares on the steady-state I-V of the square steps (closed form for the
2003 model), then a grid over time_scale and a. Training is free-running over the whole trace
(BPTT), loss = MSE in mV after the first 200 ms.

Evaluation, on the test traces:
  * one-step ("teacher-forced"): each step starts from the ground-truth Vm of the previous
    step (u is carried along driven by ground truth), like the ANN's single-step output;
  * continuous: Vm is clamped to ground truth for the seed (first SEED_MS), then the cell runs
    on its own predictions for the rest of the trace.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from izhikevich import IzhikevichLayer  # noqa: E402

DATA = os.path.join(ROOT, "simulations", "data")
RHEOBASE_PA = 83.8  # NEURON model's 1 s step rheobase (simulations/data/metadata.json)
SKIP_MS = 200      # loss/metrics ignore the start (initial-state transient)
SEED_MS = 450      # continuous prediction: ground-truth seed (stimulus starts at 500 ms)


def load(seed=0):
    rng = np.random.default_rng(seed)
    sets = {}
    for proto in ("square", "noise"):
        x = np.load(os.path.join(DATA, f"{proto}.npy"), mmap_mode="r")
        par = pd.read_csv(os.path.join(DATA, f"{proto}_params.csv"))
        idx = rng.permutation(np.flatnonzero(par.n_spikes.values == 0))
        n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
        for split, ids in (("train", idx[:n_tr]), ("val", idx[n_tr:n_tr + n_va]), ("test", idx[n_tr + n_va:])):
            ids = np.sort(ids)
            sets[(proto, split)] = dict(ids=ids, I=np.asarray(x[ids, :, 0], np.float32),
                                        V=np.asarray(x[ids, :, 1], np.float32))
    return sets


class Readout(keras.layers.Layer):
    """Affine current map in, affine voltage map out, around the Izhikevich layer."""

    def __init__(self, init, trainable):
        super().__init__()
        self.init, self.train_names = init, trainable
        self.izh = IzhikevichLayer(1, {"a": init["a"], "b": init["b"], "c": -65.0, "d": 8.0}, output_mode="v",
                                   input_gain_trainable="g" in trainable,
                                   trainable_params=tuple(n for n in ("a", "b", "time_scale") if n in trainable))

    def build(self, input_shape):
        for n in ("I0", "V0", "Vs"):
            setattr(self, n, self.add_weight(name=n, shape=(), trainable=n in self.train_names,
                                             initializer=keras.initializers.Constant(self.init[n])))
        self.izh.build((None, None, 1))
        self.izh.cell.build((None, 1))
        self.izh.input_gain.assign([self.init["g"]])
        self.izh.cell.time_scale.assign([self.init["time_scale"]])

    def call(self, I):
        return self.izh(I[..., None] + self.I0)[..., 0] * self.Vs + self.V0


def params_of(m):
    c = m.izh.cell
    return dict(a=float(c.a[0]), b=float(c.b[0]), c=float(c.c[0]), d=float(c.d[0]),
                time_scale=float(c.time_scale[0]), g=float(m.izh.input_gain[0]),
                I0=float(m.I0), V0=float(m.V0), Vs=float(m.Vs), dt=c.dt, n_substeps=c.n_substeps)


def simulate_np(I, p, V_gt=None, clamp_until=None, one_step=False, x=None):
    """NumPy copy of the fitted model (float64). I, V_gt: (n, T) in pA / mV.

    x: the cell's input current, if it comes from elsewhere (e.g. a dense front end).
    one_step: every step starts from ground truth Vm[t-1]. clamp_until: Vm is set to ground
    truth for t < clamp_until, then runs free. u always evolves with the model's update.
    """
    n, T = I.shape
    a, b, ts = p["a"], p["b"], p["time_scale"]
    h, dtu = p["dt"] / p["n_substeps"] * ts, p["dt"] * ts
    if x is None:
        x = p["g"] * (I.astype(np.float64) + p["I0"])   # same as layer: (I + I0) * g
    v = np.full(n, -65.0) if V_gt is None else (V_gt[:, 0].astype(np.float64) - p["V0"]) / p["Vs"]
    u = b * v
    out = np.empty((n, T))
    for t in range(T):
        if V_gt is not None and t > 0 and (one_step or (clamp_until is not None and t < clamp_until)):
            v = (V_gt[:, t - 1].astype(np.float64) - p["V0"]) / p["Vs"]
        fired = v >= 30.0
        v = np.where(fired, p["c"], v)
        u = np.where(fired, u + p["d"], u)
        for _ in range(p["n_substeps"]):
            v = v + h * (0.04 * (v * v) + 5.0 * v + 140.0 - u + x[:, t])
        u = u + dtu * (a * (b * v - u))
        out[:, t] = v
    return out * p["Vs"] + p["V0"]


def rheobase_pA(p):
    """Steady-state rheobase of the 2003 cell (saddle-node), mapped back to pA."""
    return ((5 - p["b"]) ** 2 / 0.16 - 140) / p["g"] - p["I0"]


def steady_state_init(sets, variant):
    """Least squares on the square-step I-V (V at 1.4 s) plus the resting potential."""
    from scipy.optimize import least_squares
    s = sets[("square", "train")]
    I = s["I"][:, 1000].astype(float)
    V = s["V"][:, 1400].astype(float)
    I, V = np.r_[I, 0.0], np.r_[V, s["V"][0, 400]]

    def vss(q):
        g, I0, V0, Vs, b = q
        disc = (5 - b) ** 2 - 0.16 * (140 + g * (I + I0))
        v = (-(5 - b) - np.sqrt(np.maximum(disc, 0))) / 0.08
        return np.where(disc > 0, Vs * v + V0, np.nan)

    def res(q):
        r = vss(q) - V
        if variant == "spiking":
            r = np.r_[r, 10 * (rheobase_pA(dict(zip(("g", "I0", "V0", "Vs", "b"), q))) - RHEOBASE_PA)]
        return np.nan_to_num(r, nan=100.0)

    starts = {"spiking": [0.0346, 0.08, 198.0, 4.17, 0.218], "free": [3.58, -9100.0, 6889.0, 8.96, 15.5]}
    q = least_squares(res, starts[variant], x_scale=[0.04, 10, 10, 1, 0.1]).x
    return dict(zip(("g", "I0", "V0", "Vs", "b"), map(float, q)))


def time_init(sets, p):
    """Grid over time_scale and a on 32 training traces (free run)."""
    s = sets[("square", "train")]
    k = np.random.default_rng(0).choice(len(s["ids"]), 32, replace=False)
    best = (np.inf, None)
    for ts in np.logspace(-3, 0.5, 15):
        for a in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            q = dict(p, time_scale=float(ts), a=a, c=-65.0, d=8.0, dt=1.0, n_substeps=2)
            with np.errstate(all="ignore"):
                e = np.nanmean((simulate_np(s["I"][k], q) - s["V"][k])[:, SKIP_MS:] ** 2)
            if np.isfinite(e) and e < best[0]:
                best = (e, dict(time_scale=float(ts), a=a))
    return best[1]


def metrics(pred, true):
    e = (pred - true)[:, SKIP_MS:]
    t = true[:, SKIP_MS:]
    return dict(rmse_mV=float(np.sqrt(np.mean(e ** 2))), mae_mV=float(np.mean(np.abs(e))),
                r2=float(1 - np.mean(e ** 2) / np.var(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--variant", choices=("spiking", "free"), default="spiking")
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam step as a fraction of each parameter's start value (default 0.01 spiking, 0.0005 free)")
    args = ap.parse_args()
    global OUT
    OUT = os.path.join(HERE, "results", args.variant)
    os.makedirs(OUT, exist_ok=True)
    tf.random.set_seed(0)
    rng = np.random.default_rng(0)
    sets = load()
    for k, s in sets.items():
        print(k, len(s["ids"]))

    init = steady_state_init(sets, args.variant)
    init.update(time_init(sets, init))
    print("init", init, "rheobase", rheobase_pA(init))
    trainable = ("V0", "Vs", "a", "time_scale") if args.variant == "spiking" else \
        ("g", "I0", "V0", "Vs", "a", "b", "time_scale")
    model = Readout(init, trainable)
    model(tf.zeros((1, 10)))
    # Adam step = a fraction of each parameter's starting magnitude. The free fit sits in a
    # large-offset regime (V0 and Vs*v nearly cancel), so it needs much smaller relative steps.
    rel = args.lr or (0.01 if args.variant == "spiking" else 0.0005)
    floor = {"I0": 1.0, "V0": 1.0}
    opts = {v.name: keras.optimizers.Adam(rel * max(abs(float(np.ravel(v.numpy())[0])), floor.get(v.name, 1e-3)))
            for v in model.trainable_variables}
    for v in model.trainable_variables:
        opts[v.name].build([v])

    @tf.function(reduce_retracing=True)
    def step(I, V):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(I)[:, SKIP_MS:] - V[:, SKIP_MS:]))
        grads = tape.gradient(loss, model.trainable_variables)
        for g, v in zip(grads, model.trainable_variables):
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))  # skip a blown-up step
            opts[v.name].apply([tf.clip_by_norm(g, 100.0)], [v])
        return loss

    def val_loss():
        tot, n = 0.0, 0
        for proto in ("square", "noise"):
            s = sets[(proto, "val")]
            p = model(tf.constant(s["I"])).numpy()
            tot += np.sum((p - s["V"])[:, SKIP_MS:] ** 2)
            n += p[:, SKIP_MS:].size
        return float(np.sqrt(tot / n))

    w_sq = len(sets[("square", "train")]["ids"]) * 2000
    w_no = len(sets[("noise", "train")]["ids"]) * 4500
    hist, best, best_w = [], np.inf, None
    t0 = time.time()
    for it in range(args.iters + 1):
        if it % 25 == 0:
            vl = val_loss()
            if vl < best:
                best, best_w = vl, [v.numpy() for v in model.weights]
            hist.append(dict(iter=it, val_rmse_mV=vl, **params_of(model)))
            print(f"it {it:4d}  val RMSE {vl:7.3f} mV  {params_of(model)}  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        proto = "square" if rng.random() < w_sq / (w_sq + w_no) else "noise"
        s = sets[(proto, "train")]
        b = rng.choice(len(s["ids"]), min(args.batch, len(s["ids"])), replace=False)
        step(tf.constant(s["I"][b]), tf.constant(s["V"][b]))

    for v, w in zip(model.weights, best_w):
        v.assign(w)
    p = params_of(model)
    p["rheobase_pA"] = rheobase_pA(p)
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "history.csv"), index=False)

    # ---- evaluation on test traces
    res = {"variant": args.variant, "init": init, "params": p, "best_val_rmse_mV": best, "n_traces": {f"{k[0]}_{k[1]}": len(v["ids"]) for k, v in sets.items()}}
    preds = {}
    for proto in ("square", "noise"):
        s = sets[(proto, "test")]
        free = simulate_np(s["I"], p)
        layer_free = model(tf.constant(s["I"])).numpy()
        diff = np.abs(free - layer_free)  # float64 NumPy copy vs float32 layer
        print(f"{proto}: NumPy copy vs layer, median {np.median(diff):.2e} mV, max {diff.max():.2e} mV")
        assert np.median(diff) < 1e-2
        one = simulate_np(s["I"], p, s["V"], one_step=True)
        cont = simulate_np(s["I"], p, s["V"], clamp_until=SEED_MS)
        preds[proto] = dict(one=one, cont=cont)
        res[proto] = {"one_step": metrics(one, s["V"]),
                      "continuous_after_seed": dict(zip(("rmse_mV", "mae_mV", "r2"),
                                                        metrics_after(cont, s["V"], SEED_MS)))}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    plots(sets, preds, p, res)


def metrics_after(pred, true, start):
    e, t = (pred - true)[:, start:], true[:, start:]
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), float(1 - np.mean(e ** 2) / np.var(t))


def plots(sets, preds, p, res, out=None, label=None):
    out = out or OUT
    label = label or f"Izhikevich layer ({res['variant']} variant, rheobase {p['rheobase_pA']:.0f} pA)"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    blue, red, grey = "#1f4e79", "#c0392b", "#7f8c8d"

    # pooled test points (t >= SKIP_MS), square + noise
    def pooled(kind):
        P, T = [], []
        for proto in ("square", "noise"):
            P.append(preds[proto][kind][:, SKIP_MS:].ravel())
            T.append(sets[(proto, "test")]["V"][:, SKIP_MS:].ravel())
        return np.concatenate(P), np.concatenate(T)

    # 1. scatter, 500 random points: one-step and continuous
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, kind, title in ((axes[0], "one", "One-step prediction"), (axes[1], "cont", "Continuous prediction")):
        P, T = pooled(kind)
        k = rng.choice(len(P), 500, replace=False)
        ax.scatter(T[k], P[k], s=8, alpha=0.6, color=blue, edgecolors="none")
        lo, hi = min(T[k].min(), P[k].min()) - 2, max(T[k].max(), P[k].max()) + 2
        ax.plot([lo, hi], [lo, hi], "--", color=grey, lw=1)
        e = P - T
        ax.set_title(f"{title}\nall test points: RMSE {np.sqrt(np.mean(e**2)):.2f} mV, "
                     f"R² {1-np.mean(e**2)/np.var(T):.3f}", fontsize=10)
        ax.set_xlabel("ground truth Vm (mV)")
        ax.set_ylabel("model Vm (mV)")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{label} on subthreshold Allen L2/3 test traces: 500 random points", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scatter_500_points.png"), dpi=150)
    plt.close(fig)

    # 2. one random test trace, one-step prediction
    proto = "square" if rng.random() < 0.8 else "noise"
    s = sets[(proto, "test")]
    i = rng.integers(len(s["ids"]))
    t = np.arange(s["V"].shape[1]) / 1000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(t, s["V"][i], color="k", lw=1.6, label="ground truth (NEURON)")
    a1.plot(t, preds[proto]["one"][i], color=red, lw=1, ls="--", label="model, one-step")
    a1.set_ylabel("Vm (mV)")
    a1.legend(frameon=False, fontsize=9)
    a1.set_title(f"Random test trace: {proto} #{s['ids'][i]}. {label}", fontsize=10)
    a2.plot(t, s["I"][i], color=grey)
    a2.set_ylabel("I (pA)")
    a2.set_xlabel("time (s)")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trace_one_step.png"), dpi=150)
    plt.close(fig)

    # 3. continuous prediction: 4 random square + 2 random noise test traces
    picks = [("square", j) for j in rng.choice(len(sets[("square", "test")]["ids"]), 4, replace=False)] + \
            [("noise", j) for j in rng.choice(len(sets[("noise", "test")]["ids"]), 2, replace=False)]
    picks.sort(key=lambda q: (q[0] != "square", sets[(q[0], "test")]["I"][q[1]].max()))
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for ax, (proto, i) in zip(axes.ravel(), picks):
        s = sets[(proto, "test")]
        t = np.arange(s["V"].shape[1]) / 1000
        ax.axvspan(0, SEED_MS / 1000, color="#f2d7d5", lw=0)
        ax.plot(t, s["V"][i], color="k", lw=1.6, label="ground truth")
        ax.plot(t, preds[proto]["cont"][i], color=red, lw=1, ls="--", label="continuous prediction")
        e = preds[proto]["cont"][i, SEED_MS:] - s["V"][i, SEED_MS:]
        amp = (f"{s['I'][i, 600:1400].mean():.0f} pA step" if proto == "square"
               else f"noise, mean {s['I'][i, 600:3400].mean():.0f} pA")
        ax.set_title(f"{proto} #{s['ids'][i]} ({amp}): RMSE after seed {np.sqrt(np.mean(e**2)):.2f} mV", fontsize=9.5)
        ax.set_ylabel("Vm (mV)")
        ax.set_xlabel("time (s)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"{label}\nContinuous prediction: Vm clamped to ground truth for the first {SEED_MS} ms (shaded), "
                 "then the cell runs on its own state", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "continuous_prediction.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
