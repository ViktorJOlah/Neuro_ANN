"""3 dense layers (32 units each) -> Izhikevich layer, on the subthreshold Allen L2/3 traces.

Run from "Isolated L2_3 circuit/" after experiments/izhikevich_subthreshold (it starts from that
fit):
    python experiments/dense3_izhikevich_subthreshold/run.py [--iters 400]

Model, at every 1 ms step:
    I (pA) / 100 -> Dense(32, tanh) -> Dense(32, tanh) -> Dense(32, tanh) -> Dense(1)
    -> x, the Izhikevich cell's input current -> IzhikevichLayer -> Vm = Vs * v + V0
The dense stack sees only the current at that step. It is a learned nonlinear current map
in front of the cell, replacing the linear g * (I + I0) of the Izhikevich-only model.
It can reshape the steady-state I-V. All memory is in the Izhikevich state (v, u).

Same data, split, training loop and evaluation as experiments/izhikevich_subthreshold.
Initialization: the cell (a, b, time_scale, Vs, V0) comes from the "spiking" Izhikevich-only fit.
The dense stack is pre-fitted to that fit's linear map. Trained: the dense stack, a, time_scale,
Vs and V0 (b is frozen; the dense stack can absorb it).
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "izhikevich_subthreshold"))
from izhikevich import IzhikevichLayer  # noqa: E402
import run as base  # noqa: E402  (load, simulate_np, metrics, plots)

OUT = os.path.join(HERE, "results")
BASE_FIT = os.path.join(ROOT, "experiments", "izhikevich_subthreshold", "results", "spiking", "metrics.json")
I_SCALE = 100.0


class DenseIzhikevich(keras.Model):
    def __init__(self, init, units=32, n_layers=3):
        super().__init__()
        self.init_p = init
        self.mlp = keras.Sequential([keras.layers.Dense(units, activation="tanh") for _ in range(n_layers)]
                                    + [keras.layers.Dense(1)], name="dense_front")
        self.izh = IzhikevichLayer(1, {"a": init["a"], "b": init["b"], "c": -65.0, "d": 8.0}, output_mode="v",
                                   trainable_params=("a", "time_scale"))

    def build(self, input_shape):
        self.mlp.build((None, None, 1))
        self.izh.build((None, None, 1))
        self.izh.cell.build((None, 1))
        self.izh.cell.time_scale.assign([self.init_p["time_scale"]])
        self.V0 = self.add_weight(name="V0", shape=(), initializer=keras.initializers.Constant(self.init_p["V0"]))
        self.Vs = self.add_weight(name="Vs", shape=(), initializer=keras.initializers.Constant(self.init_p["Vs"]))

    def current(self, I):
        return self.mlp(I[..., None] / I_SCALE)

    def call(self, I):
        return self.izh(self.current(I))[..., 0] * self.Vs + self.V0


def params_of(m):
    c = m.izh.cell
    return dict(a=float(c.a[0]), b=float(c.b[0]), c=float(c.c[0]), d=float(c.d[0]),
                time_scale=float(c.time_scale[0]), V0=float(m.V0), Vs=float(m.Vs),
                dt=c.dt, n_substeps=c.n_substeps)


def rheobase_pA(m, p):
    """Smallest step current whose mapped input reaches the cell's saddle-node (NaN if none <= 1 nA)."""
    I = np.arange(-150.0, 1000.0, 0.1, dtype=np.float32)
    x = m.mlp(I[:, None] / I_SCALE).numpy()[:, 0]
    x_rh = (5 - p["b"]) ** 2 / 0.16 - 140
    above = np.flatnonzero(x >= x_rh)
    return float(I[above[0]]) if len(above) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tf.random.set_seed(0)
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    sets = base.load()
    init = json.load(open(BASE_FIT))["params"]
    model = DenseIzhikevich(init)
    model(tf.zeros((1, 10)))

    # pre-fit the dense stack to the Izhikevich-only fit's linear current map
    Ig = np.linspace(-130, 110, 4001, dtype=np.float32)
    xg = init["g"] * (Ig + init["I0"])
    pre = keras.Sequential([model.mlp])
    pre.compile(keras.optimizers.Adam(3e-3), "mse")
    pre.fit((Ig / I_SCALE)[:, None], xg[:, None], epochs=300, batch_size=256, verbose=0)
    print("pre-fit max |error|:", float(np.abs(model.mlp((Ig / I_SCALE)[:, None]).numpy()[:, 0] - xg).max()))

    opts = {}
    for v in model.trainable_variables:
        lr = 1e-3 if "dense" in v.path else 0.005 * max(abs(float(np.ravel(v.numpy())[0])), 1.0 if v.name == "V0" else 1e-3)
        opts[v.path] = keras.optimizers.Adam(lr)
        opts[v.path].build([v])

    @tf.function(reduce_retracing=True)
    def step(I, V):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(I)[:, base.SKIP_MS:] - V[:, base.SKIP_MS:]))
        grads = tape.gradient(loss, model.trainable_variables)
        for g, v in zip(grads, model.trainable_variables):
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            opts[v.path].apply([tf.clip_by_norm(g, 100.0)], [v])
        return loss

    def val_rmse():
        tot, n = 0.0, 0
        for proto in ("square", "noise"):
            s = sets[(proto, "val")]
            e = (model(tf.constant(s["I"])).numpy() - s["V"])[:, base.SKIP_MS:]
            tot, n = tot + np.sum(e ** 2), n + e.size
        return float(np.sqrt(tot / n))

    w_sq = len(sets[("square", "train")]["ids"]) * 2000
    w_no = len(sets[("noise", "train")]["ids"]) * 4500
    hist, best, best_w, t0 = [], np.inf, None, time.time()
    for it in range(args.iters + 1):
        if it % 25 == 0:
            vl = val_rmse()
            if vl < best:
                best, best_w = vl, [v.numpy() for v in model.weights]
            hist.append(dict(iter=it, val_rmse_mV=vl, **params_of(model)))
            print(f"it {it:4d}  val RMSE {vl:7.3f} mV  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        proto = "square" if rng.random() < w_sq / (w_sq + w_no) else "noise"
        s = sets[(proto, "train")]
        b = rng.choice(len(s["ids"]), min(args.batch, len(s["ids"])), replace=False)
        step(tf.constant(s["I"][b]), tf.constant(s["V"][b]))

    for v, w in zip(model.weights, best_w):
        v.assign(w)
    model.save_weights(os.path.join(OUT, "model.weights.h5"))
    p = params_of(model)
    p["rheobase_pA"] = rheobase_pA(model, p)
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "history.csv"), index=False)

    res = {"variant": "dense3", "params": p, "best_val_rmse_mV": best}
    preds = {}
    for proto in ("square", "noise"):
        s = sets[(proto, "test")]
        x = model.current(tf.constant(s["I"])).numpy()[..., 0].astype(np.float64)
        free = base.simulate_np(s["I"], p, x=x)
        diff = np.abs(free - model(tf.constant(s["I"])).numpy())
        print(f"{proto}: NumPy copy vs model, median {np.median(diff):.2e} mV, max {diff.max():.2e} mV")
        assert np.median(diff) < 1e-2
        one = base.simulate_np(s["I"], p, s["V"], one_step=True, x=x)
        cont = base.simulate_np(s["I"], p, s["V"], clamp_until=base.SEED_MS, x=x)
        preds[proto] = dict(one=one, cont=cont)
        res[proto] = {"one_step": base.metrics(one, s["V"]),
                      "continuous_after_seed": dict(zip(("rmse_mV", "mae_mV", "r2"),
                                                        base.metrics_after(cont, s["V"], base.SEED_MS)))}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    rh = "none below 1 nA" if np.isnan(p["rheobase_pA"]) else f"{p['rheobase_pA']:.0f} pA"
    base.plots(sets, preds, p, res, out=OUT, label=f"3x32 dense -> Izhikevich (rheobase {rh})")
    input_map_plot(model, init)


def input_map_plot(model, init):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    I = np.linspace(-130, 150, 1000, dtype=np.float32)
    x = model.mlp(I[:, None] / I_SCALE).numpy()[:, 0]
    x_rh = (5 - model.izh.cell.b.numpy()[0]) ** 2 / 0.16 - 140
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(I, init["g"] * (I + init["I0"]), "--", color="#7f8c8d", label="Izhikevich-only fit (linear)")
    ax.plot(I, x, color="#1f4e79", label="learned dense map")
    ax.axhline(x_rh, color="#c0392b", lw=0.8, label="cell's rheobase current")
    ax.axvspan(84, 150, color="#f2f2f2", lw=0)
    ax.set_xlabel("injected current (pA)")
    ax.set_ylabel("Izhikevich input current (model units)")
    ax.set_title("Current map in front of the cell (grey: above training range)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "input_map.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
