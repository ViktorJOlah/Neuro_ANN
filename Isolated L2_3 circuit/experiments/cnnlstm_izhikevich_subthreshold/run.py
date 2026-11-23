"""CNN-LSTM -> Izhikevich and Dense + CNN-LSTM -> Izhikevich, subthreshold Allen L2/3 traces.

Run from "Isolated L2_3 circuit/" after experiments/izhikevich_subthreshold:
    python experiments/cnnlstm_izhikevich_subthreshold/run.py --arch cnnlstm
    python experiments/cnnlstm_izhikevich_subthreshold/run.py --arch dense_cnnlstm

Model (causal, 1 ms steps):
    I/100 -> [dense_cnnlstm only: Dense(32, tanh) x3]
          -> Conv1D(32, 5, causal, relu) -> Conv1D(32, 5, causal, relu) -> LSTM(32) -> Dense(1)
          -> + g*(I + I0)  (linear skip, from the Izhikevich-only fit)
          -> x, the Izhikevich cell's input current -> IzhikevichLayer -> Vm = Vs*v + V0
The network sees the injected-current history (causal convolutions plus LSTM memory) and
supplies the cell's input current. The Izhikevich state (v, u) carries Vm, so one-step and
continuous prediction differ only in whether v is reset to ground truth at each step,
exactly as for the other two models. The output Dense(1) starts near zero (std 0.01), so
training starts close to the Izhikevich-only fit.

Same data, split, training loop and evaluation as experiments/izhikevich_subthreshold. Test
predictions are saved to results/<arch>/test_predictions.npz for the comparison.
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
import run as base  # noqa: E402

BASE_FIT = os.path.join(ROOT, "experiments", "izhikevich_subthreshold", "results", "spiking", "metrics.json")
I_SCALE = 100.0


class SeqIzhikevich(keras.Model):
    def __init__(self, init, arch, units=32):
        super().__init__()
        self.init_p, self.arch = init, arch
        front = [keras.layers.Dense(units, activation="tanh") for _ in range(3)] if arch == "dense_cnnlstm" else []
        self.net = keras.Sequential(front + [
            keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
            keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
            keras.layers.LSTM(units, return_sequences=True),
            keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01)),
        ], name="net")
        self.izh = IzhikevichLayer(1, {"a": init["a"], "b": init["b"], "c": -65.0, "d": 8.0}, output_mode="v",
                                   trainable_params=("a", "time_scale"))

    def build(self, input_shape):
        self.net.build((None, None, 1))
        self.izh.build((None, None, 1))
        self.izh.cell.build((None, 1))
        self.izh.cell.time_scale.assign([self.init_p["time_scale"]])
        for n in ("g", "I0", "V0", "Vs"):
            setattr(self, n, self.add_weight(name=n, shape=(), initializer=keras.initializers.Constant(self.init_p[n])))

    def current(self, I):
        return self.net(I[..., None] / I_SCALE) + (self.g * (I + self.I0))[..., None]

    def call(self, I):
        return self.izh(self.current(I))[..., 0] * self.Vs + self.V0


def params_of(m):
    c = m.izh.cell
    return dict(a=float(c.a[0]), b=float(c.b[0]), c=float(c.c[0]), d=float(c.d[0]),
                time_scale=float(c.time_scale[0]), g=float(m.g), I0=float(m.I0), V0=float(m.V0), Vs=float(m.Vs),
                dt=c.dt, n_substeps=c.n_substeps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    out = os.path.join(HERE, "results", args.arch)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    sets = base.load()
    init = json.load(open(BASE_FIT))["params"]
    model = SeqIzhikevich(init, args.arch)
    model(tf.zeros((1, 10)))

    opts = {}
    for v in model.trainable_variables:
        if v.path.startswith("net/") or "/net/" in v.path:
            lr = 1e-3
        else:
            lr = 0.005 * max(abs(float(np.ravel(v.numpy())[0])), {"V0": 1.0, "I0": 1.0}.get(v.name, 1e-3))
        opts[v.path] = keras.optimizers.Adam(lr)
        opts[v.path].build([v])
    print({k: o.learning_rate.numpy() for k, o in opts.items() if "net" not in k})

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
            hist.append(dict(iter=it, val_rmse_mV=vl))
            print(f"it {it:4d}  val RMSE {vl:7.3f} mV  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        proto = "square" if rng.random() < w_sq / (w_sq + w_no) else "noise"
        s = sets[(proto, "train")]
        b = rng.choice(len(s["ids"]), min(args.batch, len(s["ids"])), replace=False)
        step(tf.constant(s["I"][b]), tf.constant(s["V"][b]))

    for v, w in zip(model.weights, best_w):
        v.assign(w)
    model.save_weights(os.path.join(out, "model.weights.h5"))
    p = params_of(model)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)

    res = {"arch": args.arch, "params": p, "best_val_rmse_mV": best,
           "n_params": int(sum(np.prod(v.shape) for v in model.trainable_variables))}
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
    np.savez_compressed(os.path.join(out, "test_predictions.npz"),
                        **{f"{proto}_{k}": v.astype(np.float32) for proto, d in preds.items() for k, v in d.items()})
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    label = {"cnnlstm": "CNN-LSTM -> Izhikevich", "dense_cnnlstm": "3x32 dense + CNN-LSTM -> Izhikevich"}[args.arch]
    base.plots(sets, preds, p, res, out=out, label=label)


if __name__ == "__main__":
    main()
