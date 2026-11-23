"""Combined design: a fixed 64 ms current window feeds the ANN, Vm lives only in the Izhikevich state.

Run from "Isolated L2_3 circuit/":
    python experiments/current_window/run.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Model, per 1 ms step t:
    window(t) = I(t-63 .. t)                         (batch, 64, 1), the same for every ANN
    x(t)      = g*(I(t) + I0) + ANN(window(t)/100)   the Izhikevich cell's input current
    Vm(t)     = Vs * v(t) + V0,  v and u integrated by the Izhikevich layer over the whole trace
The ANN never sees voltage. Before 64 ms the window is padded with the trace's first current
sample (the cell is at rest with constant current at the start of every trace).
ANNs, all ending in Dense(1) initialised near zero so training starts at the Izhikevich-alone fit:
    dense3         Flatten -> Dense(32, tanh) x3
    cnnlstm        Conv1D(32, 5, causal, relu) x2 -> LSTM(32), reset for every window
    dense_cnnlstm  Dense(32, tanh) x3 per time step -> the cnnlstm stack
    izh            no ANN: x(t) = g*(I(t) + I0)

Training: closed loop, as in the first fits. The cell runs freely from rest over each whole
training trace and the loss is the Vm MSE after 200 ms. Square, noise and ramp traces form one
pool, and every batch is drawn from it at random, so protocols are mixed within batches. Data and split as in
experiments/expanded_subthreshold (spike-free square + noise, 70/15/15 by trace, plus the
spike-free ramp start in training, reported in-sample). Every model starts from the
Izhikevich-alone "spiking" fit. Evaluation: one-step (v reset to measured Vm(t-1) each step)
and continuous (450 ms ground-truth seed, then free running).
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
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "expanded_subthreshold"))
import continue_training as ct  # noqa: E402

base, seq = ct.base, ct.seq
W, I_SCALE = 64, 100.0
MODEL_NAMES = ("izh", "dense3", "cnnlstm", "dense_cnnlstm")
OUT = os.path.join(HERE, "results")


def current_windows(I):
    """(B, T) -> (B, T, 64): row t holds I(t-63..t), left-padded with I[:, 0]."""
    pad = tf.repeat(I[:, :1], W - 1, axis=1)
    return tf.signal.frame(tf.concat([pad, I], 1), W, 1, axis=1)


class CurrentWindowIzhikevich(seq.SeqIzhikevich):
    def __init__(self, init, arch, units=32):
        super().__init__(init, "cnnlstm", units)
        self.arch = arch
        head = [keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))]
        if arch == "izh":
            self.net = None
        elif arch == "dense3":
            self.net = keras.Sequential([keras.layers.Flatten()] +
                                        [keras.layers.Dense(units, activation="tanh") for _ in range(3)] + head, name="net")
        else:
            front = [keras.layers.Dense(units, activation="tanh") for _ in range(3)] if arch == "dense_cnnlstm" else []
            self.net = keras.Sequential(front + [
                keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
                keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
                keras.layers.LSTM(units)] + head, name="net")

    def build(self, input_shape):
        if self.net is not None:
            self.net.build((None, W, 1))
        self.izh.build((None, None, 1))
        self.izh.cell.build((None, 1))
        self.izh.cell.time_scale.assign([self.init_p["time_scale"]])
        for n in ("g", "I0", "V0", "Vs"):
            setattr(self, n, self.add_weight(name=n, shape=(), initializer=keras.initializers.Constant(self.init_p[n])))

    def current(self, I):
        lin = (self.g * (I + self.I0))[..., None]
        if self.net is None:
            return lin
        win = current_windows(I)
        B, T = tf.shape(I)[0], tf.shape(I)[1]
        y = self.net(tf.reshape(win, (-1, W, 1)) / I_SCALE)
        return lin + tf.reshape(y, (B, T, 1))

    def current_np(self, I, chunk=4):
        return np.concatenate([self.current(tf.constant(I[k:k + chunk])).numpy()
                               for k in range(0, len(I), chunk)])[..., 0].astype(np.float64)


def evaluate(m, sets):
    p, preds, res = seq.params_of(m), {}, {}
    for proto, split in (("square", "test"), ("noise", "test"), ("ramp", "eval")):
        s = sets[(proto, split)]
        x = m.current_np(s["I"])
        one = base.simulate_np(s["I"], p, s["V"], one_step=True, x=x)
        cont = base.simulate_np(s["I"], p, s["V"], clamp_until=base.SEED_MS, x=x)
        preds[proto] = dict(one=one, cont=cont)
        res[proto] = {"one_step": base.metrics(one, s["V"]),
                      "continuous_after_seed": dict(zip(("rmse_mV", "mae_mV", "r2"),
                                                        base.metrics_after(cont, s["V"], base.SEED_MS)))}
    return p, preds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--time-only", action="store_true")
    args = ap.parse_args()
    out = os.path.join(OUT, args.model)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    sets, cut = ct.load_expanded()
    init = json.load(open(seq.BASE_FIT))["params"]
    m = CurrentWindowIzhikevich(init, args.model)
    m(tf.zeros((1, 10)))

    opts = {}
    for v in m.trainable_variables:
        if "net" in v.path.split("/")[:-1]:
            lr = 1e-3
        else:
            lr = 0.005 * max(abs(float(np.ravel(v.numpy())[0])), {"V0": 1.0, "I0": 1.0}.get(v.name, 1e-3))
        opts[v.path] = keras.optimizers.Adam(lr)
        opts[v.path].build([v])

    @tf.function(reduce_retracing=True)
    def step(I, V, M):
        with tf.GradientTape() as tape:
            e = tf.square(m(I)[:, base.SKIP_MS:] - V[:, base.SKIP_MS:]) * M[:, base.SKIP_MS:]
            loss = tf.reduce_sum(e) / tf.reduce_sum(M[:, base.SKIP_MS:])
        grads = tape.gradient(loss, m.trainable_variables)
        for g, v in zip(grads, m.trainable_variables):
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            opts[v.path].apply([tf.clip_by_norm(g, 100.0)], [v])
        return loss

    def val_rmse():
        tot, n = 0.0, 0
        for proto, split in (("square", "val"), ("noise", "val"), ("ramp", "train")):
            s = sets[(proto, split)]
            for k in range(0, len(s["I"]), 8):
                e = (m(tf.constant(s["I"][k:k + 8])).numpy() - s["V"][k:k + 8])[:, base.SKIP_MS:]
                tot, n = tot + np.sum(e ** 2), n + e.size
        return float(np.sqrt(tot / n))

    # one pool of training traces from all three protocols; a batch mixes them. Shorter traces
    # are padded with their last current and voltage sample and masked out of the loss.
    tr = [(s["I"][k], s["V"][k]) for key in (("square", "train"), ("noise", "train"), ("ramp", "train"))
          for s in [sets[key]] for k in range(len(s["I"]))]
    T_max = max(len(i) for i, _ in tr)
    P_I = np.stack([np.pad(i, (0, T_max - len(i)), mode="edge") for i, _ in tr]).astype(np.float32)
    P_V = np.stack([np.pad(v, (0, T_max - len(v)), mode="edge") for _, v in tr]).astype(np.float32)
    P_M = np.stack([np.arange(T_max) < len(i) for i, _ in tr]).astype(np.float32)
    print(f"training pool: {len(tr)} traces, padded to {T_max} ms", flush=True)
    hist, best, best_w, t0 = [], np.inf, None, time.time()
    for it in range(args.iters + 1):
        if it % 50 == 0 and not args.time_only:
            vl = val_rmse()
            if vl < best:
                best, best_w = vl, [v.numpy() for v in m.weights]
            hist.append(dict(iter=it, val_rmse_mV=vl))
            print(f"it {it:4d}  val (+ramp) RMSE {vl:7.3f} mV  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        b = np.sort(rng.choice(len(tr), args.batch, replace=False))
        t1 = time.time()
        step(tf.constant(P_I[b]), tf.constant(P_V[b]), tf.constant(P_M[b]))
        if args.time_only:
            print(f"it {it}: {time.time()-t1:.1f}s", flush=True)
            if it == 6:
                return

    for v, w in zip(m.weights, best_w):
        v.assign(w)
    m.save_weights(os.path.join(out, "model.weights.h5"))
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    p, preds, res = evaluate(m, sets)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{proto}_{k}": v.astype(np.float32) for proto, d in preds.items() for k, v in d.items()})
    res = {"model": args.model, "best_val_rmse_mV": best, "params": p, "ramp_segment_ms": cut,
           "n_params": int(sum(np.prod(v.shape) for v in m.trainable_variables)), "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: round(d[k]["rmse_mV"], 3) for k in d} for pr, d in res["eval"].items()}, indent=1))


if __name__ == "__main__":
    main()
