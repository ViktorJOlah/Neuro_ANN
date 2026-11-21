"""Train one windowed model on the pooled subthreshold windows (square + noise + ramp start).

Run from "Isolated L2_3 circuit/":
    python experiments/windowed/train_subthreshold.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Data: the same spike-free traces and split as before. That is 548 square and 86 noise traces
split 70/15/15 by trace, plus the one unique spike-free ramp segment (0 to 3985 ms) in training.
Every 64-step window of every training trace goes into one pool, which is shuffled. Each epoch
draws EPOCH_SAMPLES windows from that pool. The loss is the one-step MSE of Vm(t) in mV.
Model selection uses the MSE on 50,000 fixed validation windows.

Evaluation on held-out square/noise test traces, and on the ramp segment (in-sample):
  one-step:   every window from ground truth;
  continuous: ground truth for the first 450 ms, then the model's own Vm fills the voltage column.
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "expanded_subthreshold"))
import windowed as Wd  # noqa: E402
import continue_training as ct  # noqa: E402  (load_expanded: split + ramp segment)

base = ct.base
EPOCHS, EPOCH_SAMPLES, BATCH = 30, 400_000, 512
SPLIT = {"square": "test", "noise": "test", "ramp": "eval"}


def pool(sets, split_of):
    traces = []
    for proto, split in split_of:
        s = sets[(proto, split)]
        traces += list(zip(s["I"], s["V"]))
    return Wd.Windows(traces)


def evaluate(m, sets):
    preds, res = {}, {}
    for proto, split in SPLIT.items():
        s = sets[(proto, split)]
        one = Wd.predict_one_step(m, s["I"], s["V"])
        cont = Wd.predict_continuous(m, s["I"], s["V"], base.SEED_MS)
        preds[proto] = dict(one=one, cont=cont)
        res[proto] = {"one_step": base.metrics(one, s["V"]),
                      "continuous_after_seed": dict(zip(("rmse_mV", "mae_mV", "r2"),
                                                        base.metrics_after(cont, s["V"], base.SEED_MS)))}
    return preds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=Wd.MODEL_NAMES, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    out = os.path.join(HERE, "results", "subthreshold", args.model)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    sets, cut = ct.load_expanded()
    train = pool(sets, [("square", "train"), ("noise", "train"), ("ramp", "train")])
    val = pool(sets, [("square", "val"), ("noise", "val")])
    Xv, yv = val.gather(np.random.default_rng(1).choice(len(val), 50_000, replace=False))
    print(f"train windows {len(train):,}  val windows {len(val):,}")

    m = Wd.make_model(args.model)
    step_vars = [v for v in m.trainable_variables if v.path.split("/")[-2] == "izhikevich_step" or v.name in ("g", "I0")]
    net_vars = [v for v in m.trainable_variables if all(v is not s for s in step_vars)]
    total_steps = args.epochs * (EPOCH_SAMPLES // BATCH)
    sched = keras.optimizers.schedules.CosineDecay(2e-3, total_steps, alpha=0.02)
    opt_net = keras.optimizers.Adam(sched) if net_vars else None
    opt_par = {v.path: keras.optimizers.Adam(2e-3 * max(abs(float(v.numpy())), 1e-3)) for v in step_vars}
    # (only the Izhikevich-only model trains cell and mapping parameters; see windowed.WindowModel)
    if opt_net:
        opt_net.build(net_vars)
    for v in step_vars:
        opt_par[v.path].build([v])
    print("Izhikevich/map params trained:", [v.path for v in step_vars])

    @tf.function
    def step(X, y):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(m(X) - y))
        grads = tape.gradient(loss, net_vars + step_vars)
        grads = [tf.clip_by_norm(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)), 100.0) for g in grads]
        if net_vars:
            opt_net.apply(grads[:len(net_vars)], net_vars)
        for g, v in zip(grads[len(net_vars):], step_vars):
            opt_par[v.path].apply([g], [v])
        return loss

    def val_rmse():
        p = np.concatenate([m(tf.constant(Xv[k:k + 8192])).numpy() for k in range(0, len(Xv), 8192)])
        return float(np.sqrt(np.mean((p - yv) ** 2)))

    hist, best, best_w, t0 = [], np.inf, None, time.time()
    vr = val_rmse()
    hist.append(dict(epoch=0, train_rmse=np.nan, val_rmse=vr, **m.izh_params()))
    print(f"epoch  0  val one-step RMSE {vr:.4f} mV")
    for ep in range(1, args.epochs + 1):
        idx = rng.choice(len(train), EPOCH_SAMPLES, replace=False)
        losses = []
        for k in range(0, EPOCH_SAMPLES - BATCH + 1, BATCH):
            X, y = train.gather(idx[k:k + BATCH])
            losses.append(float(step(tf.constant(X), tf.constant(y))))
        vr = val_rmse()
        if vr < best:
            best, best_w = vr, [v.numpy() for v in m.weights]
        hist.append(dict(epoch=ep, train_rmse=float(np.sqrt(np.mean(losses))), val_rmse=vr, **m.izh_params()))
        print(f"epoch {ep:2d}  train RMSE {np.sqrt(np.mean(losses)):.4f}  val one-step RMSE {vr:.4f} mV  "
              f"{time.time()-t0:.0f}s", flush=True)

    for v, w in zip(m.weights, best_w):
        v.assign(w)
    m.save_weights(os.path.join(out, "model.weights.h5"))
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    t1 = time.time()
    preds, res = evaluate(m, sets)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{p}_{k}": v.astype(np.float32) for p, d in preds.items() for k, v in d.items()})
    res = {"model": args.model, "best_val_one_step_rmse_mV": best, "izh_params": m.izh_params(),
           "n_params": int(sum(np.prod(v.shape) for v in m.trainable_variables)), "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"eval {time.time()-t1:.0f}s")
    print(json.dumps({p: {k: round(d[k]["rmse_mV"], 3) for k in d} for p, d in res["eval"].items()}, indent=1))


if __name__ == "__main__":
    main()
