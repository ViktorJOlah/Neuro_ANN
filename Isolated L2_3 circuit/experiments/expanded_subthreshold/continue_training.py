"""Continue training the four Izhikevich top-layer models with the ramp's spike-free start added.

Run from "Isolated L2_3 circuit/" after the four subthreshold experiments:
    python experiments/expanded_subthreshold/continue_training.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Data: the same spike-free square and noise traces and split, plus the start of the ramp
protocol up to 10 ms before its first spike (0 to 3985 ms, current 0 to 87 pA). All 1000 ramp
simulations are bit-identical up to that point, because they share the onset and slope and
all terminate at or after the first spike. So the ramp adds one unique trace. It goes into
training, and its fit is reported as in-sample. Test errors stay on held-out square and noise.

Each model starts from its trained weights and continues for --iters iterations. 10% of
iterations use the ramp trace and the rest use square/noise batches as before. Parameter
learning rates are half those of the first round.
"""
from __future__ import annotations

import argparse
import importlib.util
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
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
OUT_ROOT = os.path.join(HERE, "results")
RAMP_CUT_BEFORE_SPIKE_MS = 10
P_RAMP = 0.1


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("izh_base", os.path.join(EXP, "izhikevich_subthreshold", "run.py"))
sys.modules["run"] = base
dense3 = load_module("dense3", os.path.join(EXP, "dense3_izhikevich_subthreshold", "run.py"))
seq = load_module("seq", os.path.join(EXP, "cnnlstm_izhikevich_subthreshold", "run.py"))


def load_expanded():
    sets = base.load()
    x = np.load(os.path.join(base.DATA, "ramp.npy"), mmap_mode="r")
    V = np.asarray(x[:, :, 1])
    first = np.array([np.argmax(v > -20.0) for v in V])
    cut = int(first.min()) - RAMP_CUT_BEFORE_SPIKE_MS
    seg = np.asarray(x[:, :cut, :])
    assert np.all(seg == seg[:1]), "ramp starts are expected to be identical"
    r = dict(ids=np.array([0]), I=seg[:1, :, 0].astype(np.float32), V=seg[:1, :, 1].astype(np.float32))
    sets[("ramp", "train")] = r
    sets[("ramp", "eval")] = r  # in-sample: there is only one unique ramp segment
    return sets, cut


def build(model_name):
    """Model rebuilt from its first-round fit, plus a function giving the cell's input current."""
    if model_name == "izh":
        p = json.load(open(os.path.join(EXP, "izhikevich_subthreshold", "results", "spiking", "metrics.json")))["params"]
        m = base.Readout(p, ("V0", "Vs", "a", "time_scale"))
        m(tf.zeros((1, 10)))
        current = lambda I: None  # noqa: E731  simulate_np uses g*(I + I0)

        def params():
            q = base.params_of(m)
            return q
        return m, current, params
    if model_name == "dense3":
        init = json.load(open(dense3.BASE_FIT))["params"]
        m = dense3.DenseIzhikevich(init)
        m(tf.zeros((1, 10)))
        m.load_weights(os.path.join(EXP, "dense3_izhikevich_subthreshold", "results", "model.weights.h5"))
        return m, lambda I: m.current(tf.constant(I)).numpy()[..., 0].astype(np.float64), lambda: dense3.params_of(m)
    init = json.load(open(seq.BASE_FIT))["params"]
    m = seq.SeqIzhikevich(init, model_name)
    m(tf.zeros((1, 10)))
    m.load_weights(os.path.join(EXP, "cnnlstm_izhikevich_subthreshold", "results", model_name, "model.weights.h5"))
    return m, lambda I: m.current(tf.constant(I)).numpy()[..., 0].astype(np.float64), lambda: seq.params_of(m)


def evaluate(sets, current, p):
    preds, res = {}, {}
    for proto, split in (("square", "test"), ("noise", "test"), ("ramp", "eval")):
        s = sets[(proto, split)]
        x = current(s["I"])
        one = base.simulate_np(s["I"], p, s["V"], one_step=True, x=x)
        cont = base.simulate_np(s["I"], p, s["V"], clamp_until=base.SEED_MS, x=x)
        preds[proto] = dict(one=one, cont=cont)
        res[proto] = {"one_step": base.metrics(one, s["V"]),
                      "continuous_after_seed": dict(zip(("rmse_mV", "mae_mV", "r2"),
                                                        base.metrics_after(cont, s["V"], base.SEED_MS)))}
    return preds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("izh", "dense3", "cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    out = os.path.join(OUT_ROOT, args.model)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(1)
    rng = np.random.default_rng(1)
    sets, cut = load_expanded()
    model, current, params = build(args.model)

    _, res_before = evaluate(sets, current, params())

    opts = {}
    for v in model.trainable_variables:
        if v.name in ("kernel", "bias", "recurrent_kernel"):
            lr = 5e-4
        else:
            rel = 0.005 if args.model == "izh" else 0.0025
            lr = rel * max(abs(float(np.ravel(v.numpy())[0])), {"V0": 1.0, "I0": 1.0}.get(v.name, 1e-3))
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
        for proto, split in (("square", "val"), ("noise", "val"), ("ramp", "train")):
            s = sets[(proto, split)]
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
            print(f"it {it:4d}  val (+ramp) RMSE {vl:7.3f} mV  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        u = rng.random()
        if u < P_RAMP:
            s, b = sets[("ramp", "train")], np.array([0])
        else:
            proto = "square" if rng.random() < w_sq / (w_sq + w_no) else "noise"
            s = sets[(proto, "train")]
            b = rng.choice(len(s["ids"]), min(args.batch, len(s["ids"])), replace=False)
        step(tf.constant(s["I"][b]), tf.constant(s["V"][b]))

    for v, w in zip(model.weights, best_w):
        v.assign(w)
    if hasattr(model, "save_weights"):  # the Izhikevich-only model is fully described by its params
        model.save_weights(os.path.join(out, "model.weights.h5"))
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    p = params()
    preds, res_after = evaluate(sets, current, p)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{proto}_{k}": v.astype(np.float32) for proto, d in preds.items() for k, v in d.items()})
    res = {"model": args.model, "ramp_segment_ms": cut, "params": p, "before": res_before, "after": res_after}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: {pr: res[k][pr]["continuous_after_seed"]["rmse_mV"] for pr in res[k]}
                      for k in ("before", "after")}, indent=2))


if __name__ == "__main__":
    main()
