"""Spiking fit, stage 2: refit every Izhikevich-layer parameter with the ANN frozen.

Run from "Isolated L2_3 circuit/" after spiking.py (its threshold, c and d are the start):
    python experiments/spiking/spiking_all.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Stage 1 (spiking.py) kept the subthreshold cell frozen and fitted only the threshold, c and d.
That cannot time spikes in continuous prediction: after a reset the frozen cell returns to
threshold on its fast membrane time scale, while the real cell creeps up over about 100 ms.
Here the ANN stays frozen and all ten Izhikevich-layer parameters are refitted:
    V_th, c (mV), d, a, b, time_scale, Vs, V0 (Vm map), g, I0 (current map)
The cell's input current is x = g*(I + I0) + ANN(64 ms current window), as before.

Same loss as stage 1 (clipped Vm MSE + van Rossum + count, each normalized at the start), on
segments from one pool of all training traces. A quarter of the segments come from spike-free
traces, so the subthreshold fit is kept in the loss. Minimized with Nelder-Mead (positive
parameters on a log scale). Same evaluation as stage 1.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking as sp  # noqa: E402

fs = sp.fs
OUT = os.path.join(HERE, "results_all")
NAMES = ("V_th", "c_mV", "d", "a", "b", "time_scale", "Vs", "V0", "g", "I0")
LOG = {"a", "time_scale", "Vs", "g"}
STEP = dict(V_th=2.0, c_mV=4.0, d=4.0, a=0.3, b=0.05, time_scale=0.3, Vs=0.15, V0=15.0, g=0.15, I0=10.0)
P_SPIKE_FREE = 0.25


class FullCell(sp.SpikingCell):
    def __init__(self, p, init):
        super().__init__(p, init)
        self.p = {k: tf.Variable(float(p[k]), dtype=tf.float32, name=k) for k in ("a", "b", "time_scale", "Vs", "V0")}
        self.g = tf.Variable(float(p["g"]), dtype=tf.float32, name="g")
        self.I0 = tf.Variable(float(p["I0"]), dtype=tf.float32, name="I0")

    def get(self):
        q = dict(V_th=self.V_th, c_mV=self.c_mV, d=self.d, g=self.g, I0=self.I0, **self.p)
        return {k: float(q[k].numpy()) for k in NAMES}

    def set(self, vals):
        q = dict(V_th=self.V_th, c_mV=self.c_mV, d=self.d, g=self.g, I0=self.I0, **self.p)
        for k, v in vals.items():
            q[k].assign(v)

    def run_full(self, I, net, V, S, M, clamp_until, one_step=False):
        return self.run(self.g * (I + self.I0) + net, V, S, M, clamp_until, one_step)


def to_vec(vals):
    return np.array([np.log(vals[k]) if k in LOG else vals[k] for k in NAMES])


def from_vec(q):
    return {k: float(np.exp(x)) if k in LOG else float(x) for k, x in zip(NAMES, q)}


def segments(sets, nets, split, rng, n):
    """n segments: 75% from spiking traces, 25% from spike-free traces, protocols mixed."""
    L = sp.SEG_CLAMP + sp.SEG_FREE
    keys = {True: [], False: []}
    for pr in sp.PROTOS:
        s = sets[(pr, split)]
        for i in range(len(s["ids"])):
            if s["V"].shape[1] >= L:
                keys[bool(s["n_spikes"][i] > 0)].append((pr, i))
    n_free = int(round(P_SPIKE_FREE * n)) if keys[False] else 0
    pick = [keys[True][j] for j in rng.choice(len(keys[True]), n - n_free, replace=False)]
    pick += [keys[False][j] for j in rng.choice(len(keys[False]), n_free, replace=False)]
    out = {k: [] for k in ("I", "net", "V", "S", "M")}
    masks = {}
    for pr, i in pick:
        s = sets[(pr, split)]
        T = s["V"].shape[1]
        t0 = rng.integers(0, T - L + 1)
        if (pr, split) not in masks:
            masks[(pr, split)] = sp.gt_masks(s["V"])
        S, M = masks[(pr, split)]
        for k, arr in (("I", s["I"]), ("net", nets[(pr, split)]), ("V", s["V"]), ("S", S), ("M", M)):
            out[k].append(arr[i, t0:t0 + L])
    return [tf.constant(np.stack(out[k]).astype(np.float32)) for k in ("I", "net", "V", "S", "M")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sp.MODEL_NAMES, required=True)
    ap.add_argument("--evals", type=int, default=1500)
    ap.add_argument("--n-train-seg", type=int, default=384)
    ap.add_argument("--n-val-seg", type=int, default=192)
    args = ap.parse_args()
    out = os.path.join(OUT, args.model)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sets = fs.load_split()
    st1 = json.load(open(os.path.join(sp.OUT, args.model, "metrics.json")))
    p0 = st1["frozen"]
    z = np.load(os.path.join(sp.CACHE, f"front_end_{args.model}.npz"))
    nets = {(pr, s): z[f"{pr}_{s}"] - np.float32(p0["g"]) * (sets[(pr, s)]["I"] + np.float32(p0["I0"]))
            for pr in sp.PROTOS for s in ("train", "val", "test")}
    cell = FullCell(p0, dict(V_th=st1["trained"]["V_th"], c_mV=st1["trained"]["c_mV"], d=st1["trained"]["d"]))
    K = sp.gauss_kernel()
    tseg = segments(sets, nets, "train", np.random.default_rng(0), args.n_train_seg)
    vseg = segments(sets, nets, "val", np.random.default_rng(1), args.n_val_seg)

    @tf.function
    def terms(I, net, V, S, M):
        vm, s = cell.run_full(I, net, V, S, M, sp.SEG_CLAMP)
        vm, s, V, S = vm[:, sp.SEG_CLAMP:], s[:, sp.SEG_CLAMP:], V[:, sp.SEG_CLAMP:], S[:, sp.SEG_CLAMP:]
        t_vm = tf.reduce_mean(tf.square(tf.minimum(vm, sp.V_CLIP) - tf.minimum(V, sp.V_CLIP)))
        t_vr = tf.reduce_mean(tf.square(sp.smooth(s, K) - sp.smooth(S, K)))
        t_n = tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))
        return tf.stack([t_vm, t_vr, t_n])

    def batched(seg, chunk=128):
        n = int(seg[0].shape[0])
        return np.sum([terms(*(a[k:k + chunk] for a in seg)).numpy() * min(chunk, n - k) for k in range(0, n, chunk)], 0) / n

    start = cell.get()
    norm = np.maximum(batched(tseg), 1e-6)
    w = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm
    print("start", {k: round(v, 4) for k, v in start.items()})
    print("initial terms (train segments): Vm %.2f mV^2, vR %.4f, count %.2f" % tuple(norm), flush=True)
    hist, best = [], [np.inf, None]

    def objective(q):
        vals = from_vec(q)
        cell.set(vals)
        tv = batched(tseg)
        loss = float(np.sum(w * tv)) if np.all(np.isfinite(tv)) else 1e6
        hist.append(dict(eval=len(hist), loss=loss, vm_term=tv[0], vr_term=tv[1], count_term=tv[2], **vals))
        if loss < best[0]:
            best[0], best[1] = loss, vals
        if len(hist) % 50 == 1:
            print(f"eval {len(hist):5d}  train loss {loss:.4f}  best {best[0]:.4f} "
                  f"(Vm {tv[0]:.2f} mV^2, vR {tv[1]:.4f}, count {tv[2]:.2f})  {time.time()-t0:.0f}s", flush=True)
        return loss

    from scipy.optimize import minimize
    q0 = to_vec(start)
    simplex = np.vstack([q0] + [q0 + np.eye(len(NAMES))[i] * STEP[k] for i, k in enumerate(NAMES)])
    opt = minimize(objective, q0, method="Nelder-Mead",
                   options=dict(maxfev=args.evals, xatol=1e-3, fatol=1e-4, initial_simplex=simplex, adaptive=True))
    fitted = best[1]
    cell.set(start)
    val0 = batched(vseg)
    cell.set(fitted)
    val1 = batched(vseg)
    print(f"val loss: start {np.sum(w * val0):.4f} -> fitted {np.sum(w * val1):.4f}  ({opt.nfev} evaluations, {opt.message})")
    print("fitted", {k: round(v, 4) for k, v in fitted.items()}, flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)

    test = sp.pools(sets, {(pr, "test"): fitted["g"] * (sets[(pr, "test")]["I"] + fitted["I0"]) + nets[(pr, "test")]
                           for pr in sp.PROTOS}, "test")
    preds, res = sp.evaluate(cell, test)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{pr}_{k}_{q}": (v.astype(np.float32) if q == "vm" else v) for pr, d in preds.items()
                           for k, dd in d.items() for q, v in dd.items()})
    res = {"model": args.model, "trained": fitted, "start": start, "val_terms_start": val0.tolist(),
           "val_terms_fitted": val1.tolist(), "loss_norm": norm.tolist(), "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q: round(v, 3) for q, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
