"""Spiking fit, stage 3: GLIF-style after-spike currents and an adapting threshold.

Run from "Isolated L2_3 circuit/" after spiking.py (it caches the front-end currents):
    python experiments/spiking/spiking_asc.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Stages 1 and 2 showed that the Izhikevich cell's single slow variable u cannot give both the fast
after-hyperpolarization (to about -50 mV within 5 ms) and the slow creep back to threshold
(interspike intervals of about 65 to 300 ms). Following the Allen GLIF models (Teeter et al.
2018, Nat. Commun. 9:709), each spike now also triggers two after-spike currents and raises the
threshold:
    I_fast += A_fast,  dI_fast/dt = -I_fast / tau_fast     (pA)
    I_slow += A_slow,  dI_slow/dt = -I_slow / tau_slow     (pA)
    theta  += d_theta, dtheta/dt  = -theta / tau_theta     (mV, added to V_th)
The cell's input current becomes x + g * (I_fast + I_slow), with g the frozen current gain.
Everything fitted on subthreshold traces stays frozen: the ANN, g, I0, a, b, time scale, Vs, V0.
Fitted (9): V_th, c (mV), d, A_fast, tau_fast, A_slow, tau_slow, d_theta, tau_theta.

Same spike rule, loss, segments and evaluation as stage 1 (spiking.py). Nelder-Mead is started
from three initial guesses (different after-current sizes and time constants), and the best fit
on the training segments is kept.
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

fs, base = sp.fs, sp.base
OUT = os.path.join(HERE, "results_asc")
NAMES = ("V_th", "c_mV", "d", "A_fast", "tau_fast", "A_slow", "tau_slow", "d_theta", "tau_theta")
LOG = {"tau_fast", "tau_slow", "tau_theta"}
STEP = dict(V_th=1.5, c_mV=3.0, d=3.0, A_fast=20.0, tau_fast=0.4, A_slow=15.0, tau_slow=0.4, d_theta=1.0, tau_theta=0.4)
STARTS = [
    dict(V_th=-40.0, c_mV=-48.0, d=0.0, A_fast=-50.0, tau_fast=10.0, A_slow=-30.0, tau_slow=150.0, d_theta=2.0, tau_theta=50.0),
    dict(V_th=-40.0, c_mV=-48.0, d=0.0, A_fast=-100.0, tau_fast=5.0, A_slow=-60.0, tau_slow=300.0, d_theta=1.0, tau_theta=100.0),
    dict(V_th=-41.0, c_mV=-46.0, d=4.0, A_fast=-30.0, tau_fast=20.0, A_slow=-15.0, tau_slow=80.0, d_theta=3.0, tau_theta=20.0),
]


class AscCell(sp.SpikingCell):
    def __init__(self, p, init):
        super().__init__(p, dict(V_th=init["V_th"], c_mV=init["c_mV"], d=init["d"]))
        self.g = tf.constant(float(p["g"]), tf.float32)
        self.extra = {k: tf.Variable(float(init[k]), dtype=tf.float32, name=k) for k in NAMES[3:]}

    def get(self):
        q = dict(V_th=self.V_th, c_mV=self.c_mV, d=self.d, **self.extra)
        return {k: float(q[k].numpy()) for k in NAMES}

    def set(self, vals):
        q = dict(V_th=self.V_th, c_mV=self.c_mV, d=self.d, **self.extra)
        for k, v in vals.items():
            q[k].assign(v)

    def run(self, x, V_gt, gt_spk, gt_in, clamp_until, one_step=False, forced=False):
        """forced=True: spikes are forced at the ground-truth spike times (reset inside each
        ground-truth spike, after-currents triggered at its onset) and Vm runs free in between.
        No threshold is applied, so this measures the interspike trajectory alone."""
        P, E = self.p, self.extra
        a, b, ts, Vs, V0 = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"]
        c = (self.c_mV - V0) / Vs
        k_f, k_s, k_th = tf.exp(-1.0 / E["tau_fast"]), tf.exp(-1.0 / E["tau_slow"]), tf.exp(-1.0 / E["tau_theta"])
        n, T = tf.shape(x)[0], tf.shape(x)[1]
        v = (V_gt[:, 0] - V0) / Vs
        u = b * v
        z = tf.zeros([n], tf.float32)
        s, i_f, i_s, th = z, z, z, z
        vm_ta = tf.TensorArray(tf.float32, size=T)
        s_ta = tf.TensorArray(tf.float32, size=T)
        h = 0.5 * ts
        for t in tf.range(T):
            clamp = tf.logical_or(t < clamp_until, one_step)
            if clamp:
                prev = tf.maximum(t - 1, 0)
                first = tf.cast(t > 0, tf.float32)
                v = (V_gt[:, prev] - V0) / Vs
                v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
                onset = first * gt_spk[:, prev]
            elif forced:
                prev = tf.maximum(t - 1, 0)
                v = gt_in[:, prev] * c + (1.0 - gt_in[:, prev]) * v
                onset = gt_spk[:, prev]
            else:
                onset = tf.stop_gradient(s)
                v = onset * c + (1.0 - onset) * v
            u = u + onset * self.d
            i_f = i_f * k_f + onset * E["A_fast"]
            i_s = i_s * k_s + onset * E["A_slow"]
            th = th * k_th + onset * E["d_theta"]
            xt = x[:, t] + self.g * (i_f + i_s)
            for _ in range(2):
                v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
            u = u + ts * a * (b * v - u)
            if forced:
                v = tf.minimum(v, (0.0 - V0) / Vs)     # no threshold: keep the free run bounded
            vm = Vs * v + V0
            s = self.spike_fn((vm - self.V_th - th) / sp.SURR_MV)
            vm_ta = vm_ta.write(t, s * sp.SPIKE_DISPLAY_MV + (1.0 - s) * vm)
            s_ta = s_ta.write(t, s)
        return tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack())


def to_vec(vals):
    return np.array([np.log(vals[k]) if k in LOG else vals[k] for k in NAMES])


def from_vec(q):
    return {k: float(np.exp(x)) if k in LOG else float(x) for k, x in zip(NAMES, q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sp.MODEL_NAMES, required=True)
    ap.add_argument("--evals", type=int, default=900)
    ap.add_argument("--n-train-seg", type=int, default=384)
    ap.add_argument("--n-val-seg", type=int, default=192)
    args = ap.parse_args()
    out = os.path.join(OUT, args.model)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sets = fs.load_split()
    p = json.load(open(os.path.join(sp.SUB, args.model, "metrics.json")))["params"]
    z = np.load(os.path.join(sp.CACHE, f"front_end_{args.model}.npz"))
    xs = {(pr, s): z[f"{pr}_{s}"] for pr in sp.PROTOS for s in ("train", "val", "test")}
    train, val, test = (sp.pools(sets, xs, s) for s in ("train", "val", "test"))
    cell = AscCell(p, STARTS[0])
    K = sp.gauss_kernel()
    tseg = sp.draw_segments(train, np.random.default_rng(0), args.n_train_seg)
    vseg = sp.draw_segments(val, np.random.default_rng(1), args.n_val_seg)

    @tf.function
    def terms(X, V, S, M):
        vm, s = cell.run(X, V, S, M, sp.SEG_CLAMP)
        vm, s, V, S = vm[:, sp.SEG_CLAMP:], s[:, sp.SEG_CLAMP:], V[:, sp.SEG_CLAMP:], S[:, sp.SEG_CLAMP:]
        t_vm = tf.reduce_mean(tf.square(tf.minimum(vm, sp.V_CLIP) - tf.minimum(V, sp.V_CLIP)))
        t_vr = tf.reduce_mean(tf.square(sp.smooth(s, K) - sp.smooth(S, K)))
        t_n = tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))
        return tf.stack([t_vm, t_vr, t_n])

    def batched(seg, chunk=128):
        n = int(seg[0].shape[0])
        return np.sum([terms(*(a[k:k + chunk] for a in seg)).numpy() * min(chunk, n - k) for k in range(0, n, chunk)], 0) / n

    # normalization: the stage-1 model's terms (no after-currents), so stages are comparable
    st1 = json.load(open(os.path.join(sp.OUT, args.model, "metrics.json")))["trained"]
    cell.set(dict(st1, A_fast=0.0, A_slow=0.0, d_theta=0.0))
    norm = np.maximum(batched(tseg), 1e-6)
    w = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm
    print("stage-1 terms (train segments): Vm %.2f mV^2, vR %.4f, count %.2f" % tuple(norm), flush=True)
    hist, best = [], [np.inf, None]

    def objective(q):
        vals = from_vec(q)
        cell.set(vals)
        tv = batched(tseg)
        loss = float(np.sum(w * tv)) if np.all(np.isfinite(tv)) else 1e6
        hist.append(dict(eval=len(hist), loss=loss, vm_term=tv[0], vr_term=tv[1], count_term=tv[2], **vals))
        if loss < best[0]:
            best[0], best[1] = loss, vals
        if len(hist) % 100 == 1:
            print(f"eval {len(hist):5d}  loss {loss:.4f}  best {best[0]:.4f} (Vm {tv[0]:.2f} mV^2, vR {tv[1]:.4f}, "
                  f"count {tv[2]:.2f})  {time.time()-t0:.0f}s", flush=True)
        return loss

    from scipy.optimize import minimize
    for j, st in enumerate(STARTS):
        q0 = to_vec(st)
        simplex = np.vstack([q0] + [q0 + np.eye(len(NAMES))[i] * STEP[k] for i, k in enumerate(NAMES)])
        opt = minimize(objective, q0, method="Nelder-Mead",
                       options=dict(maxfev=args.evals, xatol=1e-3, fatol=1e-4, initial_simplex=simplex, adaptive=True))
        print(f"start {j}: loss {opt.fun:.4f} after {opt.nfev} evaluations; best so far {best[0]:.4f}", flush=True)
    fitted = best[1]
    cell.set(fitted)
    val1 = batched(vseg)
    print(f"stage-1 val loss 1.0 x3 terms -> fitted {np.sum(w * val1):.4f}", flush=True)
    print("fitted", {k: round(v, 3) for k, v in fitted.items()}, flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    preds, res = sp.evaluate(cell, test)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{pr}_{k}_{q}": (v.astype(np.float32) if q == "vm" else v) for pr, d in preds.items()
                           for k, dd in d.items() for q, v in dd.items()})
    res = {"model": args.model, "trained": fitted, "frozen": p, "val_terms_fitted": val1.tolist(),
           "loss_norm_stage1": norm.tolist(), "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q: round(v, 3) for q, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
