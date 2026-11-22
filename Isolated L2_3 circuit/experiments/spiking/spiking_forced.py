"""Spiking fit, stage 4: Izhikevich layer + after-spike currents trained on spike-forced trajectories.

Run from "Isolated L2_3 circuit/" after spiking.py (it caches the front-end currents):
    python experiments/spiking/spiking_forced.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Why: with spikes forced at the true times, the frozen subthreshold cell still reaches threshold
20 to 33 ms after each spike instead of about 108 ms, and its between-spike Vm is about 15 mV off
whatever after-currents are added. The spike-free training traces never went above about -46 mV,
so the subthreshold fit never saw the -46 to -40 mV range the real cell creeps through between
spikes.

Model: the stage-3 cell (spiking_asc.py): Izhikevich layer with a threshold spike rule, reset c,
u jump d, fast and slow after-spike currents and an adapting threshold. The ANN stays frozen.
Step 1 (gradient descent, Adam): train the Izhikevich layer (a, b, time_scale, Vs, V0, g, I0),
    c, d and the after-currents (A_fast, tau_fast, A_slow, tau_slow) on spike-forced
    trajectories. Spikes are forced at the ground-truth times (reset to c, u += d, after-currents
    triggered) and Vm runs free in between, without a threshold. Loss: Vm MSE outside spikes (2 ms
    before to 3 ms after each onset is left out). A quarter of the segments are spike-free, which
    keeps the subthreshold fit in the loss. With no threshold inside the loop the loss is smooth,
    so plain gradients work. Model selection on validation segments.
Step 2 (Nelder-Mead): V_th, d_theta and tau_theta on the free-running spike loss of stage 1
    (clipped Vm + van Rossum + count), everything else fixed.
Same evaluation and outputs as the other stages.
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
import spiking as sp  # noqa: E402
import spiking_all as sall  # noqa: E402  (segments with I and the ANN part kept apart)

fs = sp.fs
OUT = os.path.join(HERE, "results_forced")
LOGP = ("a", "time_scale", "Vs", "g", "tau_fast", "tau_slow", "tau_theta")
LINP = ("b", "V0", "I0", "c_mV", "d", "A_fast", "A_slow", "V_th", "d_theta")
LR = dict(a=0.01, time_scale=0.01, Vs=0.005, g=0.005, tau_fast=0.02, tau_slow=0.02, tau_theta=0.02,
          b=2e-4, V0=0.2, I0=0.2, c_mV=0.1, d=0.05, A_fast=0.5, A_slow=0.5, V_th=0.05, d_theta=0.05)
STEP1 = ("a", "b", "time_scale", "Vs", "V0", "g", "I0", "c_mV", "d", "A_fast", "tau_fast", "A_slow", "tau_slow")
ASC0 = dict(c_mV=-48.0, d=0.0, A_fast=-50.0, tau_fast=10.0, A_slow=-30.0, tau_slow=150.0,
            V_th=-40.0, d_theta=1.0, tau_theta=50.0)


class ForcedCell(tf.Module):
    """All parameters as raw variables (log for positive ones). run() matches SpikingCell.run."""

    def __init__(self, p, init):
        super().__init__()
        allp = {**{k: p[k] for k in ("a", "b", "time_scale", "Vs", "V0", "g", "I0")}, **init}
        self.raw = {k: tf.Variable(float(np.log(v) if k in LOGP else v), dtype=tf.float32, name=k)
                    for k, v in allp.items()}

    def val(self, k):
        return tf.exp(self.raw[k]) if k in LOGP else self.raw[k]

    def values(self):
        return {k: float(self.val(k).numpy()) for k in self.raw}

    def set(self, vals):
        for k, v in vals.items():
            self.raw[k].assign(float(np.log(v) if k in LOGP else v))

    def current(self, I, net):
        return self.val("g") * (I + self.val("I0")) + net

    def run(self, x, V_gt, gt_spk, gt_in, clamp_until, one_step=False, forced=False):
        a, b, ts, Vs, V0 = (self.val(k) for k in ("a", "b", "time_scale", "Vs", "V0"))
        g, d, V_th = self.val("g"), self.val("d"), self.val("V_th")
        c = (self.val("c_mV") - V0) / Vs
        k_f, k_s, k_th = (tf.exp(-1.0 / self.val(k)) for k in ("tau_fast", "tau_slow", "tau_theta"))
        A_f, A_s, d_th = self.val("A_fast"), self.val("A_slow"), self.val("d_theta")
        n, T = tf.shape(x)[0], tf.shape(x)[1]
        v = (V_gt[:, 0] - V0) / Vs
        u = b * v
        z = tf.zeros([n], tf.float32)
        s, i_f, i_s, th = z, z, z, z
        vm_ta = tf.TensorArray(tf.float32, size=T)
        s_ta = tf.TensorArray(tf.float32, size=T)
        h = 0.5 * ts
        v_cap = (0.0 - V0) / Vs
        for t in tf.range(T):
            prev = tf.maximum(t - 1, 0)
            first = tf.cast(t > 0, tf.float32)
            if tf.logical_or(t < clamp_until, one_step):
                v = (V_gt[:, prev] - V0) / Vs
                v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
                onset = first * gt_spk[:, prev]
            elif forced:
                v = gt_in[:, prev] * c + (1.0 - gt_in[:, prev]) * v
                onset = gt_spk[:, prev]
            else:
                onset = tf.stop_gradient(s)
                v = onset * c + (1.0 - onset) * v
            u = u + onset * d
            i_f = i_f * k_f + onset * A_f
            i_s = i_s * k_s + onset * A_s
            th = th * k_th + onset * d_th
            xt = x[:, t] + g * (i_f + i_s)
            for _ in range(2):
                v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
            u = u + ts * a * (b * v - u)
            if forced:
                v = tf.minimum(v, v_cap)
            vm = Vs * v + V0
            s = sp.SpikingCell.spike_fn((vm - V_th - th) / sp.SURR_MV)
            vm_ta = vm_ta.write(t, s * sp.SPIKE_DISPLAY_MV + (1.0 - s) * vm)
            s_ta = s_ta.write(t, s)
        return tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack())


def keep_mask(S):
    k = tf.ones([6, 1, 1])
    near = tf.nn.conv1d(tf.pad(S, [[0, 0], [3, 2]])[..., None], k, 1, "VALID")[..., 0]
    return tf.cast(near < 0.5, tf.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sp.MODEL_NAMES, required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--evals2", type=int, default=250)
    args = ap.parse_args()
    out = os.path.join(OUT, args.model)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    t0 = time.time()
    sets = fs.load_split()
    p0 = json.load(open(os.path.join(sp.SUB, args.model, "metrics.json")))["params"]
    z = np.load(os.path.join(sp.CACHE, f"front_end_{args.model}.npz"))
    nets = {(pr, s): z[f"{pr}_{s}"] - np.float32(p0["g"]) * (sets[(pr, s)]["I"] + np.float32(p0["I0"]))
            for pr in sp.PROTOS for s in ("train", "val", "test")}
    cell = ForcedCell(p0, ASC0)
    K = sp.gauss_kernel()
    rng = np.random.default_rng(0)
    vseg = sall.segments(sets, nets, "val", np.random.default_rng(1), 192)
    tseg2 = sall.segments(sets, nets, "train", np.random.default_rng(2), 384)

    def forced_mse(I, net, V, S, M):
        vm, _ = cell.run(cell.current(I, net), V, S, M, sp.SEG_CLAMP, forced=True)
        keep = keep_mask(S)[:, sp.SEG_CLAMP:]
        e = (vm[:, sp.SEG_CLAMP:] - V[:, sp.SEG_CLAMP:]) * keep
        return tf.reduce_sum(e * e) / tf.reduce_sum(keep)

    f_mse = tf.function(forced_mse)
    train_vars = [cell.raw[k] for k in STEP1]
    opts = {k: keras.optimizers.Adam(LR[k]) for k in STEP1}
    for k in STEP1:
        opts[k].build([cell.raw[k]])

    @tf.function
    def step(I, net, V, S, M):
        with tf.GradientTape() as tape:
            loss = forced_mse(I, net, V, S, M)
        grads = tape.gradient(loss, train_vars)
        for k, gr, v in zip(STEP1, grads, train_vars):
            gr = tf.where(tf.math.is_finite(gr), gr, tf.zeros_like(gr))
            opts[k].apply([tf.clip_by_norm(gr, 10.0)], [v])
        return loss

    def val_mse():
        return float(np.mean([f_mse(*(a[k:k + 96] for a in vseg)).numpy() for k in range(0, 192, 96)]))

    hist, best, best_v = [], np.inf, None
    for it in range(args.iters + 1):
        if it % 25 == 0:
            vm = val_mse()
            if np.isfinite(vm) and vm < best:
                best, best_v = vm, cell.values()
            hist.append(dict(step=1, iter=it, val_forced_mse=vm, **cell.values()))
            q = cell.values()
            print(f"it {it:4d}  val forced Vm MSE {vm:8.3f} mV^2  c {q['c_mV']:.1f} A_f {q['A_fast']:.1f}/{q['tau_fast']:.0f}ms "
                  f"A_s {q['A_slow']:.1f}/{q['tau_slow']:.0f}ms a {q['a']:.3f} b {q['b']:.4f} ts {q['time_scale']:.4f} "
                  f"Vs {q['Vs']:.3f} V0 {q['V0']:.1f}  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        step(*sall.segments(sets, nets, "train", rng, args.batch))
    cell.set(best_v)

    # ---- step 2: threshold on the free-running spike loss
    @tf.function
    def terms(I, net, V, S, M):
        vm, s = cell.run(cell.current(I, net), V, S, M, sp.SEG_CLAMP)
        vm, s, V, S = vm[:, sp.SEG_CLAMP:], s[:, sp.SEG_CLAMP:], V[:, sp.SEG_CLAMP:], S[:, sp.SEG_CLAMP:]
        t_vm = tf.reduce_mean(tf.square(tf.minimum(vm, sp.V_CLIP) - tf.minimum(V, sp.V_CLIP)))
        t_vr = tf.reduce_mean(tf.square(sp.smooth(s, K) - sp.smooth(S, K)))
        t_n = tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))
        return tf.stack([t_vm, t_vr, t_n])

    def batched(seg, chunk=128):
        n = int(seg[0].shape[0])
        return np.sum([terms(*(a[k:k + chunk] for a in seg)).numpy() * min(chunk, n - k) for k in range(0, n, chunk)], 0) / n

    norm = np.array(json.load(open(os.path.join(sp.OUT, args.model, "metrics.json")))["loss_norm"])
    w = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm      # stage-1 normalization
    names2 = ("V_th", "d_theta", "tau_theta")
    from scipy.optimize import minimize

    def obj2(q):
        cell.set(dict(V_th=q[0], d_theta=q[1], tau_theta=float(np.exp(q[2]))))
        tv = batched(tseg2)
        loss = float(np.sum(w * tv)) if np.all(np.isfinite(tv)) else 1e6
        hist.append(dict(step=2, loss=loss, vm_term=tv[0], vr_term=tv[1], count_term=tv[2], **cell.values()))
        return loss

    best2 = (np.inf, None)
    for vth in (-42.0, -40.0, -38.0):
        q0 = np.array([vth, 1.0, np.log(50.0)])
        simplex = np.vstack([q0, q0 + [1.5, 0, 0], q0 + [0, 1.0, 0], q0 + [0, 0, 0.4]])
        o = minimize(obj2, q0, method="Nelder-Mead", options=dict(maxfev=args.evals2, xatol=1e-3, fatol=1e-4,
                                                                   initial_simplex=simplex))
        if o.fun < best2[0]:
            best2 = (o.fun, o.x)
    q = best2[1]
    cell.set(dict(V_th=q[0], d_theta=q[1], tau_theta=float(np.exp(q[2]))))
    vals = cell.values()
    print("fitted", {k: round(v, 4) for k, v in vals.items()}, f"| train spike loss {best2[0]:.4f} (stage 1: about 1.1-1.5)", flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)

    test = sp.pools(sets, {(pr, "test"): vals["g"] * (sets[(pr, "test")]["I"] + vals["I0"]) + nets[(pr, "test")]
                           for pr in sp.PROTOS}, "test")
    preds, res = sp.evaluate(cell, test)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{pr}_{k}_{q_}": (v.astype(np.float32) if q_ == "vm" else v) for pr, d in preds.items()
                           for k, dd in d.items() for q_, v in dd.items()})
    res = {"model": args.model, "trained": vals, "start": {**{k: p0[k] for k in ("a", "b", "time_scale", "Vs", "V0", "g", "I0")}, **ASC0},
           "val_forced_mse": best, "train_spike_loss": best2[0], "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q_: round(v, 3) for q_, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
