"""Spiking fit, stage 6: stage-5 fine-tune with the subthreshold fit anchored.

Run from "Isolated L2_3 circuit/" after spiking.py:
    python experiments/spiking/spiking_anchored.py --model dense3 | cnnlstm | dense_cnnlstm

Stage 5 (spiking_finetune.py) fixed the interspike intervals but degraded the subthreshold fit
(continuous RMSE on spike-free test traces 0.3-0.6 mV -> 3.3-4.4 mV), because only a quarter of
its segments were spike-free and nothing held the weights near their subthreshold values. Here:
  * time_scale is frozen (stage 5 halved it, slowing charging at step onset);
  * each step adds W_SUB x the free-running Vm MSE on SUB_BATCH spike-free segments;
  * L2 penalties keep the ANN weights (L2_ANN, mean squared change) and g, I0, Vs, V0, a, b
    (L2_CELL, squared change relative to the step sizes in ANCHOR_SCALE) near their
    subthreshold values.
Everything else is as in stage 5 (text below).

Stage 5 description:
Stages 1 to 4 kept the ANN frozen. The subthreshold training traces never went above about
-46 mV, so the ANN's current output in the -46 to -40 mV range the real cell creeps through
between spikes is extrapolation, and no Izhikevich-layer change fixed it.

Model: the stage-4 cell (spiking_forced.ForcedCell: threshold spike rule, reset c, u jump d, fast
and slow after-spike currents, adapting threshold), driven by x = g*(I + I0) + ANN(64 ms current
window), the ANN starting from its subthreshold weights.
Step 1 (Adam): ANN weights, g, I0, c, d and the after-currents are trained on spike-forced
    trajectories (spikes forced at the ground-truth times, Vm free in between, no threshold).
    Loss: Vm MSE outside spikes. A quarter of the segments are spike-free. The Izhikevich core
    (a, b, time_scale, Vs, V0) is trained too unless --freeze-core is given (in stage 4 the core
    slowed its time scale about 5x when it was free).
Step 2 (Nelder-Mead): V_th, d_theta, tau_theta on the free-running spike loss of stage 1.
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
import spiking_all as sall  # noqa: E402
import spiking_forced as sf  # noqa: E402

fs = sp.fs
CORE = ("a", "b", "time_scale", "Vs", "V0")
W_SUB, SUB_BATCH, L2_ANN, L2_CELL = 10.0, 8, 100.0, 1.0
ANCHOR_SCALE = dict(g=0.01, I0=1.0, Vs=0.05, V0=2.0, a=0.1, b=0.002)   # raw-units change costing L2_CELL
REST = ("g", "I0", "c_mV", "d", "A_fast", "tau_fast", "A_slow", "tau_slow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("dense3", "cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-ann", type=float, default=3e-4)
    ap.add_argument("--evals2", type=int, default=200)
    args = ap.parse_args()
    tag = args.model
    out = os.path.join(HERE, "results_anchored", tag)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    t0 = time.time()
    sets = fs.load_split()
    m, p0 = sp.load_frozen(args.model)
    g_m, I0_m = float(m.g.numpy()), float(m.I0.numpy())
    cell = sf.ForcedCell(p0, sf.ASC0)
    K = sp.gauss_kernel()
    zero = {(pr, s): np.zeros_like(sets[(pr, s)]["I"]) for pr in sp.PROTOS for s in ("train", "val")}
    rng = np.random.default_rng(0)
    vseg = sall.segments(sets, zero, "val", np.random.default_rng(1), 96)
    tseg2 = sall.segments(sets, zero, "train", np.random.default_rng(2), 192)

    def x_of(I):
        net = m.current(I)[..., 0] - g_m * (I + I0_m)          # the ANN part
        return cell.current(I, net)

    def forced_mse(I, V, S, M):
        vm, _ = cell.run(x_of(I), V, S, M, sp.SEG_CLAMP, forced=True)
        keep = sf.keep_mask(S)[:, sp.SEG_CLAMP:]
        e = (vm[:, sp.SEG_CLAMP:] - V[:, sp.SEG_CLAMP:]) * keep
        return tf.reduce_sum(e * e) / tf.reduce_sum(keep)

    f_mse = tf.function(forced_mse)
    ann_vars0 = [tf.constant(v.numpy()) for v in m.net.trainable_variables]
    raw0 = {k: float(cell.raw[k].numpy()) for k in ANCHOR_SCALE}
    n_ann = float(sum(int(np.prod(v.shape)) for v in ann_vars0))

    def free_segments(split, rng_, n):
        L = sp.SEG_CLAMP + sp.SEG_FREE
        keys = [(pr, i) for pr in sp.PROTOS for i in np.flatnonzero(sets[(pr, split)]["n_spikes"] == 0)
                if sets[(pr, split)]["V"].shape[1] >= L]
        I, V = [], []
        for j in rng_.choice(len(keys), n, replace=False):
            pr, i = keys[j]
            t0 = rng_.integers(0, sets[(pr, split)]["V"].shape[1] - L + 1)
            I.append(sets[(pr, split)]["I"][i, t0:t0 + L]); V.append(sets[(pr, split)]["V"][i, t0:t0 + L])
        I, V = np.stack(I).astype(np.float32), np.stack(V).astype(np.float32)
        Z = np.zeros_like(V)
        return [tf.constant(a) for a in (I, V, Z, Z)]

    def sub_mse(I, V, S, M):
        vm, _ = cell.run(x_of(I), V, S, M, sp.SEG_CLAMP, forced=True)
        return tf.reduce_mean(tf.square(vm[:, sp.SEG_CLAMP:] - V[:, sp.SEG_CLAMP:]))

    def anchor():
        la = tf.add_n([tf.reduce_sum(tf.square(v - v0)) for v, v0 in zip(m.net.trainable_variables, ann_vars0)]) / n_ann
        lc = tf.add_n([tf.square((cell.raw[k] - raw0[k]) / ANCHOR_SCALE[k]) for k in ANCHOR_SCALE])
        return L2_ANN * la + L2_CELL * lc

    s_mse = tf.function(sub_mse)
    vsub = free_segments("val", np.random.default_rng(3), 48)
    # gradients are always taken for every cell parameter (a TF while-loop gradient bug was hit
    # when the core was left out of the tape); frozen ones are simply not updated
    cell_names = REST + CORE
    train_names = REST + tuple(k for k in CORE if k != "time_scale")
    cell_vars = [cell.raw[k] for k in cell_names]
    ann_vars = list(m.net.trainable_variables)
    opt_ann = keras.optimizers.Adam(args.lr_ann)
    opt_ann.build(ann_vars)
    opts = {k: keras.optimizers.Adam(sf.LR[k]) for k in cell_names}
    for k in cell_names:
        opts[k].build([cell.raw[k]])

    @tf.function
    def step(I, V, S, M, Is, Vs_, Ss, Ms):
        with tf.GradientTape() as tape:
            loss = forced_mse(I, V, S, M) + W_SUB * sub_mse(Is, Vs_, Ss, Ms) + anchor()
        grads = tape.gradient(loss, ann_vars + cell_vars)
        grads = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in grads]
        ga, gc = grads[:len(ann_vars)], grads[len(ann_vars):]
        ga, _ = tf.clip_by_global_norm(ga, 10.0)
        opt_ann.apply(ga, ann_vars)
        for k, g, v in zip(cell_names, gc, cell_vars):
            if k in train_names:
                opts[k].apply([tf.clip_by_norm(g, 10.0)], [v])
        return loss

    def seg4(seg):
        I, _, V, S, M = seg
        return I, V, S, M

    def val_mse():
        I, V, S, M = seg4(vseg)
        fm = float(np.mean([f_mse(I[k:k + 32], V[k:k + 32], S[k:k + 32], M[k:k + 32]).numpy() for k in range(0, 96, 32)]))
        sm = float(np.mean([s_mse(*(a[k:k + 24] for a in vsub)).numpy() for k in range(0, 48, 24)]))
        return fm, sm

    hist, best, best_w = [], np.inf, None
    for it in range(args.iters + 1):
        if it % 25 == 0:
            fm, sm = val_mse()
            vm = fm + W_SUB * sm
            if np.isfinite(vm) and vm < best:
                best, best_w = vm, ([v.numpy() for v in ann_vars], cell.values())
            q = cell.values()
            hist.append(dict(step=1, iter=it, val_forced_mse=fm, val_sub_mse=sm, **q))
            print(f"it {it:4d}  val forced Vm MSE {fm:8.3f}  spike-free Vm MSE {sm:6.3f} mV^2  c {q['c_mV']:.1f} A_f {q['A_fast']:.1f}/{q['tau_fast']:.0f}ms "
                  f"A_s {q['A_slow']:.1f}/{q['tau_slow']:.0f}ms ts {q['time_scale']:.4f} g {q['g']:.3f}  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        step(*seg4(sall.segments(sets, zero, "train", rng, args.batch)), *free_segments("train", rng, SUB_BATCH))
    for v, w in zip(ann_vars, best_w[0]):
        v.assign(w)
    cell.set(best_w[1])
    m.save_weights(os.path.join(out, "ann.weights.h5"))
    with open(os.path.join(out, "step1_cell.json"), "w") as f:
        json.dump(dict(val_loss=best, **best_w[1]), f, indent=2)

    # ---- step 2: threshold on the free-running spike loss (ANN fixed now)
    @tf.function
    def terms(I, V, S, M):
        vm, s = cell.run(x_of(I), V, S, M, sp.SEG_CLAMP)
        vm, s, V, S = vm[:, sp.SEG_CLAMP:], s[:, sp.SEG_CLAMP:], V[:, sp.SEG_CLAMP:], S[:, sp.SEG_CLAMP:]
        t_vm = tf.reduce_mean(tf.square(tf.minimum(vm, sp.V_CLIP) - tf.minimum(V, sp.V_CLIP)))
        t_vr = tf.reduce_mean(tf.square(sp.smooth(s, K) - sp.smooth(S, K)))
        t_n = tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))
        return tf.stack([t_vm, t_vr, t_n])

    I2, V2, S2, M2 = seg4(tseg2)

    def batched():
        return np.mean([terms(I2[k:k + 64], V2[k:k + 64], S2[k:k + 64], M2[k:k + 64]).numpy() for k in range(0, 192, 64)], 0)

    norm = np.array(json.load(open(os.path.join(sp.OUT, args.model, "metrics.json")))["loss_norm"])
    w = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm
    from scipy.optimize import minimize

    def obj2(q):
        cell.set(dict(V_th=q[0], d_theta=q[1], tau_theta=float(np.exp(q[2]))))
        tv = batched()
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
    print("fitted", {k: round(v, 4) for k, v in vals.items()}, f"| train spike loss {best2[0]:.4f}", flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)

    test = sp.pools(sets, {(pr, "test"): np.concatenate([x_of(tf.constant(sets[(pr, "test")]["I"][k:k + 4])).numpy()
                                                          for k in range(0, len(sets[(pr, "test")]["I"]), 4)])
                           for pr in sp.PROTOS}, "test")
    preds, res = sp.evaluate(cell, test)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{pr}_{k}_{q_}": (v.astype(np.float32) if q_ == "vm" else v) for pr, d in preds.items()
                           for k, dd in d.items() for q_, v in dd.items()})
    sub = {}
    for pr in ("square", "noise"):
        free = sets[(pr, "test")]["n_spikes"] == 0
        e = (preds[pr]["cont"]["vm"][free] - sets[(pr, "test")]["V"][free])[:, sp.base.SEED_MS:]
        sub[pr] = float(np.sqrt(np.mean(e ** 2)))
    print("continuous RMSE on spike-free test traces:", sub, flush=True)
    res = {"model": args.model, "trained": vals, "val_loss_step1": best, "train_spike_loss": best2[0],
           "spike_free_cont_rmse_mV": sub, "weights": dict(W_SUB=W_SUB, L2_ANN=L2_ANN, L2_CELL=L2_CELL), "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q_: round(v, 3) for q_, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
