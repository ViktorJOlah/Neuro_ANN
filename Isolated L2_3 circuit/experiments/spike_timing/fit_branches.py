"""Spike-forced refit of the after-spike branches (front end frozen).

Run from "Isolated L2_3 circuit/":
    python experiments/spike_timing/fit_branches.py --cell 382982932 [--start standard] [--tag standard_branches]

Diagnosis (diagnose.py, residual_by_spike.py, README): when the PR 11 fits are given the recorded spike history
(recorded Vm clamped from 2 ms before to 4 ms after each spike), their Vm 20-320 ms after a spike is 6-16 mV
too negative and they hardly fire. Turning SK off flips the error to +4..+11 mV, so SK (scale fitted ~1.2x)
is far too strong; in free runs the fit hides this behind a high reset (c ~ -42 mV) that keeps the cell near
threshold, which makes firing follow the reset rather than the input.

This stage fits, by Nelder-Mead on spike-forced 1000 ms segments (half from noise sweeps, spiking parts
included), the after-spike branch scales (SK, Ca influx, K_P), the spike-triggered u jump d and the constant
input offset I0, minimising Vm MSE with both traces clipped at -40 mV on the free samples (3 ms before every
recorded spike left out). Output: results/<cell>/<tag>/ (same files as fit_standard.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

_thr = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
tf.config.threading.set_intra_op_parallelism_threads(_thr)
tf.config.threading.set_inter_op_parallelism_threads(_thr)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timing_lib as tl  # noqa: E402
import fit_timing as ft  # noqa: E402

sm = tl.sm
L, SEED, N = 1000, 200, 64
Q_SETS = {"v1": ("ln_s_sk", "ln_s_ca", "ln_s_kp", "d"),
          "v2": ("ln_s_sk", "ln_s_ca", "ln_s_kp", "d", "ln_s_im", "ln_s_kv", "ln_s_na", "b_sk")}
STEP = dict(ln_s_sk=0.5, ln_s_ca=0.3, ln_s_kp=0.5, d=0.1, I0=3.0, ln_s_im=1.0, ln_s_kv=0.5, ln_s_na=0.3, b_sk=2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--start", default="standard")
    ap.add_argument("--tag", default="standard_branches")
    ap.add_argument("--evals", type=int, default=150)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--set", choices=tuple(Q_SETS), default="v1", help="v2 adds Im, Kv3.1 and Na scales and an SK bias")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *a: print(f"[{args.cell} {args.tag} {time.time()-t0:5.0f}s]", *a, flush=True)
    res = os.path.join(tl.RC, "results", args.cell)
    cell, p0 = tl.load_fitted(args.cell, os.path.join(res, args.start))
    cell.p["I0"] = tf.Variable(cell.p["I0"])
    sweeps = tl.load_sweeps(args.cell)
    train = [s for s in sweeps if s["split"] == "train"]
    test = [s for s in sweeps if s["split"] == "test"]

    Ie, V = ft.draw(train, np.random.default_rng(3), N, L, 0.5)
    B, T = V.shape
    y = np.concatenate([cell.net(tf.reshape(tf.signal.frame(tf.constant(Ie[k:k + 16]), 64, 1, axis=1), (-1, 64, 1))
                                 / sm.I_SCALE).numpy().reshape(-1, T) for k in range(0, B, 16)])
    g = float(cell.p["g"].numpy())
    S, M = sm.gt_masks(V)
    C = tl.spike_clamp(V, SEED)
    F = ft.loss_mask(V, C)
    F[:, :SEED] = 0
    Ic, Yc, Vc, Sc, Mc, Cc, Fc = map(tf.constant, (Ie[:, 63:], y, V, S, M, C, F))

    @tf.function
    def loss_fn():
        x = cell.p["g"] * (Ic + cell.p["I0"]) + Yc
        vm, s = tl.run_forced(cell, x, Vc, Sc, Mc, Cc)
        e = (tf.minimum(vm, ft.CLIP_A) - tf.minimum(Vc, ft.CLIP_A)) * Fc
        return tf.reduce_sum(tf.square(e)) / tf.reduce_sum(Fc), tf.reduce_sum(s * Fc) / B

    Q_PARAMS = Q_SETS[args.set]
    base = cell.params()
    if base["ln_s_im"] < -10:            # Im switched off in the Standard (scale e^-27.6): start it at a small value
        base["ln_s_im"] = -3.0
    I0_0 = float(cell.p["I0"].numpy())
    hist = []
    names = Q_PARAMS + ("I0",)

    def obj(q):
        qd = dict(zip(names, map(float, q)))
        cell.set({**base, **{k: qd[k] for k in Q_PARAMS}})
        cell.p["I0"].assign(qd["I0"])
        l, n = (a.numpy() for a in loss_fn())
        l = float(l) if np.isfinite(l) else 1e6
        hist.append(dict(it=len(hist), loss=l, false_spikes_per_seg=float(n), **qd))
        return l

    q0 = np.array([base[k] for k in Q_PARAMS] + [I0_0])
    obj(q0)
    log(f"start: forced RMSE {np.sqrt(hist[-1]['loss']):.3f} mV, model spikes/segment {hist[-1]['false_spikes_per_seg']:.2f}")
    simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * STEP[k] for i, k in enumerate(names)])
    o = minimize(obj, q0, method="Nelder-Mead", options=dict(maxfev=args.evals, xatol=1e-3, fatol=1e-4, initial_simplex=simplex))
    obj(o.x)
    log(f"end: forced RMSE {np.sqrt(hist[-1]['loss']):.3f} mV, model spikes/segment {hist[-1]['false_spikes_per_seg']:.2f}",
        {k: round(float(v), 3) for k, v in zip(names, o.x)})
    preds = ft.predict(cell, test, os.path.join(res, args.tag),
                       dict(cell=args.cell, start=args.start, input_rescaling_k=p0.get("input_rescaling_k"),
                            history=hist, wall_s=time.time() - t0))
    log(f"done; Noise 2 Γ {ft.noise_gamma(test, preds):.3f} (spike parameters not refit yet)")


if __name__ == "__main__":
    main()
