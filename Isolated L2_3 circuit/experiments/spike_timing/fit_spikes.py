"""Spike-parameter refit (front end frozen) with a direct coincidence objective.

Run from "Isolated L2_3 circuit/":
    python experiments/spike_timing/fit_spikes.py --cell 382982932 --start standard_timingA --tag standard_timing

Smoothed spike-train MSE (PR 11 uses 10 ms, fit_timing.py stage B tried 3 ms) rewards a model that fires less
when its timing is off, so the fits under-fire. Here Nelder-Mead maximises Γ (±4 ms, the score we report)
directly on free-running training segments, with a spike-count penalty:
    loss = -mean Γ(segment) + W_COUNT * mean |n_model - n_data| / (n_data + 1)
Segments: 2000 ms (500 ms recorded-Vm seed), 3/4 from Noise 1 sweeps, the rest from the other training sweeps.
The last Noise 1 sweep is kept out of the segments and used only as a validation trace (free run of the whole
20 s sweep), so the choice between objectives never looks at the Noise 2 test sweeps.
Parameters: V_th, c, d, Na bias, Na and SK scales, and the slow spike-triggered threshold (th_d, th_tau).
"""
from __future__ import annotations

import argparse
import json
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
L, SEED, N = 2000, 500, 64
PARAMS = ("V_th", "c_mV", "d", "b_na", "ln_s_na", "ln_s_sk", "th_d", "th_tau")
STEP = dict(V_th=1.5, c_mV=3.0, d=0.1, b_na=3.0, ln_s_na=0.3, ln_s_sk=0.4, th_d=1.0, th_tau=20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--start", default="standard_timingA", help="results/<cell>/<start>/ to begin from")
    ap.add_argument("--tag", default="standard_timing")
    ap.add_argument("--evals", type=int, default=250)
    ap.add_argument("--w-count", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mode", choices=("free", "forced", "smooth"), default="free",
                    help="smooth: the PR 11 stage-2 loss (clipped -20 mV Vm MSE + 10 ms smoothed spike MSE + 0.1 x count MSE, "
                         "each normalised at start) on free runs, over V_th, c, d, Na bias and Na scale (SK stays as fitted). forced: every recorded spike is imposed (spike, reset to c, branch kick) so each model spike is "
                         "scored given the true history; the loss then also has the clipped Vm MSE")
    ap.add_argument("--w-vm", type=float, default=1.0)
    ap.add_argument("--smooth-count", type=float, default=0.1, help="smooth mode: weight of the count term")
    ap.add_argument("--smooth-sk", action="store_true", help="smooth mode: also fit the SK scale")
    ap.add_argument("--coinc", type=int, default=0, help="forced mode: skip imposing a recorded spike when the model "
                    "spiked within this many ms before it")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *a: print(f"[{args.cell} {args.tag} {time.time()-t0:5.0f}s]", *a, flush=True)
    res = os.path.join(tl.RC, "results", args.cell)
    cell, p0 = tl.load_fitted(args.cell, os.path.join(res, args.start))
    sweeps = tl.load_sweeps(args.cell)
    train = [s for s in sweeps if s["split"] == "train"]
    test = [s for s in sweeps if s["split"] == "test"]
    n1 = [s for s in train if s["stimulus"] == "Noise 1"]
    val = n1[-1]
    fit_pool = [s for s in train if s is not val]

    Ie, V = ft.draw(fit_pool, np.random.default_rng(2), N, L, 0.75)
    B, T = V.shape
    y = np.concatenate([cell.net(tf.reshape(tf.signal.frame(tf.constant(Ie[k:k + 16]), 64, 1, axis=1), (-1, 64, 1))
                                 / sm.I_SCALE).numpy().reshape(-1, T) for k in range(0, B, 16)])
    x = (float(cell.p["g"].numpy()) * (Ie[:, 63:] + float(cell.p["I0"].numpy())) + y).astype(np.float32)
    S, M = sm.gt_masks(V)
    C = tl.spike_clamp(V, SEED, 0, 0) if args.mode == "forced" else tl.spike_clamp(V, SEED, 0, -1)
    F = 1.0 - C
    F[:, :SEED] = 0
    Fc = tf.constant(F.astype(np.float32))
    Xc, Vc, Sc, Mc, Cc = map(tf.constant, (x, V, S, M, C))
    td = [tl.gt_spike_ms(v) for v in V]
    td = [t[t >= SEED] for t in td]
    xv = cell.current(val["I"][None])
    tdv = tl.gt_spike_ms(val["V"]); tdv = tdv[tdv >= tl.SEED_MS]

    kk = np.arange(-40, 41)
    K = tf.constant(np.exp(-0.5 * (kk / 10.0) ** 2)[:, None, None], tf.float32)
    smo = lambda a: tf.nn.conv1d(a[..., None], K, 1, "SAME")[..., 0]

    @tf.function
    def smooth_terms():
        vm, s = tl.run_forced(cell, Xc, Vc, Sc, Mc, Cc)
        vm, s, V_, S_ = vm[:, SEED:], s[:, SEED:], Vc[:, SEED:], Sc[:, SEED:]
        return tf.stack([tf.reduce_mean(tf.square(tf.minimum(vm, -20.0) - tf.minimum(V_, -20.0))),
                         tf.reduce_mean(tf.square(smo(s) - smo(S_))),
                         tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S_, 1)))])

    def score():
        vm, s = tl.run_forced(cell, Xc, Vc, Sc, Mc, Cc, args.coinc)
        e = (tf.minimum(vm, ft.CLIP_A) - tf.minimum(Vc, ft.CLIP_A)) * Fc
        mse = float(tf.reduce_sum(tf.square(e)) / tf.reduce_sum(Fc))
        s = s.numpy() > 0.5
        g, ce = [], []
        for i in range(B):
            tm = np.flatnonzero(s[i]); tm = tm[tm >= SEED]
            if len(td[i]) or len(tm):
                g.append(tl.gamma(td[i], tm, T - SEED))
            ce.append(abs(len(tm) - len(td[i])) / (len(td[i]) + 1))
        return float(np.mean(g)), float(np.mean(ce)), mse

    def val_gamma():
        _, s = tl.simulate(cell, val["I"][None], val["V"][None], tl.spike_clamp(val["V"][None], tl.SEED_MS, 0, -1), x=xv)
        tm = np.flatnonzero(s[0]); tm = tm[tm >= tl.SEED_MS]
        return tl.gamma(tdv, tm, len(val["V"]) - tl.SEED_MS), len(tm), len(tdv)

    base = cell.params()
    hist = []

    def obj(q):
        qd = dict(zip(PARAMS, map(float, q)))
        if qd["c_mV"] > qd["V_th"] - 3.0 or qd.get("th_tau", 30.0) < 2.0 or qd.get("th_d", 0.0) < 0:
            return 10.0
        cell.set({**base, **qd})
        if args.mode == "smooth":
            t_ = smooth_terms().numpy()
            loss = float(np.sum(wsm * t_)) if np.all(np.isfinite(t_)) else 1e6
            hist.append(dict(it=len(hist), loss=loss, **qd))
            return loss
        g, ce, mse = score()
        loss = -g + args.w_count * ce + (args.w_vm * mse / mse0[0] if args.mode == "forced" else 0.0)
        hist.append(dict(it=len(hist), loss=loss, gamma=g, count_err=ce, vm_mse=mse, **qd))
        return loss

    global PARAMS
    if args.mode == "smooth":
        PARAMS = ("V_th", "c_mV", "d", "b_na", "ln_s_na") + (("ln_s_sk",) if args.smooth_sk else ())
        wsm = np.array([1.0, 1.0, args.smooth_count]) / np.maximum(smooth_terms().numpy(), 1e-6)
    q0 = np.array([base[k] for k in PARAMS])
    mse0 = [1.0]
    mse0[0] = score()[2]
    l0 = obj(q0)
    v0 = val_gamma()
    log(f"start: train Γ {score()[0]:.3f}; val Γ {v0[0]:.3f} ({v0[2]} -> {v0[1]} spikes)")
    simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * STEP[k] for i, k in enumerate(PARAMS)])
    o = minimize(obj, q0, method="Nelder-Mead", options=dict(maxfev=args.evals, xatol=1e-2, fatol=1e-4, initial_simplex=simplex))
    cell.set({**base, **dict(zip(PARAMS, map(float, o.x)))})
    g, ce, _ = score()
    v1 = val_gamma()
    log(f"end: train Γ {g:.3f} count err {ce:.3f}; val Γ {v1[0]:.3f} ({v1[2]} -> {v1[1]} spikes)",
        {k: round(float(v), 3) for k, v in zip(PARAMS, o.x)})
    preds = ft.predict(cell, test, os.path.join(res, args.tag),
                       dict(cell=args.cell, start=args.start, mode=args.mode, smooth_count=args.smooth_count, smooth_sk=args.smooth_sk, coinc=args.coinc, w_vm=args.w_vm, w_count=args.w_count, val_start=v0, val_end=v1,
                            input_rescaling_k=p0.get("input_rescaling_k"), history=hist, wall_s=time.time() - t0))
    log(f"done; Noise 2 Γ {ft.noise_gamma(test, preds):.3f}")


if __name__ == "__main__":
    main()
