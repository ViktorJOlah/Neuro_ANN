"""Timing refit of the Standard on a real Allen L2/3 recording (starts from the PR 11 fit in experiments/real_cells).

Run from "Isolated L2_3 circuit/":
    python experiments/spike_timing/fit_timing.py --cell 318808427

Why: the PR 11 fit trained the front end only on Vm before the first spike of each sweep (so almost none of the
20 s noise sweeps) and then tuned spike parameters on a 10 ms-smoothed loss. Its spike timing on Noise 2 was
poor (Γ 0.12). Same architecture (CNN-LSTM front end, Izhikevich top, conductance branches, 1 ms, 64 ms current
window); only the training changes:

Stage A, spike-forced front-end fit. Segments come from *all* training sweeps (half from the noise sweeps),
spiking parts included. Around every recorded spike (2 ms before to 4 ms after onset) the recorded Vm and
spike drive the Izhikevich state and the branches (the same clamp the seed period uses), so the model always
has the true spike history; between spikes it runs freely. Loss: Vm MSE with both traces clipped at -40 mV
(near the real spike threshold) on the free samples, leaving out 3 ms before each recorded spike. Trained: the
CNN-LSTM weights, the Izhikevich g, I0, V0, Vs, a, b, time_scale and the branch scales of Na, SK, K_P, Im and d.

Stage B, timing-weighted spike fit. Front end frozen; Nelder-Mead over V_th, c, d, Na bias, Na and SK scales
on free-running 1500 ms segments (500 ms seed, 3/4 from noise sweeps). Loss = 3 ms-smoothed spike-train MSE
(tighter than PR 11's 10 ms) + 0.1 x spike-count MSE + 0.5 x clipped (-40 mV) Vm MSE, each normalised at start.

Output in experiments/real_cells/results/<cell>/: standard_timing/ (stage A + B) and standard_timingA/ (stage A
only, PR 11 spike parameters), same files as fit_standard.py, so code/compare.py scores them with the others.
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

sm = tl.sm
A_LEN, A_SEED, A_BATCH = 1000, 200, 32
B_LEN, B_SEED, B_N = 1500, 500, 96
CLIP_A = -40.0
B_SIGMA = 3.0
S1_IZH = ("g", "I0", "V0", "Vs", "a", "b", "time_scale")
A_BRANCH = ("ln_s_na", "ln_s_sk", "ln_s_kp", "ln_s_im", "d")
B_PARAMS = ("V_th", "c_mV", "d", "b_na", "ln_s_na", "ln_s_sk")
NM_STEP = dict(V_th=1.5, c_mV=3.0, d=0.1, b_na=3.0, ln_s_na=0.3, ln_s_sk=0.4)


def seg(s, start, L):
    I, V = s["I"], s["V"]
    pre = I[max(0, start - 63):start]
    pre = np.concatenate([np.full(63 - len(pre), I[0], np.float32), pre])
    return np.concatenate([pre, I[start:start + L]]), V[start:start + L]


def draw(train, rng, n, L, noise_frac):
    noise = [s for s in train if "Noise" in s["stimulus"] and len(s["V"]) >= L]
    other = [s for s in train if "Noise" not in s["stimulus"] and len(s["V"]) >= L]
    Ie, Ve = [], []
    for k in range(n):
        pool = noise if (noise and (rng.random() < noise_frac or not other)) else other
        s = pool[rng.integers(len(pool))]
        a, b = seg(s, int(rng.integers(0, len(s["V"]) - L + 1)), L)
        Ie.append(a); Ve.append(b)
    return np.stack(Ie).astype(np.float32), np.stack(Ve).astype(np.float32)


def loss_mask(V, clamp, pre=3):
    F = 1.0 - clamp
    for i in range(len(V)):
        for t0 in tl.gt_spike_ms(V[i]):
            F[i, max(0, t0 - pre):t0] = 0
    return F.astype(np.float32)


def predict(cell, test, out, params_extra):
    os.makedirs(out, exist_ok=True)
    preds = {}
    for s in test:
        vm, spk = cell.simulate(s["I"][None], V_seed=s["V"][None], seed_ms=tl.SEED_MS)
        preds[f"s{s['sweep']}_vm"] = vm[0].astype(np.float32)
        preds[f"s{s['sweep']}_spk"] = np.flatnonzero(spk[0]).astype(np.float64)
    np.savez_compressed(os.path.join(out, "predictions.npz"), **preds)
    np.savez_compressed(os.path.join(out, "front_end.npz"), **{f"w{i}": a for i, a in enumerate(cell.net.get_weights())})
    json.dump(dict(izhikevich={k: float(v.numpy()) for k, v in cell.p.items()}, branches=cell.params(), **params_extra),
              open(os.path.join(out, "params.json"), "w"), indent=1)
    return preds


def noise_gamma(test, preds):
    g = []
    for s in test:
        if s["stimulus"] != "Noise 2":
            continue
        T = len(s["V"])
        td = tl.gt_spike_ms(s["V"]); td = td[td >= tl.SEED_MS]
        tm = preds[f"s{s['sweep']}_spk"]; tm = tm[tm >= tl.SEED_MS]
        g.append(tl.gamma(td, tm, T - tl.SEED_MS))
    return float(np.nanmean(g)) if g else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--evals", type=int, default=200)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="standard_timing")
    ap.add_argument("--start", default="standard", help="results/<cell>/<start>/ to begin from")
    ap.add_argument("--branch-lr", type=float, default=0.0,
                    help="absolute Adam lr for the branch log-scales and d (0 = 0.2 %% of the value, as in round 1)")
    ap.add_argument("--stage-a-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *a: print(f"[{args.cell} {time.time()-t0:5.0f}s]", *a, flush=True)
    res = os.path.join(tl.RC, "results", args.cell)

    sweeps = tl.load_sweeps(args.cell)
    train = [s for s in sweeps if s["split"] == "train"]
    test = [s for s in sweeps if s["split"] == "test"]
    cell, p0 = tl.load_fitted(args.cell, os.path.join(tl.RC, "results", args.cell, args.start))
    cell.p = {k: tf.Variable(v, name=k) for k, v in cell.p.items()}
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    hist = []

    # ---------------- stage A: spike-forced front-end fit
    n_net = len(cell.net.trainable_variables)
    tv = cell.net.trainable_variables + [cell.p[k] for k in S1_IZH] + [cell.q[k] for k in A_BRANCH]
    opts = []
    for j, v in enumerate(tv):
        lr = 3e-4 if j < n_net else 0.002 * max(abs(float(v.numpy())), 0.05)
        if j >= n_net + len(S1_IZH) and args.branch_lr > 0:
            lr = args.branch_lr
        o = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(lr, args.iters, alpha=0.1))
        o.build([v])
        opts.append(o)

    def ext_windows(Ie):
        return tf.signal.frame(Ie, 64, 1, axis=1)

    @tf.function(reduce_retracing=True)
    def a_step(Ie, V, S, M, C, F):
        with tf.GradientTape() as tape:
            B, T = tf.shape(V)[0], tf.shape(V)[1]
            y = cell.net(tf.reshape(ext_windows(Ie), (-1, 64, 1)) / sm.I_SCALE)
            x = cell.p["g"] * (Ie[:, 63:] + cell.p["I0"]) + tf.reshape(y, (B, T))
            vm, _ = tl.run_forced(cell, x, V, S, M, C)
            e = (tf.minimum(vm, CLIP_A) - tf.minimum(V, CLIP_A)) * F
            loss = tf.reduce_sum(tf.square(e)) / tf.reduce_sum(F)
        grads = tape.gradient(loss, tv)
        for g, v, o in zip(grads, tv, opts):
            g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
            o.apply([tf.clip_by_norm(g, 100.0)], [v])
        return loss

    def a_batch(r, n):
        Ie, V = draw(train, r, n, A_LEN, 0.5)
        S, M = sm.gt_masks(V)
        C = tl.spike_clamp(V, A_SEED)
        F = loss_mask(V, C)
        return Ie, V, S, M, C, F

    val = a_batch(np.random.default_rng(99), 32)

    @tf.function(reduce_retracing=True)
    def a_eval(Ie, V, S, M, C, F):
        B, T = tf.shape(V)[0], tf.shape(V)[1]
        y = cell.net(tf.reshape(ext_windows(Ie), (-1, 64, 1)) / sm.I_SCALE)
        x = cell.p["g"] * (Ie[:, 63:] + cell.p["I0"]) + tf.reshape(y, (B, T))
        vm, _ = tl.run_forced(cell, x, V, S, M, C)
        e = (tf.minimum(vm, CLIP_A) - tf.minimum(V, CLIP_A)) * F
        return tf.reduce_sum(tf.square(e)) / tf.reduce_sum(F)

    v0 = float(a_eval(*map(tf.constant, val)))
    log(f"stage A start: spike-forced val RMSE {np.sqrt(v0):.3f} mV")
    for it in range(args.iters):
        loss = float(a_step(*map(tf.constant, a_batch(rng, A_BATCH))))
        hist.append(dict(stage="A", it=it, loss=loss))
        if it % 10 == 0 or it == args.iters - 1:
            vl = float(a_eval(*map(tf.constant, val)))
            hist[-1]["val"] = vl
            log(f"stage A it {it:3d}  train {np.sqrt(loss):.3f}  val {np.sqrt(vl):.3f} mV")

    predsA = predict(cell, test, os.path.join(res, args.tag + "A"), dict(cell=args.cell, start=args.start, history=hist,
                     input_rescaling_k=p0.get("input_rescaling_k")))
    log(f"stage A done; Noise 2 Γ {noise_gamma(test, predsA):.3f}", {k: round(cell.params()[k], 3) for k in A_BRANCH})
    if args.stage_a_only:
        return

    # ---------------- stage B: timing-weighted spike fit
    Ie, V2 = draw(train, np.random.default_rng(1), B_N, B_LEN, 0.75)
    B, T = V2.shape
    y = np.concatenate([cell.net(tf.reshape(ext_windows(tf.constant(Ie[k:k + 16])), (-1, 64, 1)) / sm.I_SCALE)
                        .numpy().reshape(-1, T) for k in range(0, B, 16)])
    x2 = (float(cell.p["g"].numpy()) * (Ie[:, 63:] + float(cell.p["I0"].numpy())) + y).astype(np.float32)
    S2, M2 = sm.gt_masks(V2)
    kk = np.arange(-int(4 * B_SIGMA), int(4 * B_SIGMA) + 1)
    K = tf.constant(np.exp(-0.5 * (kk / B_SIGMA) ** 2)[:, None, None], tf.float32)
    smo = lambda s: tf.nn.conv1d(s[..., None], K, 1, "SAME")[..., 0]
    C2 = tl.spike_clamp(V2, B_SEED, 0, -1)
    Xc, Vc, Sc, Mc, Cc = map(tf.constant, (x2, V2, S2, M2, C2))

    @tf.function
    def terms():
        vm, s = tl.run_forced(cell, Xc, Vc, Sc, Mc, Cc)
        vm, s, V_, S_ = vm[:, B_SEED:], s[:, B_SEED:], Vc[:, B_SEED:], Sc[:, B_SEED:]
        return tf.stack([tf.reduce_mean(tf.square(smo(s) - smo(S_))),
                         tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S_, 1))),
                         tf.reduce_mean(tf.square(tf.minimum(vm, CLIP_A) - tf.minimum(V_, CLIP_A)))])

    base = cell.params()
    norm = np.maximum(terms().numpy(), 1e-6)
    wl = np.array([1.0, 0.1, 0.5]) / norm

    def obj(q):
        qd = dict(zip(B_PARAMS, map(float, q)))
        if qd["c_mV"] > qd["V_th"] - 3.0:
            return 1e3 + qd["c_mV"] - qd["V_th"]
        cell.set({**base, **qd})
        t_ = terms().numpy()
        loss = float(np.sum(wl * t_)) if np.all(np.isfinite(t_)) else 1e6
        hist.append(dict(stage="B", it=len(hist), loss=loss, **qd))
        return loss

    q0 = np.array([base[k] for k in B_PARAMS])
    simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * NM_STEP[k] for i, k in enumerate(B_PARAMS)])
    log("stage B start loss", round(obj(q0), 4))
    o = minimize(obj, q0, method="Nelder-Mead", options=dict(maxfev=args.evals, xatol=1e-2, fatol=1e-4, initial_simplex=simplex))
    cell.set({**base, **dict(zip(B_PARAMS, map(float, o.x)))})
    log("stage B loss", round(float(o.fun), 4), {k: round(float(v), 3) for k, v in zip(B_PARAMS, o.x)})

    preds = predict(cell, test, os.path.join(res, args.tag),
                    dict(cell=args.cell, input_rescaling_k=p0.get("input_rescaling_k"), history=hist, wall_s=time.time() - t0))
    log(f"done; Noise 2 Γ {noise_gamma(test, preds):.3f} (PR 11 fit: see results/metrics_summary.csv)")


if __name__ == "__main__":
    main()
