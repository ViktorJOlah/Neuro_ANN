"""Fit the Standard model to one cell's real Allen recording (train sweeps of data/<cell>/dataset.json).

Run from "Isolated L2_3 circuit/":
    python experiments/real_cells/code/fit_standard.py --cell 513531471 [--standard <Standard dir>]

Same recipe as experiments/data_budget/fit_budget.py (that script's settings and helpers are imported),
so a fit here is directly comparable with the data-budget fits:
  * input rescaling k = 83.8 pA / rheobase, rheobase = the lowest training Long Square step that spikes,
    folded exactly into the weights;
  * stage 1: CNN-LSTM + Izhikevich (g, I0, V0, Vs, a, b, time_scale), closed loop on 600 ms subthreshold
    segments, 250 Adam steps;
  * stage 2: Nelder-Mead over V_th, c, d, Na bias and SK scale on 1500 ms segments (2/3 holding a spike).
Sweeps have different lengths here, so segments are drawn per sweep.

Output: results/<cell>/standard/ (and results/<cell>/standard_unfitted/ with --unfitted):
  predictions.npz  1 ms model Vm and spikes per test sweep (keys s<sweep>_vm, s<sweep>_spk)
  params.json      k, fitted Izhikevich and branch parameters, training history
  front_end.npz    fitted CNN-LSTM weights
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

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(RC))
sys.path.insert(0, os.path.join(ROOT, "experiments", "data_budget"))
import fit_budget as fb  # noqa: E402

PROTOS = fb.PROTOS
SEED_MS = fb.SEED_MS


def load(cell):
    meta = json.load(open(os.path.join(RC, "data", cell, "dataset.json")))
    z = np.load(os.path.join(RC, "data", cell, "sweeps_1ms.npz"))
    sw = []
    for r in meta["sweeps"]:
        k = f"s{r['sweep']}"
        sw.append(dict(r, I=z[k + "_I"], V=z[k + "_V"], spk_ms=z[k + "_spk_ms"]))
    return sw


def seg(s, start, L):
    I, V = s["I"], s["V"]
    pre = I[max(0, start - 63):start]
    pre = np.concatenate([np.full(63 - len(pre), I[0], np.float32), pre])
    return np.concatenate([pre, I[start:start + L]]), V[start:start + L]


def first_spike(V):
    s = np.flatnonzero(fb.gt_spikes(V[None])[0])
    return s[0] if len(s) else None


def stage1_segments(train, rng, n):
    cands = []
    for s in train:
        f = first_spike(s["V"])
        end = (f - 10) if f is not None else len(s["V"])
        if end >= fb.S1_LEN:
            cands.append((s, end))
    Ie, Ve = [], []
    for j in rng.integers(len(cands), size=n):
        s, end = cands[j]
        a, b = seg(s, int(rng.integers(0, end - fb.S1_LEN + 1)), fb.S1_LEN)
        Ie.append(a); Ve.append(b)
    return np.stack(Ie), np.stack(Ve)


def stage2_segments(train, rng, n):
    L = fb.S2_LEN
    ok = [s for s in train if len(s["V"]) >= L]
    spk = [(s, np.flatnonzero(fb.gt_spikes(s["V"][None])[0])) for s in ok]
    spk = [(s, t) for s, t in spk if len(t)]
    Ie, Ve = [], []
    for k in range(n):
        if spk and k % 3 != 2:
            s, t = spk[rng.integers(len(spk))]
            tt = int(t[rng.integers(len(t))])
            st = int(np.clip(tt - fb.S2_SEED - rng.integers(0, L - fb.S2_SEED), 0, len(s["V"]) - L))
        else:
            s = ok[rng.integers(len(ok))]
            st = int(rng.integers(0, len(s["V"]) - L + 1))
        a, b = seg(s, st, L)
        Ie.append(a); Ve.append(b)
    return np.stack(Ie), np.stack(Ve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--standard", default=os.path.join(ROOT, "Standard"))
    ap.add_argument("--out", default=os.path.join(RC, "results"))
    ap.add_argument("--iters", type=int, default=fb.S1_ITERS)
    ap.add_argument("--evals", type=int, default=fb.S2_EVALS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--unfitted", action="store_true", help="only evaluate the Standard as it is")
    args = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    sys.path.insert(0, args.standard)
    import standard_model as sm

    tag = "standard_unfitted" if args.unfitted else "standard"
    out = os.path.join(args.out, args.cell, tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{args.cell} {tag} {time.time()-t0:5.0f}s]", *a, flush=True)

    sweeps = load(args.cell)
    train = [s for s in sweeps if s["split"] == "train"]
    test = [s for s in sweeps if s["split"] == "test"]
    log(f"train {len(train)} sweeps ({sum(len(s['I']) for s in train)/6e4:.2f} min), test {len(test)} sweeps")

    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    cell = sm.load(args.standard)
    cell.p = {k: tf.Variable(v, name=k) for k, v in cell.p.items()}
    hist, k_in = [], 1.0

    if not args.unfitted:
        ls = [s for s in train if s["stimulus"] == "Long Square" and s["amp_pA"] > 0 and s["n_spikes"] > 0]
        rheo = min(s["amp_pA"] for s in ls)
        k_in = fb.STANDARD_RHEO_PA / rheo
        w = cell.net.layers[0].kernel
        w.assign(w * k_in)
        cell.p["g"].assign(cell.p["g"] * k_in)
        cell.p["I0"].assign(cell.p["I0"] / k_in)
        log(f"input rescaling k = {k_in:.3f} (rheobase {rheo:.0f} pA)")

        # ---------------- stage 1
        n_net = len(cell.net.trainable_variables)
        tv = cell.net.trainable_variables + [cell.p[k] for k in fb.S1_PARAMS]
        opts = []
        for j, v in enumerate(tv):
            lr = 1e-3 if j < n_net else 0.005 * max(abs(float(v.numpy())), 1e-3)
            o = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(lr, args.iters, alpha=0.1))
            o.build([v])
            opts.append(o)

        @tf.function(reduce_retracing=True)
        def s1_step(Ie, V):
            S = tf.zeros_like(V)
            with tf.GradientTape() as tape:
                B, T = tf.shape(V)[0], tf.shape(V)[1]
                y = cell.net(tf.reshape(fb.ext_windows(Ie), (-1, 64, 1)) / sm.I_SCALE)
                x = cell.p["g"] * (Ie[:, 63:] + cell.p["I0"]) + tf.reshape(y, (B, T))
                vm, _ = cell.run(x, V, S, S, tf.constant(fb.S1_SEED), False)
                e = tf.minimum(vm[:, fb.S1_SEED:], fb.CLIP) - tf.minimum(V[:, fb.S1_SEED:], fb.CLIP)
                loss = tf.reduce_mean(tf.square(e))
            grads = tape.gradient(loss, tv)
            for g, v, o in zip(grads, tv, opts):
                g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                o.apply([tf.clip_by_norm(g, 100.0)], [v])
            return loss

        for it in range(args.iters):
            Ie, V = stage1_segments(train, rng, fb.S1_BATCH)
            loss = float(s1_step(tf.constant(Ie), tf.constant(V)))
            hist.append(dict(stage=1, it=it, loss=loss))
            if it % 20 == 0 or it == args.iters - 1:
                log(f"stage 1 it {it:3d}  train Vm RMSE {np.sqrt(loss):.3f} mV")

        # ---------------- stage 2
        Ie, V2 = stage2_segments(train, np.random.default_rng(1), fb.S2_N)
        B, T = V2.shape
        y = np.concatenate([cell.net(tf.reshape(fb.ext_windows(tf.constant(Ie[k:k + 16])), (-1, 64, 1)) / sm.I_SCALE)
                            .numpy().reshape(-1, T) for k in range(0, B, 16)])
        x2 = (float(cell.p["g"].numpy()) * (Ie[:, 63:] + float(cell.p["I0"].numpy())) + y).astype(np.float32)
        S2, M2 = fb.masks(V2)
        kk = np.arange(-int(4 * fb.SIGMA), int(4 * fb.SIGMA) + 1)
        K = tf.constant(np.exp(-0.5 * (kk / fb.SIGMA) ** 2)[:, None, None], tf.float32)
        smo = lambda s: tf.nn.conv1d(s[..., None], K, 1, "SAME")[..., 0]
        Xc, Vc, Sc, Mc = map(tf.constant, (x2, V2, S2, M2))
        SD = fb.S2_SEED

        @tf.function
        def terms():
            vm, s = cell.run(Xc, Vc, Sc, Mc, tf.constant(SD), False)
            vm, s, V_, S_ = vm[:, SD:], s[:, SD:], Vc[:, SD:], Sc[:, SD:]
            return tf.stack([tf.reduce_mean(tf.square(tf.minimum(vm, fb.CLIP) - tf.minimum(V_, fb.CLIP))),
                             tf.reduce_mean(tf.square(smo(s) - smo(S_))),
                             tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S_, 1)))])

        base = cell.params()
        norm = np.maximum(terms().numpy(), 1e-6)
        wl = np.array([1.0, fb.W_SPIKE, fb.W_COUNT]) / norm

        def obj(q):
            qd = dict(zip(fb.S2_PARAMS, map(float, q)))
            if qd["c_mV"] > qd["V_th"] - 3.0:
                return 1e3 + qd["c_mV"] - qd["V_th"]
            cell.set({**base, **qd})
            t_ = terms().numpy()
            loss = float(np.sum(wl * t_)) if np.all(np.isfinite(t_)) else 1e6
            hist.append(dict(stage=2, it=len(hist), loss=loss, **qd))
            return loss

        q0 = np.array([base[k] for k in fb.S2_PARAMS])
        simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * fb.NM_STEP[k] for i, k in enumerate(fb.S2_PARAMS)])
        log("stage 2 start loss", round(obj(q0), 4))
        o = minimize(obj, q0, method="Nelder-Mead",
                     options=dict(maxfev=args.evals, xatol=1e-2, fatol=1e-4, initial_simplex=simplex))
        cell.set({**base, **dict(zip(fb.S2_PARAMS, map(float, o.x)))})
        log("stage 2 loss", round(float(o.fun), 4), {k: round(float(v), 3) for k, v in zip(fb.S2_PARAMS, o.x)})

    # ---------------- predictions on the test sweeps (recorded Vm seed for the first 450 ms = pre-stimulus)
    preds = {}
    for s in test:
        vm, spk = cell.simulate(s["I"][None], V_seed=s["V"][None], seed_ms=SEED_MS)
        preds[f"s{s['sweep']}_vm"] = vm[0].astype(np.float32)
        preds[f"s{s['sweep']}_spk"] = np.flatnonzero(spk[0]).astype(np.float64)    # spike times, ms
        log(f"test sweep {s['sweep']} ({s['stimulus']}, {s['amp_pA']:.0f} pA): spikes {s['n_spikes']} -> {int(spk[0].sum())}")
    np.savez_compressed(os.path.join(out, "predictions.npz"), **preds)
    if not args.unfitted:
        np.savez_compressed(os.path.join(out, "front_end.npz"),
                            **{f"w{i}": a for i, a in enumerate(cell.net.get_weights())})
    json.dump(dict(cell=args.cell, input_rescaling_k=k_in, izhikevich={k: float(v.numpy()) for k, v in cell.p.items()},
                   branches=cell.params(), history=hist, train_minutes=sum(len(s["I"]) for s in train) / 6e4,
                   wall_s=time.time() - t0), open(os.path.join(out, "params.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
