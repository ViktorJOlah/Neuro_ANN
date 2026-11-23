"""Fit the Standard model to one of the 5 extra Allen L2/3 PCs on a fixed recording budget.

How much patch-clamp data does a cell need? Each fit starts from the same initial condition (the
Standard model: weights, Izhikevich parameters and branch values, same seeds, same number of steps)
and sees only `--minutes` of the cell's *noisy* simulations, mixed from the three protocols.

Run from "Isolated L2_3 circuit/":
    python experiments/data_budget/fit_budget.py --cell 521409057 --minutes 5     # 0, 5, 10, 20, 30 or full

Data (simulations/additional_cells/data/<cell>/<protocol>_noisy.npy, 100 traces per protocol):
  * test: 15 traces per protocol, evenly spread over the amplitude range, the same for every fit.
  * training pool: the other 85 per protocol. A budget of B minutes takes B/3 of time from each protocol,
    traces in a fixed random order per cell, so smaller budgets are nested in larger ones. If a protocol
    runs out, the rest is spread over the others. A budget larger than the pool gives the full pool.
  * minutes 0 = the Standard model unchanged (the starting point of every fit).

Input rescaling (before any training): the cells differ up to 6x in rheobase, so the Standard's input is
first rescaled by k = 83.8 pA / the lowest training square step that spikes (a rheobase any recording gives).
k is folded exactly into existing weights (first Conv1D kernel x k, g x k, I0 / k), so the architecture
is unchanged and the rule is the same for every fit.
Stage 1, front end (as the Standard recipe): closed loop on subthreshold segments (600 ms: 200 ms recorded
Vm seed, then free running; segments end 10 ms before the first spike of a trace). Trained: CNN-LSTM weights
and the Izhikevich g, I0, V0, Vs, a, b, time_scale. Loss: Vm MSE (clipped at -20 mV) in the free-running part.
250 Adam steps, batch 64, cosine decay to 10 % (lr 1e-3 for the network, 0.5 % of each value for the scalars).
Stage 2, spiking: front end frozen; Nelder-Mead (120 evaluations) over V_th, reset c, d, Na bias and the SK
scale on 1500 ms segments of the training traces (500 ms seed, then free running), loss = clipped Vm MSE
+ smoothed spike-train MSE + 0.1 x spike-count MSE, each normalised by its value at the start.
Evaluation: continuous prediction on the test traces (450 ms recorded seed, then free running), scored
against the noisy traces (what a recording gives) and the clean ones (the true cell).
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
ROOT = os.path.dirname(os.path.dirname(HERE))             # Isolated L2_3 circuit/
DATA = os.path.join(ROOT, "simulations", "additional_cells", "data")
PROTOS = ("square", "noise", "ramp")
N_TEST = 15
SEED_MS, CLIP = 450, -20.0
S1_LEN, S1_SEED, S1_BATCH, S1_ITERS = 600, 200, 64, 250
S2_LEN, S2_SEED, S2_N, S2_EVALS = 1500, 500, 96, 120
SIGMA, W_SPIKE, W_COUNT = 10.0, 1.0, 0.1
GAMMA_WIN = 4
STANDARD_RHEO_PA = 83.8      # rheobase of the Standard's cell (Allen model 488462783), 1 s step
S1_PARAMS = ("g", "I0", "V0", "Vs", "a", "b", "time_scale")
S2_PARAMS = ("V_th", "c_mV", "d", "b_na", "ln_s_sk")
NM_STEP = dict(V_th=1.5, c_mV=3.0, d=0.1, b_na=3.0, ln_s_sk=0.4)


# ------------------------------------------------------------------ data
def gt_spikes(V):
    S = np.zeros(V.shape, bool)
    S[:, 1:] = (V[:, 1:] > CLIP) & (V[:, :-1] <= CLIP)
    return S


def load_cell(cell):
    d = os.path.join(DATA, cell)
    out = {}
    for p in PROTOS:
        noisy, clean = np.load(os.path.join(d, f"{p}_noisy.npy")), np.load(os.path.join(d, f"{p}.npy"))
        out[p] = dict(I=noisy[..., 0].astype(np.float32), V=noisy[..., 1].astype(np.float32),
                      Vc=clean[..., 1].astype(np.float32))
    return out


def split(data, cell):
    """Fixed test traces and a fixed training order per protocol (same for every budget)."""
    rng = np.random.default_rng(int(cell))
    test, order = {}, {}
    for p in PROTOS:
        n = len(data[p]["I"])
        test[p] = np.unique(np.round(np.linspace(3, n - 4, N_TEST)).astype(int))
        rest = np.setdiff1d(np.arange(n), test[p])
        order[p] = rest[rng.permutation(len(rest))]
    return test, order


def budget(data, order, minutes):
    """Nested training subset: about minutes/3 of time from each protocol."""
    dur = {p: data[p]["I"].shape[1] / 1000.0 for p in PROTOS}                 # s per trace
    pool_s = {p: dur[p] * len(order[p]) for p in PROTOS}
    if minutes is None or minutes * 60 >= sum(pool_s.values()):
        return {p: order[p] for p in PROTOS}, sum(pool_s.values())
    total, share, left = minutes * 60.0, {}, list(PROTOS)
    remaining = total
    while left:                                     # water-filling: equal shares, capped at what a protocol has
        each = remaining / len(left)
        capped = [p for p in left if pool_s[p] <= each]
        if not capped:
            for p in left:
                share[p] = each
            break
        for p in capped:
            share[p] = pool_s[p]
            remaining -= pool_s[p]
            left.remove(p)
    sel = {}
    for p in PROTOS:
        k = int(np.clip(np.round(share[p] / dur[p]), 1, len(order[p])))
        sel[p] = order[p][:k]
    return sel, sum(dur[p] * len(sel[p]) for p in PROTOS)


# ------------------------------------------------------------------ model helpers
def ext_windows(I_ext):
    """I_ext (B, T+63) -> (B, T, 64) windows ending at each of the last T samples."""
    return tf.signal.frame(I_ext, 64, 1, axis=1)


def segment(I, V, i, start, L):
    """Current with 63 ms of real history (padded with I[0] at the trace start), Vm, for one segment."""
    pre = I[i, max(0, start - 63):start]
    pre = np.concatenate([np.full(63 - len(pre), I[i, 0], np.float32), pre])
    return np.concatenate([pre, I[i, start:start + L]]), V[i, start:start + L]


def masks(V):
    S = gt_spikes(V)
    M = S.copy()
    for k in range(1, 4):
        M[:, k:] |= S[:, :-k] & (V[:, k:] >= CLIP)
    return S.astype(np.float32), M.astype(np.float32)


def stage1_segments(data, sel, rng, n):
    """Random subthreshold segments (end >= 10 ms before the trace's first spike)."""
    cands = []
    for p in PROTOS:
        V = data[p]["V"]
        for i in sel[p]:
            s = np.flatnonzero(gt_spikes(V[i:i + 1])[0])
            end = (s[0] - 10) if len(s) else V.shape[1]
            if end >= S1_LEN:
                cands.append((p, i, end))
    if not cands:
        raise RuntimeError("no subthreshold segment long enough")
    Ie, Ve = [], []
    for j in rng.integers(len(cands), size=n):
        p, i, end = cands[j]
        st = int(rng.integers(0, end - S1_LEN + 1))
        a, b = segment(data[p]["I"], data[p]["V"], i, st, S1_LEN)
        Ie.append(a); Ve.append(b)
    return np.stack(Ie), np.stack(Ve)


def stage2_segments(data, sel, rng, n):
    """Segments for the spike fit: 2/3 with spikes in the free part, 1/3 any."""
    spk, anyc = [], []
    for p in PROTOS:
        V = data[p]["V"]
        T = V.shape[1]
        for i in sel[p]:
            s = np.flatnonzero(gt_spikes(V[i:i + 1])[0])
            anyc.append((p, i, None))
            if len(s):
                spk.append((p, i, s))
    Ie, Ve = [], []
    for k in range(n):
        use_spk = spk and (k % 3 != 2)
        p, i, s = spk[rng.integers(len(spk))] if use_spk else anyc[rng.integers(len(anyc))]
        T = data[p]["V"].shape[1]
        if s is not None:          # put a random spike somewhere in the free-running part
            t = int(s[rng.integers(len(s))])
            st = int(np.clip(t - S2_SEED - rng.integers(0, S2_LEN - S2_SEED), 0, T - S2_LEN))
        else:
            st = int(rng.integers(0, T - S2_LEN + 1))
        a, b = segment(data[p]["I"], data[p]["V"], i, st, S2_LEN)
        Ie.append(a); Ve.append(b)
    return np.stack(Ie), np.stack(Ve)


# ------------------------------------------------------------------ metrics
def gamma(t_true, t_model, T, win=GAMMA_WIN):
    n_d, n_m = len(t_true), len(t_model)
    if n_d == 0 and n_m == 0:
        return np.nan
    if n_d == 0 or n_m == 0:
        return 0.0
    used, coinc = np.zeros(n_m, bool), 0
    for t in t_true:
        k = np.flatnonzero(~used & (np.abs(t_model - t) <= win))
        if len(k):
            used[k[np.argmin(np.abs(t_model[k] - t))]] = True
            coinc += 1
    nu = n_m / T
    norm = 1 - 2 * nu * win
    return float((coinc - 2 * nu * win * n_d) / (0.5 * (n_d + n_m)) / norm) if norm > 0 else np.nan


def score(vm, spk, V):
    """Continuous-prediction metrics after the seed, against recorded Vm V."""
    V, vm, spk = V[:, SEED_MS:], vm[:, SEED_MS:], spk[:, SEED_MS:]
    S = gt_spikes(V)
    T = V.shape[1]
    gam = np.array([gamma(np.flatnonzero(a), np.flatnonzero(b), T) for a, b in zip(S, spk)])
    nt, nm = S.sum(1), spk.sum(1)
    free = nt == 0
    sub = np.sqrt(np.mean((vm[free] - V[free]) ** 2)) if free.any() else np.nan
    clip = np.sqrt(np.mean((np.minimum(vm, CLIP) - np.minimum(V, CLIP)) ** 2))
    return dict(gamma=float(np.nanmean(gam)) if np.isfinite(gam).any() else np.nan,
                gamma_per_trace=[None if np.isnan(g) else float(g) for g in gam],
                n_true=nt.tolist(), n_model=nm.tolist(),
                count_abs_err=float(np.mean(np.abs(nm - nt))), count_bias=float(np.mean(nm - nt)),
                count_r=float(np.corrcoef(nt, nm)[0, 1]) if nt.std() > 0 and nm.std() > 0 else np.nan,
                spike_free_rmse_mV=float(sub), n_spike_free=int(free.sum()),
                clipped_rmse_mV=float(clip))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--minutes", required=True, help="0, 5, 10, 20, 30 or full")
    ap.add_argument("--standard", default=os.path.join(ROOT, "Standard"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--iters", type=int, default=S1_ITERS)
    ap.add_argument("--evals", type=int, default=S2_EVALS)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    sys.path.insert(0, args.standard)
    import standard_model as sm

    tag = "full" if args.minutes == "full" else f"{int(args.minutes)}min"
    out = os.path.join(args.out, args.cell, tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{args.cell} {tag} {time.time()-t0:5.0f}s]", *a, flush=True)

    data = load_cell(args.cell)
    test, order = split(data, args.cell)
    minutes = None if args.minutes == "full" else float(args.minutes)
    sel, used_s = budget(data, order, minutes if minutes else 1e9)
    if minutes == 0:
        sel, used_s = {p: order[p][:0] for p in PROTOS}, 0.0
    log("training traces", {p: len(sel[p]) for p in PROTOS}, f"{used_s/60:.1f} min")

    keras_seed = 0
    tf.keras.utils.set_random_seed(keras_seed)
    rng = np.random.default_rng(0)
    cell = sm.load(args.standard)
    cell.p = {k: tf.Variable(v, name=k) for k, v in cell.p.items()}
    hist = []
    k_in = 1.0
    if minutes != 0:
        # input rescaling by the measured rheobase (the lowest training square step that spikes), folded
        # exactly into existing weights: Net(k I/100) = first Conv1D kernel x k; g(kI + I0) = (gk)(I + I0/k)
        sq = data["square"]
        amp = sq["I"][sel["square"], 1000]
        fires = gt_spikes(sq["V"][sel["square"]]).any(1)
        if fires.any():
            k_in = STANDARD_RHEO_PA / float(amp[fires].min())
            w = cell.net.layers[0].kernel
            w.assign(w * k_in)
            cell.p["g"].assign(cell.p["g"] * k_in)
            cell.p["I0"].assign(cell.p["I0"] / k_in)
        log(f"input rescaling k = {k_in:.3f} (rheobase {STANDARD_RHEO_PA:.1f} pA / {STANDARD_RHEO_PA / k_in:.1f} pA)")

    # ---------------- stage 1: front end + Izhikevich on subthreshold segments
    if minutes != 0:
        n_net = len(cell.net.trainable_variables)
        tv = cell.net.trainable_variables + [cell.p[k] for k in S1_PARAMS]
        opts = []
        for j, v in enumerate(tv):
            lr = 1e-3 if j < n_net else 0.005 * max(abs(float(v.numpy())), 1e-3)
            o = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(lr, args.iters, alpha=0.1))
            o.build([v])
            opts.append(o)

        @tf.function(reduce_retracing=True)
        def s1_step(Ie, V):
            S, M = tf.zeros_like(V), tf.zeros_like(V)
            with tf.GradientTape() as tape:
                B, T = tf.shape(V)[0], tf.shape(V)[1]
                y = cell.net(tf.reshape(ext_windows(Ie), (-1, 64, 1)) / sm.I_SCALE)
                x = cell.p["g"] * (Ie[:, 63:] + cell.p["I0"]) + tf.reshape(y, (B, T))
                vm, _ = cell.run(x, V, S, M, tf.constant(S1_SEED), False)
                e = tf.minimum(vm[:, S1_SEED:], CLIP) - tf.minimum(V[:, S1_SEED:], CLIP)
                loss = tf.reduce_mean(tf.square(e))
            grads = tape.gradient(loss, tv)
            for g, v, o in zip(grads, tv, opts):
                g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                o.apply([tf.clip_by_norm(g, 100.0)], [v])
            return loss

        for it in range(args.iters):
            Ie, V = stage1_segments(data, sel, rng, S1_BATCH)
            loss = float(s1_step(tf.constant(Ie), tf.constant(V)))
            hist.append(dict(stage=1, it=it, loss=loss))
            if it % 20 == 0 or it == args.iters - 1:
                log(f"stage 1 it {it:3d}  train Vm RMSE {np.sqrt(loss):.3f} mV")

    # ---------------- stage 2: spike parameters, front end frozen
    if minutes != 0:
        Ie, V2 = stage2_segments(data, sel, np.random.default_rng(1), S2_N)
        B, T = V2.shape
        y = np.concatenate([cell.net(tf.reshape(ext_windows(tf.constant(Ie[k:k + 16])), (-1, 64, 1)) / sm.I_SCALE).numpy()
                            .reshape(-1, T) for k in range(0, B, 16)])
        x2 = (float(cell.p["g"].numpy()) * (Ie[:, 63:] + float(cell.p["I0"].numpy())) + y).astype(np.float32)
        S2, M2 = masks(V2)
        k = np.arange(-int(4 * SIGMA), int(4 * SIGMA) + 1)
        K = tf.constant(np.exp(-0.5 * (k / SIGMA) ** 2)[:, None, None], tf.float32)
        sm_ = lambda s: tf.nn.conv1d(s[..., None], K, 1, "SAME")[..., 0]
        Xc, Vc, Sc, Mc = map(tf.constant, (x2, V2, S2, M2))

        @tf.function
        def terms():
            vm, s = cell.run(Xc, Vc, Sc, Mc, tf.constant(S2_SEED), False)
            vm, s, V_, S_ = vm[:, S2_SEED:], s[:, S2_SEED:], Vc[:, S2_SEED:], Sc[:, S2_SEED:]
            return tf.stack([tf.reduce_mean(tf.square(tf.minimum(vm, CLIP) - tf.minimum(V_, CLIP))),
                             tf.reduce_mean(tf.square(sm_(s) - sm_(S_))),
                             tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S_, 1)))])

        base = cell.params()
        norm = np.maximum(terms().numpy(), 1e-6)
        wl = np.array([1.0, W_SPIKE, W_COUNT]) / norm

        def obj(q):
            qd = dict(zip(S2_PARAMS, map(float, q)))
            if qd["c_mV"] > qd["V_th"] - 3.0:
                return 1e3 + qd["c_mV"] - qd["V_th"]
            cell.set({**base, **qd})
            tv_ = terms().numpy()
            loss = float(np.sum(wl * tv_)) if np.all(np.isfinite(tv_)) else 1e6
            hist.append(dict(stage=2, it=len(hist), loss=loss, **qd))
            return loss

        q0 = np.array([base[k] for k in S2_PARAMS])
        simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * NM_STEP[k] for i, k in enumerate(S2_PARAMS)])
        log("stage 2 start loss", round(obj(q0), 4))
        o = minimize(obj, q0, method="Nelder-Mead",
                     options=dict(maxfev=args.evals, xatol=1e-2, fatol=1e-4, initial_simplex=simplex))
        cell.set({**base, **dict(zip(S2_PARAMS, map(float, o.x)))})
        log("stage 2 loss", round(float(o.fun), 4), {k: round(float(v), 3) for k, v in zip(S2_PARAMS, o.x)})

    # ---------------- evaluation on the fixed test traces
    res, preds = {}, {}
    for p in PROTOS:
        idx = test[p]
        I, V, Vc = data[p]["I"][idx], data[p]["V"][idx], data[p]["Vc"][idx]
        vm, spk = cell.simulate(I, V_seed=V, seed_ms=SEED_MS)
        res[p] = dict(noisy=score(vm, spk, V), clean=score(vm, spk, Vc))
        preds[f"{p}_vm"], preds[f"{p}_idx"] = vm.astype(np.float32), idx
        log(p, "Γ %.3f  count bias %+.2f  spike-free RMSE %.2f mV (clean: Γ %.3f, %.2f mV)" % (
            res[p]["noisy"]["gamma"], res[p]["noisy"]["count_bias"], res[p]["noisy"]["spike_free_rmse_mV"],
            res[p]["clean"]["gamma"], res[p]["clean"]["spike_free_rmse_mV"]))

    izh = {k: float(v.numpy()) for k, v in cell.p.items()}
    np.savez_compressed(os.path.join(out, "test_predictions.npz"), **preds)
    w = cell.net.get_weights()
    np.savez_compressed(os.path.join(out, "front_end.npz"), **{f"w{i}": a for i, a in enumerate(w)})
    json.dump(dict(cell=args.cell, budget=args.minutes, train_minutes=used_s / 60.0,
                   train_traces={p: [int(i) for i in sel[p]] for p in PROTOS},
                   test_traces={p: [int(i) for i in test[p]] for p in PROTOS},
                   input_rescaling_k=k_in, izhikevich=izh, branches=cell.params(), eval=res, history=hist,
                   settings=dict(S1_LEN=S1_LEN, S1_SEED=S1_SEED, S1_BATCH=S1_BATCH, iters=args.iters,
                                 S2_LEN=S2_LEN, S2_SEED=S2_SEED, S2_N=S2_N, evals=args.evals, seed=keras_seed),
                   wall_s=time.time() - t0),
              open(os.path.join(out, "metrics.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
