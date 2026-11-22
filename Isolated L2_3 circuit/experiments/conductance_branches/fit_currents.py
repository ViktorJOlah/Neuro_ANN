"""Fit the branch scales and the Izhikevich d to the input the cell actually needed (least squares).

Run from "Isolated L2_3 circuit/" after spiking_branches.py --fast-k:
    python experiments/conductance_branches/fit_currents.py --model dense3 | cnnlstm | dense_cnnlstm

Along each NEURON training trace (spiking and spike-free, all protocols) the Izhikevich layer is
inverted: for every 1 ms step, the input that carries the recorded Vm(t-1) to Vm(t) through the model's
two half-steps (near_threshold.py). That input equals x(t) - u(t) + u(t) where u, the Izhikevich
recovery variable along the recorded trace, is linear in d (u = u_0 + d * U_1). The model supplies
  ANN(t) + g * (I + I0) + g * (sum_k s_k * i_k(t-1) + b) ,
with i_k the unit-scale branch currents (Na, SK, Kv3.1, BK) computed along the recorded voltage. So
  needed(t) - ANN(t) - g(I + I0) - u_0(t) = g * sum_k s_k i_k(t-1) + g * b - d * U_1(t)
is linear in (s_Na, s_SK, s_Kv, s_BK, b, d): solved by bounded weighted least squares (scales >= 0) on
all samples outside the spikes (2 ms before to 3 ms after onset excluded), with the -50 to -40 mV range
up-weighted. Why: a current-only ANN cannot see spike history, and after the earlier fits the leftover
error is a clean function of time since the last spike (too little outward current 3-25 ms after a
spike, too much 25-200 ms after), which is exactly what these terms shape.
Then V_th, c and the summed bias are refit by Nelder-Mead on the free-running spike loss (everything else fixed), and
the usual evaluation, spike-free check and block tests are run. Results: results_ls/<model>/.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import lsq_linear, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb  # noqa: E402
import channels as ch  # noqa: E402

sp, fs, base = sb.sp, sb.fs, sb.base
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spiking"))
import spiking_forced as sf  # noqa: E402

W_NEAR, NEAR, W_WIN = 5.0, (-50.0, -40.0), 64
BR = ("na", "sk", "kv", "bk")
SLOW_BR = ("kp", "im", "ih")


POST_W, POST_MS = 3.0, 10   # --post-spike: weight of the samples 1-10 ms after a spike onset


def components(cell, x, I, V, chunk=128, post=False):
    """Per-sample pieces of the linear problem for traces (n, T)."""
    P = {k: float(v.numpy()) for k, v in cell.p.items()}
    Q = cell.get()
    a, b, ts, Vs, V0 = (P[k] for k in ("a", "b", "time_scale", "Vs", "V0"))
    h = 0.5 * ts
    S, M = sp.gt_masks(V)
    T = V.shape[1]
    unit = {**Q, **{f"ln_s_{k}": 0.0 for k in BR + SLOW_BR}}
    cell.set(unit)
    f = tf.function(cell.run)
    cur = {k: [] for k in BR + SLOW_BR}
    for k in range(0, len(V), chunk):
        sl = slice(k, k + chunk)
        out = f(tf.constant(x[sl]), tf.constant(V[sl]), tf.constant(S[sl]), tf.constant(M[sl]), tf.constant(T),
                tf.constant(False), record=True)
        na, sk, _, kv, bk, kp, im, ih = (o.numpy() for o in out[2:])
        for key, arr in zip(BR + SLOW_BR, (na, sk, kv, bk, kp, im, ih)):
            cur[key].append(arr)
    cell.set(Q)
    cur = {k: np.concatenate(v) for k, v in cur.items()}
    cur = {k: np.concatenate([np.zeros_like(v[:, :1]), v[:, :-1]], 1) for k, v in cur.items()}   # used at t+1
    c = (Q["c_mV"] - V0) / Vs
    vs = (V - V0) / Vs
    v_prev = np.concatenate([vs[:, :1], vs[:, :-1]], 1)
    in_prev = np.concatenate([np.zeros_like(M[:, :1]), M[:, :-1]], 1)
    v_prev = np.where(in_prev > 0.5, c, v_prev)
    spk_prev = np.concatenate([np.zeros_like(S[:, :1]), S[:, :-1]], 1)
    u0, u1 = np.zeros_like(vs), np.zeros_like(vs)
    a0, a1 = b * vs[:, 0], np.zeros(len(vs))
    for t in range(T):
        a1 = a1 + spk_prev[:, t]
        u0[:, t], u1[:, t] = a0, a1
        a0 = a0 + ts * a * (b * vs[:, t] - a0)
        a1 = a1 - ts * a * a1
    F = lambda v_: 0.04 * v_ * v_ + 5.0 * v_ + 140.0
    z = (vs - v_prev) / (2 * h) - F(v_prev)                    # z = x - u, Newton on the two half-steps
    for _ in range(8):
        v1 = v_prev + h * (F(v_prev) + z)
        v2 = v1 + h * (F(v1) + z)
        z = z - (v2 - vs) / (h * (1 + h * (0.08 * v1 + 5.0)) + h)
    if post:
        # keep the samples right after a spike (where Kv3.1 / BK act); drop only the onset, the in-spike
        # samples and the 2 ms upstroke before an onset (the Izhikevich threshold makes that part)
        pre = np.zeros_like(S)
        pre[:, :-1] = np.maximum(pre[:, :-1], S[:, 1:]); pre[:, :-2] = np.maximum(pre[:, :-2], S[:, 2:])
        keep = (1 - S) * (1 - M) * (1 - pre)
    else:
        keep = sf.keep_mask(tf.constant(S)).numpy() * (1 - M) * (1 - in_prev)
    keep[:, :W_WIN] = 0
    w = keep * (1 + (W_NEAR - 1) * ((V >= NEAR[0]) & (V <= NEAR[1])))
    if post:
        since = np.full(S.shape, 10 ** 6)
        last = np.full(len(S), -10 ** 6)
        for t in range(S.shape[1]):
            last = np.where(S[:, t] > 0.5, t, last); since[:, t] = t - last
        w = w * np.where((since >= 1) & (since <= POST_MS), POST_W, 1.0)
    return dict(z=z, u0=u0, u1=u1, cur=cur, w=w, S=S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("dense3", "cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--base", default=os.path.join(HERE, "results_fast_k"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_ls"))
    ap.add_argument("--evals", type=int, default=150)
    ap.add_argument("--na-threshold", action="store_true", help="also fit th_k, the Na-availability threshold gain")
    ap.add_argument("--post-spike", action="store_true", help="fit from 1 ms after each spike, first 10 ms up-weighted")
    ap.add_argument("--slow", action="store_true", help="also fit the K_P, Im and Ih branches; results go to results_slow_k/")
    ap.add_argument("--slow-th", action="store_true", help="also fit th_d, th_tau: slow spike-triggered threshold")
    ap.add_argument("--isi-loss", type=float, default=0.0,
                    help="weight of a first-ISI term on strong (>=180 pA) training square steps, run continuously")
    ap.add_argument("--keep-base", action="store_true", help="keep the base fit's branch scales, bias and d (skip the LS result)")
    args = ap.parse_args()
    if args.slow and args.out == os.path.join(HERE, "results_ls"):
        args.out = os.path.join(HERE, "results_slow_k")
    br = BR + (SLOW_BR if args.slow else ())
    out = os.path.join(args.out, args.model)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sets = fs.load_split()
    m, _ = sp.load_frozen(args.model)
    r = json.load(open(os.path.join(args.base, args.model, "metrics.json")))
    cell = sb.BranchCell(r["frozen"], r["trained"])
    g, I0 = float(r["frozen"]["g"]), float(r["frozen"]["I0"])
    z_cache = np.load(os.path.join(sp.CACHE, f"front_end_{args.model}.npz"))

    def design(split, rng=None, per=40000):
        A, y, w = [], [], []
        for pr in sp.PROTOS:
            s = sets[(pr, split)]
            I, V = s["I"].astype(np.float32), s["V"].astype(np.float32)
            x = z_cache[f"{pr}_{split}"].astype(np.float32)
            c = components(cell, x, I, V, post=args.post_spike)
            lhs = c["z"] + c["u0"] - x            # x = ANN + g(I + I0), the frozen front end's output
            rr, cc = np.nonzero(c["w"])
            if rng is not None and len(rr) > per:   # equal share per protocol
                j = rng.choice(len(rr), per, replace=False); rr, cc = rr[j], cc[j]
            cols = [g * c["cur"][k][rr, cc] for k in br] + [np.full(len(rr), g), -c["u1"][rr, cc]]
            A.append(np.stack(cols, 1)); y.append(lhs[rr, cc]); w.append(c["w"][rr, cc])
            print(f"  {pr} {split}: {len(rr)} samples  {time.time()-t0:.0f}s", flush=True)
        return np.concatenate(A), np.concatenate(y), np.concatenate(w)

    A, y, w = design("train", np.random.default_rng(0), per=400000)
    sw = np.sqrt(w)
    lo = [0] * len(br) + [-np.inf, -20.0]
    hi = [np.inf] * len(br) + [np.inf, 40.0]
    sol = lsq_linear(A * sw[:, None], y * sw, bounds=(lo, hi))
    scales, b_all, d = dict(zip(br, sol.x[:len(br)])), sol.x[-2], sol.x[-1]
    old = {**sb.FAST_K_OFF, **sb.SLOW_OFF, **r["trained"]}
    old_vec = np.array([np.exp(old[f"ln_s_{k}"]) for k in br] + [sum(old[f"b_{k}"] for k in br), old["d"]])
    Av, yv, wv = design("val", np.random.default_rng(1), per=100000)
    rms = lambda vec: float(np.sqrt(np.sum(wv * (Av @ vec - yv) ** 2) / np.sum(wv)))
    print("LS solution:", {**{f"s_{k}": round(float(v), 3) for k, v in scales.items()}, "b": round(float(b_all), 2), "d": round(float(d), 3)}, flush=True)
    print("val weighted RMS (x units): base fit %.3f -> LS %.3f" % (rms(old_vec), rms(sol.x)), flush=True)
    lg = lambda v_: float(np.log(max(v_, 1e-12)))
    vals = {**old, **{f"ln_s_{k}": lg(v) for k, v in scales.items()}, **{f"b_{k}": 0.0 for k in br},
            "b_na": float(b_all), "d": float(d)}
    if args.keep_base:      # start the spike fit from the base fit's branch scales, bias and d (LS only reported)
        vals = dict(old)
    cell.set(vals)

    # V_th and c on the free-running spike loss of stage 1 (normalized like spiking_branches.py)
    xs = {(pr, s): z_cache[f"{pr}_{s}"] for pr in sp.PROTOS for s in ("train", "val", "test")}
    train, test = sp.pools(sets, xs, "train"), sp.pools(sets, xs, "test")
    K = sp.gauss_kernel()
    tseg = sb.draw_mixed(train, np.random.default_rng(0), 192)
    C = sp.SEG_CLAMP

    @tf.function
    def terms(X, V, S, M):
        vm, s = cell.run(X, V, S, M, C)
        vm, s, V, S = vm[:, C:], s[:, C:], V[:, C:], S[:, C:]
        return tf.stack([tf.reduce_mean(tf.square(tf.minimum(vm, sp.V_CLIP) - tf.minimum(V, sp.V_CLIP))),
                         tf.reduce_mean(tf.square(sp.smooth(s, K) - sp.smooth(S, K))),
                         tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))])

    def batched(seg, chunk=96):
        n = int(seg[0].shape[0])
        return np.sum([terms(*(a_[k:k + chunk] for a_ in seg)).numpy() * min(chunk, n - k) for k in range(0, n, chunk)], 0) / n

    norm = np.array(r["loss_norm_stage1"])
    wl = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm
    hist = []

    # first-ISI term: strong training square steps, ground-truth seed to SEED_MS, then free-running to ISI_END
    ISI_END, ISI_CAP = 800, 300.0
    if args.isi_loss > 0:
        sq = sets[("square", "train")]
        strong = np.flatnonzero(sq["I"][:, 1000] >= 180)
        Vq = sq["V"][strong, :ISI_END].astype(np.float32)
        xq = z_cache["square_train"][strong, :ISI_END].astype(np.float32)
        Sq, Mq = sp.gt_masks(Vq)

        def first_isi(spk):
            t_ = np.flatnonzero(spk[500:]); return float(t_[1] - t_[0]) if len(t_) > 1 else ISI_CAP
        isi_gt = np.array([first_isi(Sq[i] > 0.5) for i in range(len(Sq))])
        frun = tf.function(cell.run)
        print(f"first-ISI term: {len(strong)} strong train steps, NEURON median {np.median(isi_gt):.0f} ms", flush=True)

    def isi_term():
        _, sq_ = frun(tf.constant(xq), tf.constant(Vq), tf.constant(Sq), tf.constant(Mq), tf.constant(base.SEED_MS), tf.constant(False))
        sq_ = sq_.numpy() > 0.5
        isi_m = np.array([first_isi(sq_[i]) for i in range(len(sq_))])
        return float(np.mean(((isi_m - isi_gt) / isi_gt) ** 2)), float(np.median(isi_m)), int(np.sum(isi_m < 20))

    free = ["V_th", "c_mV", "b_na"] + (["th_k"] if args.na_threshold else []) + (["th_d", "th_tau"] if args.slow_th else [])
    start = {"th_k": 0.0, "th_d": 0.0, "th_tau": 30.0}
    nm_step = dict(V_th=1.5, c_mV=3.0, b_na=3.0, th_k=10.0, th_d=3.0, th_tau=20.0)

    def obj(q):
        qd = dict(zip(free, map(float, q)))
        cell.set({**vals, **qd})
        if q[1] > q[0] - 3.0:      # reset must sit below threshold, or the cell re-fires every step
            return 1e3 + float(q[1] - q[0])
        if qd.get("th_d", 0.0) < 0 or qd.get("th_tau", 30.0) < 1.0:
            return 1e3 + abs(qd.get("th_d", 0.0)) + abs(qd.get("th_tau", 30.0))
        tv = batched(tseg)
        loss = float(np.sum(wl * tv)) if np.all(np.isfinite(tv)) else 1e6
        isi = isi_term() if args.isi_loss > 0 else (0.0, np.nan, -1)
        loss += args.isi_loss * isi[0]
        hist.append(dict(loss=loss, **qd, vm=tv[0], vr=tv[1], count=tv[2], isi_err=isi[0], isi_med=isi[1], doublets=isi[2]))
        if len(hist) % 25 == 1:
            print(f"  NM {len(hist)}  loss {loss:.4f}  {qd}  isi {isi[1]:.0f} ms dbl {isi[2]}  {time.time()-t0:.0f}s", flush=True)
        return loss

    # V_th, c and the summed bias (it sets the closed-loop drift) on the free-running spike loss
    q0 = np.array([vals[k] if k in ("V_th", "c_mV", "b_na") else start[k] for k in free])
    simplex = np.vstack([q0] + [q0 + np.eye(len(q0))[i] * nm_step[k] for i, k in enumerate(free)])
    o = minimize(obj, q0, method="Nelder-Mead", options=dict(maxfev=args.evals, xatol=1e-2, fatol=1e-4, initial_simplex=simplex))
    vals.update(dict(zip(free, map(float, o.x))))
    cell.set(vals)
    print("fitted", {k: round(v, 3) for k, v in vals.items()}, f"loss {o.fun:.4f}", flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    preds, res = sp.evaluate(cell, test)
    sub = {}
    for pr in ("square", "noise"):
        keep = test[pr]["n"] == 0
        vm, _ = sp.simulate(cell, test[pr]["x"][keep], test[pr]["V"][keep], "cont")
        sub[pr] = fs.vm_rmse(vm, test[pr]["V"][keep], base.SEED_MS)
    print("spike-free continuous RMSE (mV):", sub, flush=True)
    xt = {pr: test[pr]["x"] for pr in sp.PROTOS}
    blk = sb.sk_block_test(cell, xt, sets)
    kvb = sb.sk_block_test(cell, xt, sets, cond_block="kv3_block")
    bkb = sb.bk_block_counts(cell, test)
    arrays = {f"{pr}_{k}_{q}": (v.astype(np.float32) if q == "vm" else v) for pr, d_ in preds.items()
              for k, dd in d_.items() for q, v in dd.items()}
    for pr in blk:
        arrays.update({f"skblock_{pr}_{k}": v for k, v in blk[pr].pop("_arrays").items()})
    for pr in kvb:
        kvb[pr].pop("_arrays")
    np.savez_compressed(os.path.join(out, "predictions.npz"), **arrays)
    res = {"model": args.model, "base": args.base, "trained": vals, "frozen": r["frozen"], "ls_solution": sol.x.tolist(),
           "ls_val_rms": {"base": rms(old_vec), "ls": rms(sol.x)}, "loss_norm_stage1": r["loss_norm_stage1"],
           "train_spike_loss": o.fun, "spike_free_cont_rmse_mV": sub, "sk_block": blk, "kv3_block": kvb,
           "bk_block": bkb, "fixed": r.get("fixed", {}), "eval": res}
    json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(json.dumps({pr: {k: {q: round(v, 3) for q, v in d_.items()} for k, d_ in rr.items()} for pr, rr in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
