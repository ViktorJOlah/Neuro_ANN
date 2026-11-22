"""Anchored ANN fine-tune with the conductance branches in place and SK pinned at Allen's strength.

Run from "Isolated L2_3 circuit/" after spiking_branches.py --pin-sk:
    python experiments/conductance_branches/branch_finetune.py --model dense3 | cnnlstm | dense_cnnlstm

Why: with the ANN frozen, the fit could slow the climb back to threshold between spikes only by
inflating SK (1.7-2.7x), which made the first AHP far too deep. The climb is too fast because the
subthreshold training data never went above about -46 mV, so the ANN's current in the -46 to
-40 mV range is extrapolation. Here the ANN learns that range directly, with SK held at s_SK = 1.

Step 1 (Adam, ANN weights only; Izhikevich and branch parameters from the SK-pinned fit, frozen):
  spikes are forced at the ground-truth times (Izhikevich reset, u jump and the branches' spike
  pulse at each onset), Vm runs free in between with no threshold.
  loss = Vm MSE below -40 mV outside spikes (2 ms before to 3 ms after each onset excluded)
       + W_SUB x free-running Vm MSE on spike-free segments
       + L2_ANN x mean squared change of the ANN weights from their subthreshold values.
Step 2 (Nelder-Mead, ANN fixed): V_th, c, d, Na scale and bias, SK bias and Ca scale refit on the
  free-running spike loss of stage 1, SK scale still pinned at 1.
Then the usual evaluation, spike-free check and the SK-block test.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

if os.environ.get("TF_THREADS"):   # several fits side by side on a few cores
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_THREADS"]))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb  # noqa: E402  (sets sp.SEG_CLAMP = 1000)
import channels as ch  # noqa: E402

sp, fs, base = sb.sp, sb.fs, sb.base
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spiking"))
import spiking_all as sall  # noqa: E402
import spiking_forced as sf  # noqa: E402

PIN = os.path.join(HERE, "results_sk_pinned")
OUT = os.path.join(HERE, "results_finetune")
W_SUB, SUB_BATCH, L2_ANN = 10.0, 8, 100.0
V_LOSS = -40.0  # forced-mode Vm compared below this, so an early run-up costs a bounded error with a useful gradient
SOFT_MV = 8.0   # --soft: above V_LOSS Vm is squashed with tanh (scale 8 mV) instead of cut, so an early run-up
                # still has a gradient pushing it back down


def squash(v, soft):
    if not soft:
        return tf.minimum(v, V_LOSS)
    return tf.where(v < V_LOSS, v, V_LOSS + SOFT_MV * tf.tanh((v - V_LOSS) / SOFT_MV))


class ForcedBranchCell(sb.BranchCell):
    def run_forced(self, x, V_gt, gt_spk, gt_in, clamp_until):
        """Spikes at the ground-truth onsets, Vm free in between (no threshold; held at V_th if it would have fired)."""
        P = self.p
        Q = {k: tf.stop_gradient(tf.convert_to_tensor(v)) for k, v in self.q.items()}  # fixed in step 1
        a, b, ts, Vs, V0, g = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"], P["g"]
        c = (Q["c_mV"] - V0) / Vs
        s_na, s_sk, s_ca = tf.exp(Q["ln_s_na"]) * self.na_on, tf.exp(Q["ln_s_sk"]) * self.sk_on, tf.exp(Q["ln_s_ca"])
        s_kv, s_bk = tf.exp(Q["ln_s_kv"]) * self.kv_on, tf.exp(Q["ln_s_bk"]) * self.bk_on
        n, T = tf.shape(x)[0], tf.shape(x)[1]
        v = (V_gt[:, 0] - V0) / Vs
        u = b * v
        st = ch.init_state(V_gt[:, 0] - ch.JP)
        ib = tf.zeros([n], tf.float32)
        vm_ta = tf.TensorArray(tf.float32, size=T)
        # x is read through a TensorArray: slicing the captured tensor makes the loop gradient accumulate
        # into it with a TemporaryVariable, which crashes (AlreadyExistsError) on the CNN-LSTM models
        x_ta = tf.TensorArray(tf.float32, size=T).unstack(tf.transpose(x))
        h = 0.5 * ts
        vcap = (Q["V_th"] - V0) / Vs  # where the model would have fired: hold there instead of running away
        for t in tf.range(T):
            prev = tf.maximum(t - 1, 0)
            first = tf.cast(t > 0, tf.float32)
            if t < clamp_until:
                v = (V_gt[:, prev] - V0) / Vs
            v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
            u = u + first * gt_spk[:, prev] * Q["d"]
            xt = x_ta.read(t) + g * ib
            for _ in range(2):
                v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
            u = u + ts * a * (b * v - u)
            v = tf.minimum(v, vcap)
            vm = Vs * v + V0
            if t < clamp_until:
                vfree = V_gt[:, t]
            else:
                vfree = tf.minimum(vm, Q["V_th"])
            vb = gt_spk[:, t] * sb.V_SPIKE + (1 - gt_spk[:, t]) * (
                gt_in[:, t] * (Q["c_mV"] - ch.JP) + (1 - gt_in[:, t]) * (vfree - ch.JP))
            st, cur = ch.step(st, vb, s_ca)
            i_na = -(cur["NaTs"] + cur["Nap"] - ch.na_steady(tf.minimum(vb, sb.V_REF - ch.JP))) * ch.PA_PER_DENSITY
            i_sk = -cur["SK"] * ch.PA_PER_DENSITY
            i_kv = -(cur["Kv3_1"] - ch.kv31_steady(tf.minimum(vb, sb.V_REF - ch.JP))) * ch.PA_PER_DENSITY * (1 - gt_spk[:, t])
            i_bk = -cur["BK"] * ch.PA_PER_DENSITY * (1 - gt_spk[:, t])
            ib = (s_na * i_na + Q["b_na"] + s_sk * i_sk + Q["b_sk"]
                  + s_kv * i_kv + Q["b_kv"] + s_bk * i_bk + Q["b_bk"])
            vm_ta = vm_ta.write(t, vm)
        return tf.transpose(vm_ta.stack())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("dense3", "cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr-ann", type=float, default=3e-4)
    ap.add_argument("--evals2", type=int, default=400)
    ap.add_argument("--base", default=PIN, help="fit to start from (Izhikevich + branch parameters)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--anchor", type=float, default=L2_ANN, help="weight of the pull back to the subthreshold ANN weights")
    ap.add_argument("--soft", action="store_true", help="tanh-squash Vm above -40 mV in the forced loss instead of cutting it")
    args = ap.parse_args()
    out = os.path.join(args.out, args.model)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    # Grappler rewrites the loop-gradient AddN into a TemporaryVariable accumulator whose name clashes
    # across iterations on the CNN-LSTM graphs (AlreadyExistsError); run the graphs unoptimized
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    t0 = time.time()
    sets = fs.load_split()
    m, p0 = sp.load_frozen(args.model)
    pin = json.load(open(os.path.join(args.base, args.model, "metrics.json")))
    cell = ForcedBranchCell(pin["frozen"], pin["trained"])
    C = sp.SEG_CLAMP

    def x_of(I):
        return m.current(I)[..., 0]

    def forced_mse(I, V, S, M):
        vm = cell.run_forced(x_of(I), V, S, M, C)
        keep = sf.keep_mask(S)[:, C:]
        e = (squash(vm[:, C:], args.soft) - squash(V[:, C:], args.soft)) * keep
        return tf.reduce_sum(e * e) / tf.reduce_sum(keep)

    def sub_mse(I, V, S, M):
        vm = cell.run_forced(x_of(I), V, S, M, C)
        return tf.reduce_mean(tf.square(vm[:, C:] - V[:, C:]))

    ann_vars = list(m.net.trainable_variables)
    ann0 = [tf.constant(v.numpy()) for v in ann_vars]
    n_ann = float(sum(int(np.prod(v.shape)) for v in ann0))

    def anchor():
        return args.anchor * tf.add_n([tf.reduce_sum(tf.square(v - v0)) for v, v0 in zip(ann_vars, ann0)]) / n_ann

    zero = {(pr, s): np.zeros_like(sets[(pr, s)]["I"]) for pr in sp.PROTOS for s in ("train", "val")}

    def free_segments(split, rng_, n):
        L = C + sp.SEG_FREE
        keys = [(pr, i) for pr in sp.PROTOS for i in np.flatnonzero(sets[(pr, split)]["n_spikes"] == 0)
                if sets[(pr, split)]["V"].shape[1] >= L]
        I, V = [], []
        for j in rng_.choice(len(keys), n, replace=False):
            pr, i = keys[j]
            t_ = rng_.integers(0, sets[(pr, split)]["V"].shape[1] - L + 1)
            I.append(sets[(pr, split)]["I"][i, t_:t_ + L]); V.append(sets[(pr, split)]["V"][i, t_:t_ + L])
        I, V = np.stack(I).astype(np.float32), np.stack(V).astype(np.float32)
        Z = np.zeros_like(V)
        return [tf.constant(a_) for a_ in (I, V, Z, Z)]

    def seg4(seg):
        I, _, V, S, M = seg
        return I, V, S, M

    opt = keras.optimizers.Adam(args.lr_ann)
    opt.build(ann_vars)

    # the two simulation losses are differentiated in separate graphs: two while-loop gradients in one
    # graph hit a TensorFlow AlreadyExistsError (TemporaryVariable name clash) on the CNN-LSTM models
    @tf.function
    def grad_forced(I, V, S, M):
        with tf.GradientTape() as tape:
            loss = forced_mse(I, V, S, M) + anchor()
        return tape.gradient(loss, ann_vars)

    @tf.function
    def grad_sub(I, V, S, M):
        with tf.GradientTape() as tape:
            loss = W_SUB * sub_mse(I, V, S, M)
        return tape.gradient(loss, ann_vars)

    def step(I, V, S, M, Is, Vs_, Ss, Ms):
        gr = [a_ + b_ for a_, b_ in zip(grad_forced(I, V, S, M), grad_sub(Is, Vs_, Ss, Ms))]
        gr = [tf.where(tf.math.is_finite(g_), g_, tf.zeros_like(g_)) for g_ in gr]
        gr, _ = tf.clip_by_global_norm(gr, 10.0)
        opt.apply(gr, ann_vars)

    f_mse, s_mse = tf.function(forced_mse), tf.function(sub_mse)
    vseg = seg4(sall.segments(sets, zero, "val", np.random.default_rng(1), 64))
    vsub = free_segments("val", np.random.default_rng(3), 32)

    def val():
        fm = float(np.mean([f_mse(*(a_[k:k + 32] for a_ in vseg)).numpy() for k in (0, 32)]))
        sm = float(np.mean([s_mse(*(a_[k:k + 16] for a_ in vsub)).numpy() for k in (0, 16)]))
        return fm, sm

    rng = np.random.default_rng(0)
    hist, best, best_w = [], np.inf, None
    for it in range(args.iters + 1):
        if it % 25 == 0:
            fm, sm = val()
            vl = fm + W_SUB * sm
            if np.isfinite(vl) and vl < best:
                best, best_w = vl, [v.numpy() for v in ann_vars]
            hist.append(dict(step=1, iter=it, val_forced_mse=fm, val_sub_mse=sm))
            print(f"it {it:4d}  val forced Vm MSE {fm:8.3f}  spike-free Vm MSE {sm:6.3f} mV^2  {time.time()-t0:.0f}s", flush=True)
        if it == args.iters:
            break
        step(*seg4(sall.segments(sets, zero, "train", rng, args.batch)), *free_segments("train", rng, SUB_BATCH))
    for v, w_ in zip(ann_vars, best_w):
        v.assign(w_)
    m.save_weights(os.path.join(out, "ann.weights.h5"))

    # ---- step 2: spiking parameters on the free-running loss, ANN fixed
    def xs_of(split):
        return {(pr, split): np.concatenate([x_of(tf.constant(sets[(pr, split)]["I"][k:k + 4])).numpy()
                                             for k in range(0, len(sets[(pr, split)]["I"]), 4)]) for pr in sp.PROTOS}
    xs = {**xs_of("train"), **xs_of("test")}
    train, test = sp.pools(sets, xs, "train"), sp.pools(sets, xs, "test")
    K = sp.gauss_kernel()
    tseg = sb.draw_mixed(train, np.random.default_rng(0), 192)

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

    norm = np.array(pin["loss_norm_stage1"])
    w = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm
    fixed = {**pin.get("fixed", {}), "ln_s_sk": 0.0}   # keep whatever the base fit held fixed
    fit = [k for k in sb.NAMES if k not in fixed]
    v0 = cell.get()

    def obj(q):
        vals = {**v0, **dict(zip(fit, map(float, q)))}
        cell.set(vals)
        tv = batched(tseg)
        loss = float(np.sum(w * tv)) if np.all(np.isfinite(tv)) else 1e6
        hist.append(dict(step=2, loss=loss, vm_term=tv[0], vr_term=tv[1], count_term=tv[2], **vals))
        if len(hist) % 50 == 0:
            print(f"step 2 eval {len(hist)}  loss {loss:.4f}  {time.time()-t0:.0f}s", flush=True)
        return loss

    from scipy.optimize import minimize
    q0 = np.array([v0[k] for k in fit])
    simplex = np.vstack([q0] + [q0 + np.eye(len(fit))[i] * sb.STEP[k] for i, k in enumerate(fit)])
    o = minimize(obj, q0, method="Nelder-Mead", options=dict(maxfev=args.evals2, xatol=1e-3, fatol=1e-4,
                                                             initial_simplex=simplex, adaptive=True))
    vals = {**v0, **dict(zip(fit, map(float, o.x)))}
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
    blk = sb.sk_block_test(cell, {pr: test[pr]["x"] for pr in sp.PROTOS}, sets)
    kvb = sb.sk_block_test(cell, {pr: test[pr]["x"] for pr in sp.PROTOS}, sets, cond_block="kv3_block")
    bkb = sb.bk_block_counts(cell, test)
    if kvb:
        for pr in kvb:
            kvb[pr].pop("_arrays")
    arrays = {f"{pr}_{k}_{q}": (v.astype(np.float32) if q == "vm" else v) for pr, d in preds.items()
              for k, dd in d.items() for q, v in dd.items()}
    for pr in blk:
        arrays.update({f"skblock_{pr}_{k}": v for k, v in blk[pr].pop("_arrays").items()})
    np.savez_compressed(os.path.join(out, "predictions.npz"), **arrays)
    res = {"model": args.model, "trained": vals, "frozen": pin["frozen"], "val_loss_step1": best, "train_spike_loss": o.fun,
           "spike_free_cont_rmse_mV": sub, "sk_block": blk, "kv3_block": kvb, "bk_block": bkb, "eval": res,
           "weights": dict(W_SUB=W_SUB, L2_ANN=args.anchor, soft=args.soft), "base": args.base, "fixed": fixed}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q: round(v, 3) for q, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
