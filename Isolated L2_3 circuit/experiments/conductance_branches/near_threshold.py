"""Teach the frozen ANN front end the near-threshold range (-50 to -40 mV) it never saw in training.

Run from "Isolated L2_3 circuit/" after spiking_branches.py --fast-k:
    python experiments/conductance_branches/near_threshold.py --model dense3 | cnnlstm | dense_cnnlstm
then branch_finetune.py --ann-weights <out>/<model>/ann.weights.h5 --iters 0 refits threshold and branches
and evaluates.

Why: the subthreshold ANN was trained on spike-free traces whose Vm stayed below about -46 mV, so the
run-up to threshold between spikes (-46 to -40 mV) is extrapolation, and its errors shift every spike.
Spike-forced training through the 1 ms loop (branch_finetune.py) barely moves the weights: once the
model runs ahead it sits at threshold, where there is no gradient.

Here the ANN is trained directly on the input it should have produced. Along each recorded trace
(spiking and spike-free, all protocols) the Izhikevich layer and the branches are run clamped to the
NEURON voltage. For every 1 ms step the Izhikevich input x*(t) that carries Vm(t-1) exactly to Vm(t)
is solved for (Newton, the same two half-steps as the model). Subtracting the branch current and the
linear part g*(I + I0) leaves the target for the ANN's output. Samples within 2 ms before to 3 ms after
a spike onset are left out (the reset and the spike pulse model that part). Samples between -50 and
-40 mV are up-weighted (W_NEAR). Loss: weighted MSE + an L2 pull to the subthreshold weights.
This is one-step (teacher-forced) supervision, so the free-running spike-free RMSE is checked after
training; the subthreshold fit must survive.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import tensorflow as tf
import keras

if os.environ.get("TF_THREADS"):
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_THREADS"]))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb  # noqa: E402
import channels as ch  # noqa: E402

sp, fs, base = sb.sp, sb.fs, sb.base
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spiking"))
import spiking_forced as sf  # noqa: E402
cw = sp.cw   # current_window/run.py, loaded by spiking.py

W_NEAR, NEAR = 5.0, (-50.0, -40.0)
W, I_SCALE = cw.W, cw.I_SCALE


def clamp_targets(cell, x_lin, I, V, chunk=128):
    """Per-sample ANN target, weight, for traces (n, T) of one protocol."""
    P, Q = {k: float(v.numpy()) for k, v in cell.p.items()}, cell.get()
    a, b, ts, Vs, V0, g = (P[k] for k in ("a", "b", "time_scale", "Vs", "V0", "g"))
    h = 0.5 * ts
    S, M = sp.gt_masks(V)
    T = V.shape[1]
    f = tf.function(cell.run)
    ib = []
    for k in range(0, len(V), chunk):
        sl = slice(k, k + chunk)
        out = f(tf.constant(x_lin[sl]), tf.constant(V[sl]), tf.constant(S[sl]), tf.constant(M[sl]), tf.constant(T),
                tf.constant(False), record=True)
        na, sk, _, kv, bk = (o.numpy() for o in out[2:])
        ib.append(na + sk + kv + bk + Q["b_na"] + Q["b_sk"] + Q["b_kv"] + Q["b_bk"])
    ib = np.concatenate(ib)                                   # branch current computed at step t, used at t+1
    ib_used = np.concatenate([np.zeros_like(ib[:, :1]), ib[:, :-1]], 1)
    c = (Q["c_mV"] - V0) / Vs
    vs = (V - V0) / Vs
    v_prev = np.concatenate([vs[:, :1], vs[:, :-1]], 1)
    in_prev = np.concatenate([np.zeros_like(M[:, :1]), M[:, :-1]], 1)
    v_prev = np.where(in_prev > 0.5, c, v_prev)
    spk_prev = np.concatenate([np.zeros_like(S[:, :1]), S[:, :-1]], 1)
    u = np.zeros_like(vs)
    uu = b * vs[:, 0]
    for t in range(T):                                        # u along the recorded trajectory
        uu = uu + Q["d"] * spk_prev[:, t]
        u[:, t] = uu
        uu = uu + ts * a * (b * vs[:, t] - uu)
    F = lambda v_: 0.04 * v_ * v_ + 5.0 * v_ + 140.0
    xs = (vs - v_prev) / (2 * h) - F(v_prev) + u               # Newton for the two half-steps
    for _ in range(8):
        v1 = v_prev + h * (F(v_prev) - u + xs)
        v2 = v1 + h * (F(v1) - u + xs)
        dv2 = h * (1 + h * (0.08 * v1 + 5.0)) + h
        xs = xs - (v2 - vs) / dv2
    target = xs - g * ib_used - g * (I + cell.I0)
    keep = sf.keep_mask(tf.constant(S)).numpy() * (1 - M) * (1 - in_prev)
    keep[:, :W] = 0
    wgt = keep * (1 + (W_NEAR - 1) * ((V >= NEAR[0]) & (V <= NEAR[1])))
    return target.astype(np.float32), wgt.astype(np.float32)


def windows_at(I, rows, cols):
    idx = cols[:, None] + np.arange(-W + 1, 1)[None, :]
    return (I[rows[:, None], np.maximum(idx, 0)] / I_SCALE)[..., None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("dense3", "cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--base", default=os.path.join(HERE, "results_fast_k"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_near_threshold"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anchor", type=float, default=1.0)
    args = ap.parse_args()
    out = os.path.join(args.out, args.model)
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(0)
    t0 = time.time()
    sets = fs.load_split()
    m, _ = sp.load_frozen(args.model)
    r = json.load(open(os.path.join(args.base, args.model, "metrics.json")))
    cell = sb.BranchCell(r["frozen"], r["trained"])
    cell.I0 = float(r["frozen"]["I0"])
    g = float(r["frozen"]["g"])

    def net_out(I):
        return m.current_np(I) - g * (I + cell.I0)

    data = {}
    for split in ("train", "val"):
        for pr in sp.PROTOS:
            s = sets[(pr, split)]
            I, V = s["I"].astype(np.float32), s["V"].astype(np.float32)
            x = m.current_np(I).astype(np.float32)
            tgt, wgt = clamp_targets(cell, x, I, V)
            data[(pr, split)] = dict(I=I, V=V, tgt=tgt, w=wgt, y0=(x - g * (I + cell.I0)).astype(np.float32))
            near = wgt > 1.5
            e0 = (data[(pr, split)]["y0"] - tgt)
            print(f"{pr:6s} {split:5s}: {int((wgt > 0).sum())} samples, {int(near.sum())} near threshold; "
                  f"current ANN error RMS {np.sqrt((e0[wgt > 0] ** 2).mean()):.3f} all, {np.sqrt((e0[near] ** 2).mean()):.3f} near (x units)  "
                  f"{time.time()-t0:.0f}s", flush=True)

    # sample pool: (protocol, row, col) with probability proportional to weight
    def pool(split):
        P_, R_, C_, Wt = [], [], [], []
        for j, pr in enumerate(sp.PROTOS):
            rr, cc = np.nonzero(data[(pr, split)]["w"])
            P_.append(np.full(len(rr), j)); R_.append(rr); C_.append(cc); Wt.append(data[(pr, split)]["w"][rr, cc])
        P_, R_, C_, Wt = map(np.concatenate, (P_, R_, C_, Wt))
        return P_, R_, C_, Wt / Wt.sum()

    tr, va = pool("train"), pool("val")

    def batch(pl, rng, n):
        j = rng.choice(len(pl[3]), n, p=pl[3])
        Xs, Ys = [], []
        for k, pr in enumerate(sp.PROTOS):
            sel = j[pl[0][j] == k]
            if not len(sel):
                continue
            d = data[(pr, "train" if pl is tr else "val")]
            Xs.append(windows_at(d["I"], pl[1][sel], pl[2][sel])); Ys.append(d["tgt"][pl[1][sel], pl[2][sel]])
        return tf.constant(np.concatenate(Xs)), tf.constant(np.concatenate(Ys))

    net = m.net
    net.trainable = True   # load_frozen freezes the front end
    w0 = [tf.constant(v.numpy()) for v in net.trainable_variables]
    n_w = float(sum(int(np.prod(v.shape)) for v in w0))
    opt = keras.optimizers.Adam(args.lr)

    @tf.function
    def step(X, Y):
        with tf.GradientTape() as tape:
            e = net(X)[:, 0] - Y
            loss = tf.reduce_mean(e * e)
        gr = tape.gradient(loss, net.trainable_variables)
        # L2 pull to the subthreshold weights, added analytically (putting it on the tape with Keras 3
        # variables disconnects the whole gradient)
        gr = [g_ + 2.0 * args.anchor / n_w * (tf.convert_to_tensor(v) - v0) for g_, v, v0 in zip(gr, net.trainable_variables, w0)]
        opt.apply(gr, net.trainable_variables)
        return loss

    @tf.function
    def vloss(X, Y):
        e = net(X)[:, 0] - Y
        return tf.reduce_mean(e * e)

    vb = [batch(va, np.random.default_rng(5 + k), args.batch) for k in range(4)]
    rng = np.random.default_rng(0)
    best, best_w, hist = np.inf, None, []
    for it in range(args.steps + 1):
        if it % 250 == 0:
            vl = float(np.mean([vloss(*b_).numpy() for b_ in vb]))
            if vl < best:
                best, best_w = vl, [v.numpy() for v in net.trainable_variables]
            hist.append(dict(step=it, val_mse=vl))
            print(f"step {it:5d}  val weighted MSE {vl:.4f} (x units^2)  {time.time()-t0:.0f}s", flush=True)
        if it == args.steps:
            break
        step(*batch(tr, rng, args.batch))
    for v, w_ in zip(net.trainable_variables, best_w):
        v.assign(w_)
    m.save_weights(os.path.join(out, "ann.weights.h5"))

    # checks: target error near threshold, and the free-running spike-free fit with the base parameters
    chk = {}
    for pr in sp.PROTOS:
        d = data[(pr, "val")]
        y = net_out(d["I"])
        near, allk = d["w"] > 1.5, d["w"] > 0
        chk[pr] = dict(err_all_before=float(np.sqrt(((d["y0"] - d["tgt"])[allk] ** 2).mean())),
                       err_all_after=float(np.sqrt(((y - d["tgt"])[allk] ** 2).mean())),
                       err_near_before=float(np.sqrt(((d["y0"] - d["tgt"])[near] ** 2).mean())),
                       err_near_after=float(np.sqrt(((y - d["tgt"])[near] ** 2).mean())))
    sub = {}
    for pr in ("square", "noise"):
        s = sets[(pr, "test")]
        keep = s["n_spikes"] == 0
        x = m.current_np(s["I"][keep]).astype(np.float32)
        vm, _ = sp.simulate(cell, x, s["V"][keep].astype(np.float32), "cont")
        sub[pr] = fs.vm_rmse(vm, s["V"][keep], base.SEED_MS)
    res = dict(model=args.model, base=args.base, steps=args.steps, lr=args.lr, anchor=args.anchor, W_NEAR=W_NEAR,
               val_mse_best=best, target_error_val=chk, spike_free_cont_rmse_mV_base_params=sub, history=hist)
    json.dump(res, open(os.path.join(out, "train.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("target_error_val", "spike_free_cont_rmse_mV_base_params")}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
