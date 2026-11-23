"""Spike generation on top of the frozen subthreshold models (64 ms current window -> Izhikevich state).

Run from "Isolated L2_3 circuit/":
    python experiments/spiking/spiking.py --model izh | dense3 | cnnlstm | dense_cnnlstm

Everything fitted on subthreshold traces is frozen: the ANN, the current map (g, I0), the cell's
a, b and time scale, and the Vm map (Vs, V0). The frozen cell cannot spike on time. Its slowed
time scale (about 0.014) makes the escape from threshold to the published 30 mV peak take hundreds
of ms. The real cell instead fires about 3 ms after Vm reaches -40 mV, with little spread. So the
spike condition is a fitted threshold in mV, followed by the published reset. Trainable:
    V_th  spike threshold (mV)      a spike is emitted at the step where Vm >= V_th
    c     reset voltage (mV here; c in model units = (c_mV - V0) / Vs)
    d     jump of u at each spike (model units), i.e. spike-triggered adaptation
At the step after a spike, v <- c and u <- u + d, then the usual two half-steps for v and the
u update (the published 2003 scheme). The displayed Vm at a spike step is +30 mV.

Loss, on the free-running part of each training segment:
    L = Vm_term / Vm_term0 + W_SPIKE * vR_term / vR_term0 + W_COUNT * count_term / count_term0
  Vm_term     MSE of Vm with both traces clipped at -40 mV (spike peaks are not fitted)
  vR_term     van Rossum distance: MSE of the spike trains smoothed with a 10 ms Gaussian
  count_term  squared difference of spike counts per segment
Each term is normalized by its value at the initial parameters. With only three parameters the
loss is minimized directly with Nelder-Mead on a fixed set of training segments, using hard
spikes. A first attempt with surrogate gradients (fast sigmoid on Vm - V_th, kept in
SpikingCell.spike_fn) drifted: the surrogate pushed c and d to suppress near-threshold
spikes while the clipped Vm error grew, so the gradient-free fit is used. Segments are SEG_CLAMP + SEG_FREE ms long. In the
first SEG_CLAMP ms, Vm is clamped to ground truth and u evolves, with ground-truth spikes adding
d, as in the 450 ms seed of continuous prediction. The segments (384 for fitting, 192 for
validation) are drawn at random from one pool of spiking square, noise and ramp traces.

Evaluation on full test traces (all square/noise/ramp test traces, spiking or not):
  one-step:   every step starts from ground-truth Vm(t-1); inside a ground-truth spike (onset
              plus up to 3 ms above -40 mV) v starts from the reset c instead, and u gets d once
              per spike, so one ground-truth spike cannot trigger several model spikes;
  continuous: ground truth for 450 ms, then free running.
Metrics: clipped Vm RMSE, coincidence factor Gamma (+-4 ms), Allen explained variance of
spike trains smoothed with a 10 ms Gaussian, spike-count error and bias. Ground-truth spikes are
upward crossings of -20 mV. Model spikes are threshold crossings.
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
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "full_spiking"))
import common as fs  # noqa: E402  (load_split, gt_spikes, gamma, spike_metrics, vm_rmse)

base = fs.base
cw = fs.load_module("cw_run", os.path.join(EXP, "current_window", "run.py"))
SUB = os.path.join(EXP, "current_window", "results")
OUT = os.path.join(HERE, "results")
CACHE = os.environ.get("SPIKING_CACHE", "/tmp/spiking_cache")   # frozen front-end currents (not in git)
MODEL_NAMES = cw.MODEL_NAMES
PROTOS = ("square", "noise", "ramp")
V_CLIP, SPIKE_DISPLAY_MV, SURR_MV, SIGMA = -40.0, 30.0, 2.0, 10.0
SEG_CLAMP, SEG_FREE = 300, 1000
W_SPIKE, W_COUNT = 1.0, 0.1
INIT = dict(V_th=-40.0, c_mV=-46.0, d=8.0)


def load_frozen(name):
    p = json.load(open(os.path.join(SUB, name, "metrics.json")))["params"]
    m = cw.CurrentWindowIzhikevich(json.load(open(cw.seq.BASE_FIT))["params"], name)
    m(tf.zeros((1, 10)))
    m.load_weights(os.path.join(SUB, name, "model.weights.h5"))
    return m, p


def input_current(m, I, chunk=4):
    """The cell's input current from the frozen front end, (n, T) float32."""
    return np.concatenate([m.current(tf.constant(I[k:k + chunk])).numpy()[..., 0]
                           for k in range(0, len(I), chunk)]).astype(np.float32)


class SpikingCell(tf.Module):
    def __init__(self, p, init=INIT):
        super().__init__()
        self.p = {k: tf.constant(float(p[k]), tf.float32) for k in ("a", "b", "time_scale", "Vs", "V0")}
        self.V_th = tf.Variable(init["V_th"], dtype=tf.float32, name="V_th")
        self.c_mV = tf.Variable(init["c_mV"], dtype=tf.float32, name="c_mV")
        self.d = tf.Variable(init["d"], dtype=tf.float32, name="d")

    def params(self):
        return dict(V_th=float(self.V_th.numpy()), c_mV=float(self.c_mV.numpy()), d=float(self.d.numpy()))

    @staticmethod
    @tf.custom_gradient
    def spike_fn(z):
        s = tf.cast(z >= 0, z.dtype)

        def grad(dy):
            return dy / tf.square(1.0 + tf.abs(z))
        return s, grad

    def run(self, x, V_gt, gt_spk, gt_in, clamp_until, one_step=False):
        """x, V_gt: (n, T). gt_spk: ground-truth spike onsets; gt_in: samples inside a ground-truth
        spike (see gt_masks). Returns Vm (n, T) with spikes shown as +30 mV, and spikes (n, T)."""
        P = self.p
        a, b, ts, Vs, V0 = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"]
        c = (self.c_mV - V0) / Vs
        n, T = tf.shape(x)[0], tf.shape(x)[1]
        v = (V_gt[:, 0] - V0) / Vs
        u = b * v
        s = tf.zeros([n], tf.float32)
        vm_ta = tf.TensorArray(tf.float32, size=T)
        s_ta = tf.TensorArray(tf.float32, size=T)
        h = 0.5 * ts
        for t in tf.range(T):
            clamp = tf.logical_or(t < clamp_until, one_step)
            if clamp:
                # v from ground truth. Inside a ground-truth spike v is reset to c, and u gets d
                # once per spike (at its onset).
                prev = tf.maximum(t - 1, 0)
                first = tf.cast(t > 0, tf.float32)
                v = (V_gt[:, prev] - V0) / Vs
                v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
                u = u + first * gt_spk[:, prev] * self.d
            else:
                fired = tf.stop_gradient(s)
                v = fired * c + (1.0 - fired) * v
                u = u + fired * self.d
            for _ in range(2):
                v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + x[:, t])
            u = u + ts * a * (b * v - u)
            vm = Vs * v + V0
            s = self.spike_fn((vm - self.V_th) / SURR_MV)
            vm_ta = vm_ta.write(t, s * SPIKE_DISPLAY_MV + (1.0 - s) * vm)
            s_ta = s_ta.write(t, s)
        return tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack())


def gt_masks(V):
    """Spike onsets (upward -20 mV crossings) and the samples inside each spike: the onset and
    the next 3 ms while Vm stays above -40 mV."""
    S = fs.gt_spikes(V)
    M = S.copy()
    for k in range(1, 4):
        M[:, k:] |= S[:, :-k] & (V[:, k:] >= V_CLIP)
    return S.astype(np.float32), M.astype(np.float32)


def gauss_kernel():
    k = np.arange(-int(4 * SIGMA), int(4 * SIGMA) + 1)
    return tf.constant(np.exp(-0.5 * (k / SIGMA) ** 2)[:, None, None], tf.float32)


def smooth(s, K):
    return tf.nn.conv1d(s[..., None], K, 1, "SAME")[..., 0]


# ------------------------------------------------------------------ data
def pools(sets, xs, split):
    """All traces of a split, protocols kept separate (lengths differ)."""
    out = {}
    for pr in PROTOS:
        V = sets[(pr, split)]["V"]
        S, M = gt_masks(V)
        out[pr] = dict(x=xs[(pr, split)], V=V, S=S, M=M, n=sets[(pr, split)]["n_spikes"])
    return out


def draw_segments(pool, rng, batch):
    """Random segments from spiking traces of all protocols, mixed in one batch."""
    L = SEG_CLAMP + SEG_FREE
    keys = [(pr, i) for pr in PROTOS for i in np.flatnonzero(pool[pr]["n"] > 0)]
    pick = rng.choice(len(keys), batch, replace=False)
    X, V, S, M = [], [], [], []
    for j in pick:
        pr, i = keys[j]
        T = pool[pr]["V"].shape[1]
        t0 = rng.integers(0, T - L + 1)
        X.append(pool[pr]["x"][i, t0:t0 + L]); V.append(pool[pr]["V"][i, t0:t0 + L]); S.append(pool[pr]["S"][i, t0:t0 + L])
        M.append(pool[pr]["M"][i, t0:t0 + L])
    return [tf.constant(np.stack(a)) for a in (X, V, S, M)]


# ------------------------------------------------------------------ evaluation
def simulate(cell, x, V, kind, chunk=64):
    S, M = gt_masks(V)
    f = tf.function(cell.run)
    outs = [f(tf.constant(x[k:k + chunk]), tf.constant(V[k:k + chunk]), tf.constant(S[k:k + chunk]), tf.constant(M[k:k + chunk]),
              tf.constant(base.SEED_MS if kind == "cont" else 0), tf.constant(kind == "one"))
            for k in range(0, len(x), chunk)]
    return (np.concatenate([o[0].numpy() for o in outs]), np.concatenate([o[1].numpy() for o in outs]) > 0.5)


def evaluate(cell, test):
    preds, res = {}, {}
    for pr in PROTOS:
        d = test[pr]
        preds[pr], res[pr] = {}, {}
        ts = d["S"] > 0.5
        for kind, start in (("one", base.SKIP_MS), ("cont", base.SEED_MS)):
            vm, spk = simulate(cell, d["x"], d["V"], kind)
            preds[pr][kind] = dict(vm=vm, spikes=spk)
            res[pr][kind] = dict(vm_rmse_clipped_mV=fs.vm_rmse(np.minimum(vm, V_CLIP), np.minimum(d["V"], V_CLIP), start),
                                 **fs.spike_metrics(ts, spk, start))
    return preds, res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--evals", type=int, default=250)
    ap.add_argument("--n-train-seg", type=int, default=384)
    ap.add_argument("--n-val-seg", type=int, default=192)
    args = ap.parse_args()
    out = os.path.join(OUT, args.model)
    os.makedirs(out, exist_ok=True)
    tf.random.set_seed(0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    sets = fs.load_split()
    m, p = load_frozen(args.model)
    cache = os.path.join(CACHE, f"front_end_{args.model}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        xs = {(pr, sp): z[f"{pr}_{sp}"] for pr in PROTOS for sp in ("train", "val", "test")}
    else:
        xs = {(pr, sp): input_current(m, sets[(pr, sp)]["I"]) for pr in PROTOS for sp in ("train", "val", "test")}
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache, **{f"{pr}_{sp}": v for (pr, sp), v in xs.items()})
    print(f"front-end current ready after {time.time()-t0:.0f}s", flush=True)
    train, val, test = (pools(sets, xs, sp) for sp in ("train", "val", "test"))
    cell = SpikingCell(p)
    K = gauss_kernel()
    tseg = draw_segments(train, rng, args.n_train_seg)
    vseg = draw_segments(val, np.random.default_rng(1), args.n_val_seg)

    @tf.function
    def terms(X, V, S, M):
        vm, s = cell.run(X, V, S, M, SEG_CLAMP)
        vm, s, V, S = vm[:, SEG_CLAMP:], s[:, SEG_CLAMP:], V[:, SEG_CLAMP:], S[:, SEG_CLAMP:]
        t_vm = tf.reduce_mean(tf.square(tf.minimum(vm, V_CLIP) - tf.minimum(V, V_CLIP)))
        t_vr = tf.reduce_mean(tf.square(smooth(s, K) - smooth(S, K)))
        t_n = tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))
        return tf.stack([t_vm, t_vr, t_n])

    def set_params(q):
        cell.V_th.assign(q[0]); cell.c_mV.assign(q[1]); cell.d.assign(q[2])

    def batched_terms(seg, chunk=128):
        n = int(seg[0].shape[0])
        tv = [terms(*(a[k:k + chunk] for a in seg)).numpy() * min(chunk, n - k) for k in range(0, n, chunk)]
        return np.sum(tv, 0) / n

    q0 = np.array([INIT["V_th"], INIT["c_mV"], INIT["d"]])
    set_params(q0)
    norm = np.maximum(batched_terms(tseg), 1e-6)
    w = np.array([1.0, W_SPIKE, W_COUNT]) / norm
    print("initial terms (train segments): Vm %.2f mV^2, vR %.4f, count %.2f" % tuple(norm), flush=True)
    hist = []

    def objective(q):
        set_params(q)
        tv = batched_terms(tseg)
        loss = float(np.sum(w * tv))
        hist.append(dict(eval=len(hist), loss=loss, vm_term=tv[0], vr_term=tv[1], count_term=tv[2],
                         V_th=q[0], c_mV=q[1], d=q[2]))
        if len(hist) % 10 == 1:
            print(f"eval {len(hist):4d}  train loss {loss:.4f} (Vm {tv[0]:.2f} mV^2, vR {tv[1]:.4f}, count {tv[2]:.2f})  "
                  f"V_th {q[0]:.2f} c {q[1]:.2f} mV d {q[2]:.3f}  {time.time()-t0:.0f}s", flush=True)
        return loss

    from scipy.optimize import minimize
    simplex = np.array([q0, q0 + [2, 0, 0], q0 + [0, 4, 0], q0 + [0, 0, 4]])
    opt = minimize(objective, q0, method="Nelder-Mead",
                   options=dict(maxfev=args.evals, xatol=0.02, fatol=1e-4, initial_simplex=simplex))
    best_p = dict(V_th=float(opt.x[0]), c_mV=float(opt.x[1]), d=float(opt.x[2]))
    set_params(q0)
    val0 = batched_terms(vseg)
    set_params(opt.x)
    val1 = batched_terms(vseg)
    best = float(np.sum(w * val1))
    print(f"val loss: init {np.sum(w * val0):.4f} -> fitted {best:.4f}  ({opt.nfev} evaluations, {opt.message})", flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    preds, res = evaluate(cell, test)
    np.savez_compressed(os.path.join(out, "predictions.npz"),
                        **{f"{pr}_{k}_{q}": (v.astype(np.float32) if q == "vm" else v) for pr, d in preds.items()
                           for k, dd in d.items() for q, v in dd.items()})
    res = {"model": args.model, "trained": best_p, "init": INIT, "frozen": p, "val_loss": best,
           "val_terms_init": val0.tolist(), "val_terms_fitted": val1.tolist(), "loss_norm": norm.tolist(),
           "loss_weights": dict(vm=1.0, van_rossum=W_SPIKE, count=W_COUNT), "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q: round(v, 3) for q, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
