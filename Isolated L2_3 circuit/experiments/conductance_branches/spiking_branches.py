"""Izhikevich top layer + parallel conductance branches (Na and Ca -> SK), all at 1 ms.

Run from "Isolated L2_3 circuit/" after experiments/spiking/spiking.py (front-end cache):
    python experiments/conductance_branches/spiking_branches.py --model dense3 | cnnlstm | dense_cnnlstm

Architecture (Viktor, 2026-11-22). The frozen 64 ms current-window ANN gives the Izhikevich
cell's input x(t), exactly as in the subthreshold fits. Two branches run in parallel with it.
Each one reads the Izhikevich Vm, integrates the Allen model's somatic channel kinetics from the
mod files (channels.py) over 1 ms, and returns a current that is added to x(t):
  Na branch:      NaTs + Nap                                 -> I_Na
  Ca -> SK branch: Ca_HVA + Ca_LVA -> CaDynamics -> SK        -> I_SK
  x(t) <- x(t) + g * (s_Na * I_Na + b_Na + s_SK * I_SK + b_SK)        (pA; g = frozen current gain)
s_* are trainable scales (fitted in log space, so they can move by orders of magnitude) and
b_* trainable biases, because the branch currents of a single soma need not match the scale of
the ANN's effective current. A third scale, s_Ca, multiplies the calcium influx.
The Izhikevich layer still decides spikes (Vm >= V_th, then v <- c, u += d). In the step of a
spike the branches see a 1 ms pulse at V_SPIKE (spike peak, sim convention) instead of the
Izhikevich value, so NaTs inactivates and Ca_HVA opens during the spike as in the real cell.
Subthreshold-only conductances (Im, K_P, K_T, Ih, Kv3_1) are left out; the ANN already
learned them. For the same reason the Na branch subtracts the steady-state Na current below
V_REF = -46 mV, the top of the spike-free training range (see V_REF).
Frozen: ANN, g, I0, a, b, time scale, Vs, V0. Fitted (8): V_th, c, d, log s_Na, b_Na,
log s_SK, b_SK, log s_Ca. Same loss, segments (but 1000 ms clamp so Ca has history, and a quarter of the
segments spike-free) and evaluation as stage 1 (experiments/spiking/spiking.py). Nelder-Mead from 3 starts.
Afterwards, the SK edge is zeroed (s_SK = 0) and the model is run on the test traces that
NEURON was also run on with gbar_SK = 0 (neuron_reference.py) - the in-silico apamin test.
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.environ.get("SB_HERE", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spiking"))
import channels as ch  # noqa: E402
import spiking as sp  # noqa: E402

fs, base = sp.fs, sp.base
OUT = os.path.join(HERE, "results")
NREF = os.environ.get("NEURON_REF", "/tmp/nref_kv" if os.path.isdir("/tmp/nref_kv") else "/tmp/nref")
sp.SEG_CLAMP = 1000          # Ca decays with tau ~900 ms: give the branch a second of history
V_SPIKE = sp.SPIKE_DISPLAY_MV - ch.JP     # +16 mV sim
# The subthreshold ANN was trained on spike-free traces whose Vm stayed below about -46 mV
# (99.9th percentile of spike-free training Vm: -47 mV square, -45 mV noise), so it has already
# learned the steady-state Na current up to there. The Na branch therefore returns only what the
# ANN could not have learned: I_Na(V, h) - I_Na,steady(min(V, V_REF)). That is the regenerative
# current above V_REF plus the non-steady-state part (e.g. NaTs inactivated after a spike).
V_REF = -46.0
NAMES = ("V_th", "c_mV", "d", "ln_s_na", "b_na", "ln_s_sk", "b_sk", "ln_s_ca", "ln_s_kv", "b_kv", "ln_s_bk", "b_bk",
         "ln_s_kp", "b_kp", "ln_s_im", "b_im", "ln_s_ih", "b_ih")
STEP = dict(V_th=1.5, c_mV=3.0, d=3.0, ln_s_na=0.7, b_na=10.0, ln_s_sk=0.7, b_sk=10.0, ln_s_ca=0.7,
            ln_s_kv=0.7, b_kv=10.0, ln_s_bk=0.7, b_bk=10.0,
            ln_s_kp=0.7, b_kp=10.0, ln_s_im=0.7, b_im=10.0, ln_s_ih=0.7, b_ih=10.0)
STARTS = [dict(ln_s_na=0.0, b_na=0.0, ln_s_sk=0.0, b_sk=0.0, ln_s_ca=0.0, ln_s_kv=0.0, b_kv=0.0, ln_s_bk=0.0, b_bk=0.0),
          dict(ln_s_na=-1.5, b_na=0.0, ln_s_sk=1.0, b_sk=0.0, ln_s_ca=1.0, ln_s_kv=-1.0, b_kv=0.0, ln_s_bk=-2.0, b_bk=0.0),
          dict(ln_s_na=-3.0, b_na=0.0, ln_s_sk=2.0, b_sk=0.0, ln_s_ca=0.0, ln_s_kv=0.5, b_kv=0.0, ln_s_bk=1.0, b_bk=0.0)]
# fast-AHP branches (Kv3.1 from the Allen mod file, generic BK) are off unless --fast-k: scale exp(-30) ~ 0
FAST_K_OFF = dict(ln_s_kv=-30.0, b_kv=0.0, ln_s_bk=-30.0, b_bk=0.0)
# slow stateful K_P, Im, Ih branches: off here; they are fitted by least squares in fit_currents.py --slow
SLOW_OFF = dict(ln_s_kp=-30.0, b_kp=0.0, ln_s_im=-30.0, b_im=0.0, ln_s_ih=-30.0, b_ih=0.0)


class BranchCell(tf.Module):
    def __init__(self, p, vals):
        super().__init__()
        self.p = {k: tf.constant(float(p[k]), tf.float32) for k in ("a", "b", "time_scale", "Vs", "V0", "g")}
        vals = {**FAST_K_OFF, **SLOW_OFF, **vals}   # fits from before these branches existed
        self.q = {k: tf.Variable(float(vals[k]), dtype=tf.float32, name=k) for k in NAMES}
        self.sk_on = tf.Variable(1.0, dtype=tf.float32, name="sk_on")    # 0 = SK edge removed
        self.na_on = tf.Variable(1.0, dtype=tf.float32, name="na_on")
        self.kv_on = tf.Variable(1.0, dtype=tf.float32, name="kv_on")    # 0 = Kv3.1 edge removed (TEA-like)
        self.bk_on = tf.Variable(1.0, dtype=tf.float32, name="bk_on")    # 0 = BK edge removed (paxilline-like)
        self.kp_on = tf.Variable(1.0, dtype=tf.float32, name="kp_on")    # K_P
        self.im_on = tf.Variable(1.0, dtype=tf.float32, name="im_on")    # Im (XE991/linopirdine-like)
        self.ih_on = tf.Variable(1.0, dtype=tf.float32, name="ih_on")    # Ih (ZD7288-like)

    def get(self):
        return {k: float(v.numpy()) for k, v in self.q.items()}

    def set(self, vals):
        for k, v in vals.items():
            self.q[k].assign(v)

    def run(self, x, V_gt, gt_spk, gt_in, clamp_until, one_step=False, record=False):
        P, Q = self.p, self.q
        a, b, ts, Vs, V0, g = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"], P["g"]
        c = (Q["c_mV"] - V0) / Vs
        s_na, s_sk, s_ca = tf.exp(Q["ln_s_na"]) * self.na_on, tf.exp(Q["ln_s_sk"]) * self.sk_on, tf.exp(Q["ln_s_ca"])
        s_kv, s_bk = tf.exp(Q["ln_s_kv"]) * self.kv_on, tf.exp(Q["ln_s_bk"]) * self.bk_on
        s_kp, s_im, s_ih = tf.exp(Q["ln_s_kp"]) * self.kp_on, tf.exp(Q["ln_s_im"]) * self.im_on, tf.exp(Q["ln_s_ih"]) * self.ih_on
        n, T = tf.shape(x)[0], tf.shape(x)[1]
        v = (V_gt[:, 0] - V0) / Vs
        u = b * v
        st = ch.init_state(V_gt[:, 0] - ch.JP)
        s = tf.zeros([n], tf.float32)
        ib = tf.zeros([n], tf.float32)
        vm_ta, s_ta = tf.TensorArray(tf.float32, size=T), tf.TensorArray(tf.float32, size=T)
        na_ta, sk_ta, ca_ta, kv_ta, bk_ta, kp_ta, im_ta, ih_ta = (tf.TensorArray(tf.float32, size=T) for _ in range(8))
        h = 0.5 * ts
        for t in tf.range(T):
            clamp = tf.logical_or(t < clamp_until, one_step)
            if clamp:
                prev = tf.maximum(t - 1, 0)
                first = tf.cast(t > 0, tf.float32)
                v = (V_gt[:, prev] - V0) / Vs
                v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
                u = u + first * gt_spk[:, prev] * Q["d"]
            else:
                v = s * c + (1.0 - s) * v
                u = u + s * Q["d"]
            xt = x[:, t] + g * ib
            for _ in range(2):
                v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
            u = u + ts * a * (b * v - u)
            vm = Vs * v + V0
            s = tf.cast(vm >= Q["V_th"], tf.float32)
            if clamp:      # branches follow the recorded voltage (spike onset -> spike pulse, rest of spike -> reset)
                vb = gt_spk[:, t] * V_SPIKE + (1 - gt_spk[:, t]) * (
                    gt_in[:, t] * (Q["c_mV"] - ch.JP) + (1 - gt_in[:, t]) * (V_gt[:, t] - ch.JP))
                spike_now = gt_spk[:, t]
            else:
                vb = s * V_SPIKE + (1 - s) * (tf.minimum(vm, Q["V_th"]) - ch.JP)
                spike_now = s
            st, cur = ch.step(st, vb, s_ca)
            i_na = -(cur["NaTs"] + cur["Nap"] - ch.na_steady(tf.minimum(vb, V_REF - ch.JP))) * ch.PA_PER_DENSITY  # pA, depolarizing +
            i_sk = -cur["SK"] * ch.PA_PER_DENSITY
            # fast K: the current during the spike step itself repolarizes the spike, which the Izhikevich
            # reset already does, so only the after-current (from the step after the spike) is passed on.
            # Kv3.1's small steady current below V_REF is subtracted, as for Na (the ANN learned it).
            i_kv = -(cur["Kv3_1"] - ch.kv31_steady(tf.minimum(vb, V_REF - ch.JP))) * ch.PA_PER_DENSITY * (1 - spike_now)
            i_bk = -cur["BK"] * ch.PA_PER_DENSITY * (1 - spike_now)
            # slow, stateful K_P / Im / Ih (gates carried across steps = memory of past spikes); their steady
            # state below V_REF is subtracted, as for Na and Kv3.1, because the ANN learned it
            ss = ch.slow_steady(tf.minimum(vb, V_REF - ch.JP))
            i_kp = -(cur["K_P"] - ss["K_P"]) * ch.PA_PER_DENSITY * (1 - spike_now)
            i_im = -(cur["Im"] - ss["Im"]) * ch.PA_PER_DENSITY * (1 - spike_now)
            i_ih = -(cur["Ih"] - ss["Ih"]) * ch.PA_PER_DENSITY * (1 - spike_now)
            ib = (s_na * i_na + Q["b_na"] + s_sk * i_sk + Q["b_sk"]
                  + s_kv * i_kv + Q["b_kv"] + s_bk * i_bk + Q["b_bk"]
                  + s_kp * i_kp + Q["b_kp"] + s_im * i_im + Q["b_im"] + s_ih * i_ih + Q["b_ih"])
            vm_ta = vm_ta.write(t, s * sp.SPIKE_DISPLAY_MV + (1.0 - s) * vm)
            s_ta = s_ta.write(t, s)
            if record:
                na_ta = na_ta.write(t, s_na * i_na); sk_ta = sk_ta.write(t, s_sk * i_sk); ca_ta = ca_ta.write(t, st["cai"])
                kv_ta = kv_ta.write(t, s_kv * i_kv); bk_ta = bk_ta.write(t, s_bk * i_bk)
                kp_ta = kp_ta.write(t, s_kp * i_kp); im_ta = im_ta.write(t, s_im * i_im); ih_ta = ih_ta.write(t, s_ih * i_ih)
        out = (tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack()))
        if record:
            out = out + tuple(tf.transpose(a_.stack()) for a_ in (na_ta, sk_ta, ca_ta, kv_ta, bk_ta, kp_ta, im_ta, ih_ta))
        return out


def draw_mixed(pool, rng, n, frac_free=0.25):
    """Spiking segments (as in stage 1) plus a quarter spike-free segments, so the fit also sees
    that the branches must stay quiet on traces that never spike."""
    n_free = int(round(frac_free * n))
    spk = sp.draw_segments(pool, rng, n - n_free)
    L = sp.SEG_CLAMP + sp.SEG_FREE
    keys = [(pr, i) for pr in ("square", "noise") for i in np.flatnonzero(pool[pr]["n"] == 0)]
    pick = rng.choice(len(keys), n_free, replace=False)
    cols = [[], [], [], []]
    for j in pick:
        pr, i = keys[j]
        t0 = rng.integers(0, pool[pr]["V"].shape[1] - L + 1)
        for c_, k in zip(cols, ("x", "V", "S", "M")):
            c_.append(pool[pr][k][i, t0:t0 + L])
    return [tf.concat([a_, tf.constant(np.stack(c_))], 0) for a_, c_ in zip(spk, cols)]


def to_vec(v):
    return np.array([v[k] for k in NAMES])


def from_vec(q):
    return {k: float(x) for k, x in zip(NAMES, q)}


BLOCKS = dict(sk_block="sk_on", kv3_block="kv_on")   # NEURON condition -> model edge


def sk_block_test(cell, x_test, sets, chunk=64, cond_block="sk_block"):
    """Model vs NEURON with one conductance removed (SK by default; kv3_block = Kv3.1), on the NEURON
    reference test traces. The model's edge is zeroed without retraining."""
    out = {}
    edge = getattr(cell, BLOCKS[cond_block])
    f = tf.function(cell.run)
    for pr in sp.PROTOS:
        path = os.path.join(NREF, f"test_{pr}.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        if f"{cond_block}_v" not in z.files:
            return None
        rows = z["rows"]
        X = x_test[pr][rows]
        res = {}
        for cond, on in (("control", 1.0), (cond_block, 0.0)):
            edge.assign(on)
            Vn = z[f"{cond}_v"]
            S, M = sp.gt_masks(Vn)
            spk = np.concatenate([f(tf.constant(X[k:k + chunk]), tf.constant(Vn[k:k + chunk]), tf.constant(S[k:k + chunk]),
                                    tf.constant(M[k:k + chunk]), tf.constant(base.SEED_MS), tf.constant(False))[1].numpy()
                                  for k in range(0, len(X), chunk)]) > 0.5
            res[cond] = dict(model_spikes=spk, neuron_spikes=S > 0.5)
        edge.assign(1.0)
        st = base.SEED_MS
        cnt = {k: {c: res[c][k][:, st:].sum(1) for c in res} for k in ("model_spikes", "neuron_spikes")}
        dn, dm = cnt["neuron_spikes"][cond_block] - cnt["neuron_spikes"]["control"], cnt["model_spikes"][cond_block] - cnt["model_spikes"]["control"]
        out[pr] = dict(
            n=int(len(rows)),
            neuron_count_control=float(cnt["neuron_spikes"]["control"].mean()), neuron_count_block=float(cnt["neuron_spikes"][cond_block].mean()),
            model_count_control=float(cnt["model_spikes"]["control"].mean()), model_count_block=float(cnt["model_spikes"][cond_block].mean()),
            delta_count_corr=float(np.corrcoef(dn, dm)[0, 1]) if dn.std() > 0 and dm.std() > 0 else float("nan"),
            gamma_block=fs.spike_metrics(res[cond_block]["neuron_spikes"], res[cond_block]["model_spikes"], st)["gamma"],
            gamma_control=fs.spike_metrics(res["control"]["neuron_spikes"], res["control"]["model_spikes"], st)["gamma"])
        out[pr]["_arrays"] = {f"{c}_{k}": res[c][k] for c in res for k in res[c]}
    return out


def bk_block_counts(cell, test, chunk=64):
    """Spikes per test trace with and without the BK edge (no NEURON counterpart: Allen has no BK)."""
    f = tf.function(cell.run)
    out = {}
    for pr in sp.PROTOS:
        X, V, S, M = (test[pr][k] for k in ("x", "V", "S", "M"))
        cnt = {}
        for lab, on in (("control", 1.0), ("bk_block", 0.0)):
            cell.bk_on.assign(on)
            spk = np.concatenate([f(*(tf.constant(a_[k:k + chunk]) for a_ in (X, V, S, M)), tf.constant(base.SEED_MS),
                                    tf.constant(False))[1].numpy() for k in range(0, len(X), chunk)])
            cnt[lab] = float(spk[:, base.SEED_MS:].sum(1).mean())
        cell.bk_on.assign(1.0)
        out[pr] = dict(model_count_control=cnt["control"], model_count_bk_block=cnt["bk_block"],
                       gt_count=float(S[:, base.SEED_MS:].sum(1).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("dense3", "cnnlstm", "dense_cnnlstm"), required=True)
    ap.add_argument("--evals", type=int, default=500)
    ap.add_argument("--n-train-seg", type=int, default=256)
    ap.add_argument("--n-val-seg", type=int, default=128)
    ap.add_argument("--pin-sk", action="store_true",
                    help="fix the SK scale at Allen's strength (s_SK = 1); results go to results_sk_pinned/")
    ap.add_argument("--fast-k", action="store_true",
                    help="add the Kv3.1 and BK branches (SK pinned); results go to results_fast_k/")
    ap.add_argument("--pin-kv", type=float, default=None,
                    help="with --fast-k: fix the Kv3.1 scale (e.g. 0.67, its NEURON-matched strength) and start from the SK-pinned fit")
    ap.add_argument("--warm", action="store_true",
                    help="with --fast-k: start from the SK-pinned fit, fast-K scales free; results go to results_fast_k_warm/")
    args = ap.parse_args()
    global OUT
    fixed = dict(ln_s_sk=0.0) if (args.pin_sk or args.fast_k) else {}
    if not args.fast_k:
        fixed.update(FAST_K_OFF)
    fixed.update(SLOW_OFF)
    if args.pin_sk:
        OUT = os.path.join(HERE, "results_sk_pinned")
    if args.fast_k:
        OUT = os.path.join(HERE, "results_fast_k")
    if args.fast_k and args.pin_kv is not None:
        fixed.update(ln_s_kv=float(np.log(args.pin_kv)))
        OUT = os.path.join(HERE, "results_fast_k_kvpinned")
    if args.fast_k and args.warm:
        OUT = os.path.join(HERE, "results_fast_k_warm")
    warm = args.fast_k and (args.pin_kv is not None or args.warm)
    fit = [k for k in NAMES if k not in fixed]
    out = os.path.join(OUT, args.model)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    sets = fs.load_split()
    p = json.load(open(os.path.join(sp.SUB, args.model, "metrics.json")))["params"]
    z = np.load(os.path.join(sp.CACHE, f"front_end_{args.model}.npz"))
    xs = {(pr, s): z[f"{pr}_{s}"] for pr in sp.PROTOS for s in ("train", "val", "test")}
    train, val, test = (sp.pools(sets, xs, s) for s in ("train", "val", "test"))
    st1 = json.load(open(os.path.join(sp.OUT, args.model, "metrics.json")))["trained"]
    if warm:   # start every simplex from the SK-pinned solution
        pin = json.load(open(os.path.join(HERE, "results_sk_pinned", args.model, "metrics.json")))["trained"]
    else:
        pin = {}
    cell = BranchCell(p, dict(st1, **STARTS[0]))
    K = sp.gauss_kernel()
    tseg = draw_mixed(train, np.random.default_rng(0), args.n_train_seg)
    vseg = draw_mixed(val, np.random.default_rng(1), args.n_val_seg)
    C = sp.SEG_CLAMP

    @tf.function
    def terms(X, V, S, M):
        vm, s = cell.run(X, V, S, M, C)
        vm, s, V, S = vm[:, C:], s[:, C:], V[:, C:], S[:, C:]
        t_vm = tf.reduce_mean(tf.square(tf.minimum(vm, sp.V_CLIP) - tf.minimum(V, sp.V_CLIP)))
        t_vr = tf.reduce_mean(tf.square(sp.smooth(s, K) - sp.smooth(S, K)))
        t_n = tf.reduce_mean(tf.square(tf.reduce_sum(s, 1) - tf.reduce_sum(S, 1)))
        return tf.stack([t_vm, t_vr, t_n])

    def batched(seg, chunk=128):
        n = int(seg[0].shape[0])
        return np.sum([terms(*(a_[k:k + chunk] for a_ in seg)).numpy() * min(chunk, n - k) for k in range(0, n, chunk)], 0) / n

    # normalization: stage-1 model (branches off) on these segments, so stages are comparable
    edges = (cell.na_on, cell.sk_on, cell.kv_on, cell.bk_on, cell.kp_on, cell.im_on, cell.ih_on)
    for e_ in edges:
        e_.assign(0.0)
    cell.set(dict(st1, b_na=0.0, b_sk=0.0, b_kv=0.0, b_bk=0.0, b_kp=0.0, b_im=0.0, b_ih=0.0))
    norm = np.maximum(batched(tseg), 1e-6)
    for e_ in edges:
        e_.assign(1.0)
    w = np.array([1.0, sp.W_SPIKE, sp.W_COUNT]) / norm
    print("stage-1 terms (train segments): Vm %.2f mV^2, vR %.4f, count %.2f" % tuple(norm), flush=True)
    hist, best = [], [np.inf, None]

    def objective(q):
        vals = {**dict(zip(fit, map(float, q))), **fixed}
        cell.set(vals)
        tv = batched(tseg)
        loss = float(np.sum(w * tv)) if np.all(np.isfinite(tv)) else 1e6
        hist.append(dict(eval=len(hist), loss=loss, vm_term=tv[0], vr_term=tv[1], count_term=tv[2], **vals))
        if loss < best[0]:
            best[0], best[1] = loss, vals
        if len(hist) % 50 == 1:
            print(f"eval {len(hist):5d}  loss {loss:.4f}  best {best[0]:.4f} (Vm {tv[0]:.2f} mV^2, vR {tv[1]:.4f}, "
                  f"count {tv[2]:.2f})  {time.time()-t0:.0f}s", flush=True)
        return loss

    from scipy.optimize import minimize
    starts = STARTS
    if warm:   # fast-K starts off, small or at 1 (Kv3.1 follows BK unless pinned)
        starts = [dict(ln_s_bk=lb, b_bk=0.0, b_kv=0.0, ln_s_kv=lb) for lb in (-4.0, -1.5, 0.0)]
    for j, st in enumerate(starts):
        v0 = {**st1, **pin, **st, **fixed}   # stage-1 normalization above still uses st1 alone
        q0 = np.array([v0[k] for k in fit])
        simplex = np.vstack([q0] + [q0 + np.eye(len(fit))[i] * STEP[k] for i, k in enumerate(fit)])
        opt = minimize(objective, q0, method="Nelder-Mead",
                       options=dict(maxfev=args.evals, xatol=1e-3, fatol=1e-4, initial_simplex=simplex, adaptive=True))
        print(f"start {j}: loss {opt.fun:.4f} after {opt.nfev} evaluations; best so far {best[0]:.4f}", flush=True)
    fitted = best[1]
    cell.set(fitted)
    val1 = batched(vseg)
    print("fitted", {k: round(v, 3) for k, v in fitted.items()},
          "scales", {k: round(float(np.exp(fitted[k])), 4) for k in NAMES if k.startswith("ln_")}, flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    preds, res = sp.evaluate(cell, test)
    # spike-free subthreshold check (continuous, clipped not needed: no spikes)
    sub = {}
    for pr in ("square", "noise"):
        keep = test[pr]["n"] == 0
        vm, _ = sp.simulate(cell, test[pr]["x"][keep], test[pr]["V"][keep], "cont")
        sub[pr] = fs.vm_rmse(vm, test[pr]["V"][keep], base.SEED_MS)
    print("spike-free continuous RMSE (mV):", sub, flush=True)
    blk = sk_block_test(cell, {pr: test[pr]["x"] for pr in sp.PROTOS}, sets)
    kvb = sk_block_test(cell, {pr: test[pr]["x"] for pr in sp.PROTOS}, sets, cond_block="kv3_block") if args.fast_k else None
    bkb = bk_block_counts(cell, test) if args.fast_k else None
    arrays = {f"{pr}_{k}_{q}": (v.astype(np.float32) if q == "vm" else v) for pr, d in preds.items()
              for k, dd in d.items() for q, v in dd.items()}
    if blk:
        for pr in blk:
            arrays.update({f"skblock_{pr}_{k}": v for k, v in blk[pr].pop("_arrays").items()})
        print("SK-block test:", json.dumps(blk, indent=1), flush=True)
    if kvb:
        for pr in kvb:
            arrays.update({f"kvblock_{pr}_{k}": v for k, v in kvb[pr].pop("_arrays").items()})
        print("Kv3.1-block test:", json.dumps(kvb, indent=1), flush=True)
    if bkb:
        print("BK-block (model only):", json.dumps(bkb, indent=1), flush=True)
    np.savez_compressed(os.path.join(out, "predictions.npz"), **arrays)
    res = {"model": args.model, "trained": fitted, "scales": {k: float(np.exp(fitted[k])) for k in NAMES if k.startswith("ln_")},
           "frozen": p, "val_terms_fitted": val1.tolist(), "loss_norm_stage1": norm.tolist(),
           "spike_free_cont_rmse_mV": sub, "sk_block": blk, "kv3_block": kvb, "bk_block": bkb,
           "fixed": fixed, "eval": res}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({pr: {k: {q: round(v, 3) for q, v in d.items()} for k, d in r.items()} for pr, r in res["eval"].items()}, indent=1))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
