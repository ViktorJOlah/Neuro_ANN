"""Quick check (before building branches): how much of the spike-history-dependent error left after the
fast-K fit do the slow, stateful K_P, Im and Ih currents explain?

Same least-squares problem as fit_currents.py (inverted Izhikevich input along NEURON traces), with the
unit-scale K_P, Im and Ih currents added as regressors. They are computed along the recorded voltage in the
model's convention (spike onset = 1 ms pulse, rest of spike = reset level), steady-state part below V_REF
subtracted (the ANN learned it), passed on from the step after. Reports the validation RMS with and without
them, the fitted scales, and the residual binned by time since the last spike.
Run from "Isolated L2_3 circuit/":  python experiments/conductance_branches/check_slow_k.py --model dense3
"""
import argparse, json, os, sys, time
import numpy as np
import tensorflow as tf
from scipy.optimize import lsq_linear
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fit_currents as fc
sb, sp, fs, ch = fc.sb, fc.sp, fc.fs, fc.ch
SLOW = ("K_P", "Im", "Ih")
BINS = ((3, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 10 ** 9))


@tf.function
def drive(V, S, M, c_mV):
    T = tf.shape(V)[1]
    st = ch.init_state(V[:, 0] - ch.JP)
    ta0, ta1, ta2 = (tf.TensorArray(tf.float32, size=T) for _ in SLOW)
    for t in tf.range(T):
        vb = S[:, t] * sb.V_SPIKE + (1 - S[:, t]) * (M[:, t] * (c_mV - ch.JP) + (1 - M[:, t]) * (V[:, t] - ch.JP))
        st, cur = ch.step(st, vb)
        ss = ch.slow_steady(tf.minimum(vb, sb.V_REF - ch.JP))
        val = [-(cur[k] - ss[k]) * ch.PA_PER_DENSITY * (1 - S[:, t]) for k in SLOW]
        ta0, ta1, ta2 = ta0.write(t, val[0]), ta1.write(t, val[1]), ta2.write(t, val[2])
    return {k: tf.transpose(ta.stack()) for k, ta in zip(SLOW, (ta0, ta1, ta2))}


def tss_of(S):
    out = np.full(S.shape, 10 ** 9, float)
    last = np.full(len(S), -10 ** 9)
    for t in range(S.shape[1]):
        last = np.where(S[:, t] > 0.5, t, last)
        out[:, t] = t - last
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dense3")
    ap.add_argument("--base", default=os.path.join(HERE, "results_fast_k"))
    args = ap.parse_args()
    t0 = time.time()
    sets = fs.load_split()
    r = json.load(open(os.path.join(args.base, args.model, "metrics.json")))
    cell = sb.BranchCell(r["frozen"], r["trained"])
    g = float(r["frozen"]["g"])
    zc = np.load(os.path.join(sp.CACHE, f"front_end_{args.model}.npz"))

    def design(split, per, seed):
        rng = np.random.default_rng(seed)
        A, y, w, tss = [], [], [], []
        for pr in sp.PROTOS:
            s = sets[(pr, split)]
            I, V = s["I"].astype(np.float32), s["V"].astype(np.float32)
            x = zc[f"{pr}_{split}"].astype(np.float32)
            c = fc.components(cell, x, I, V)
            S, M = sp.gt_masks(V)
            slow = {k: [] for k in SLOW}
            for k0 in range(0, len(V), 128):
                d = drive(tf.constant(V[k0:k0 + 128]), tf.constant(S[k0:k0 + 128]), tf.constant(M[k0:k0 + 128]),
                          tf.constant(float(r["trained"]["c_mV"])))
                for k in SLOW:
                    slow[k].append(d[k].numpy())
            slow = {k: np.concatenate(v) for k, v in slow.items()}
            slow = {k: np.concatenate([np.zeros_like(v[:, :1]), v[:, :-1]], 1) for k, v in slow.items()}
            T_ = tss_of(S)
            rr, cc = np.nonzero(c["w"])
            if len(rr) > per:
                j = rng.choice(len(rr), per, replace=False); rr, cc = rr[j], cc[j]
            cols = [g * c["cur"][k][rr, cc] for k in fc.BR] + [np.full(len(rr), g), -c["u1"][rr, cc]] + \
                   [g * slow[k][rr, cc] for k in SLOW]
            A.append(np.stack(cols, 1)); y.append((c["z"] + c["u0"] - x)[rr, cc]); w.append(c["w"][rr, cc]); tss.append(T_[rr, cc])
            print(f"  {pr} {split}: {len(rr)} samples  {time.time()-t0:.0f}s", flush=True)
        return np.concatenate(A), np.concatenate(y), np.concatenate(w), np.concatenate(tss)

    A, y, w, _ = design("train", 300000, 0)
    Av, yv, wv, tv = design("val", 100000, 1)
    sw = np.sqrt(w)
    res = {}
    for lab, ncol in (("fast-K branches only", 6), ("+ K_P, Im, Ih", 9)):
        lo = [0, 0, 0, 0, -np.inf, -20.0] + [0] * (ncol - 6)
        hi = [np.inf] * 4 + [np.inf, 40.0] + [np.inf] * (ncol - 6)
        sol = lsq_linear(A[:, :ncol] * sw[:, None], y * sw, bounds=(lo, hi))
        e = Av[:, :ncol] @ sol.x - yv          # model minus needed (x units)
        rms = float(np.sqrt(np.sum(wv * e ** 2) / np.sum(wv)))
        near = wv > 1.5
        bins = {f"{a}-{b if b < 10 ** 8 else 'inf'} ms": (round(float(e[(tv >= a) & (tv < b)].mean() / g), 1),
                                                         round(float(np.sqrt((e[(tv >= a) & (tv < b)] ** 2).mean()) / g), 1))
                for a, b in BINS}
        res[lab] = dict(scales=dict(zip(["Na", "SK", "Kv3_1", "BK", "bias", "d"] + list(SLOW[:ncol - 6]), np.round(sol.x, 3).tolist())),
                        val_rms_pA=round(rms / g, 2), val_rms_near_pA=round(float(np.sqrt((e[near] ** 2).mean())) / g, 2),
                        residual_by_time_since_spike_mean_rms_pA=bins)
        print(lab, json.dumps(res[lab], indent=1), flush=True)
    out = os.path.join(HERE, "results_slow_k_check"); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"{args.model}.json"), "w"), indent=1)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
