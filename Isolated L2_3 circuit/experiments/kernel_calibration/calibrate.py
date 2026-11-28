"""Step 0 of the kernel-layer rerun: calibrate each cell's membrane correction from its somatic model S alone.

With the synaptic kernel layer as the default (evoked_synaptic/code/synaptic_kernel.py, PR 22), synaptic current
bypasses the CNN-LSTM front end and enters the Izhikevich layer directly. The bare Izhikevich membrane responds to fast
current more slowly than the cell (the front end normally corrects this), so the layer applies a per-cell correction

    I_izh = k0 * I_syn  +  c1 * LP_tau1(I_syn)  +  c2 * LP_tau2(I_syn)    (LP = unit-area exponential low-pass, recursive)

In the evoked fits (PR 22) this was a second, subtracted kernel component learned from evoked recordings. Library
cells have no evoked recordings, so here (k0, c1, tau1, c2, tau2) are fitted so that the bypass path reproduces S's own
response to injected PSC-shaped currents through the front end, which is what S was trained on. Inputs: subthreshold
DC holding plus Poisson trains of alpha currents (taus 2-9 ms, excitatory and inhibitory, sizes scaled by the cell's
rheobase). Loss: Vm MSE (clipped at -40 mV, 30 ms after spikes masked), from 300 ms on.

    python calibrate.py --cell 487661754            (Standard)  | --cell 521409057 (data_budget fit) | --cell <library id>
Output: results/<cell>/calibration.json (+ eval numbers vs no correction).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_th = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
import tensorflow as tf  # noqa: E402
tf.config.threading.set_intra_op_parallelism_threads(_th)
tf.config.threading.set_inter_op_parallelism_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(ROOT, "Standard"))
sys.path.insert(0, os.path.join(EXP, "evoked_synaptic", "code"))
sys.path.insert(0, os.path.join(EXP, "population"))
import standard_model as sm  # noqa: E402
import synaptic_kernel as SK  # noqa: E402

T, T0, N_TRAIN, N_VAL, N_TEST, BATCH = 1200, 300, 40, 8, 12, 12
TAUS = np.array([2.0, 2.8, 3.5, 5.5, 5.8, 8.5])


def load_cell(cell, model_dir=None):
    if model_dir:                                   # any fit folder (Standard, data_budget, real_cells, library)
        sys.path.insert(0, ROOT)
        from circuit.population import load_model
        return load_model(model_dir, os.path.join(ROOT, "Standard"))
    if cell == "487661754":
        return sm.load()
    d = os.path.join(EXP, "data_budget", "results", cell, "full")
    if os.path.isdir(d):
        m = json.load(open(os.path.join(d, "metrics.json")))
        c = sm.load()
        w = np.load(os.path.join(d, "front_end.npz"))
        c.net.set_weights([w[f"w{i}"] for i in range(len(w.files))])
        c.p = {k: tf.constant(float(m["izhikevich"].get(k, float(c.p[k].numpy()))), tf.float32) for k in c.p}
        c.set(m["branches"])
        return c
    import library
    return library.load_cell(cell)


def rheobase(S, hi=800.0, step=10.0):
    amps = np.arange(step, hi + step, step, dtype=np.float32)
    I = np.zeros((len(amps), 1300), np.float32); I[:, 200:1200] = amps[:, None]
    _, s = S.simulate(I)
    fired = s[:, 200:1250].any(1)
    return float(amps[np.argmax(fired)]) if fired.any() else float(hi)


def make_inputs(n, rheo, rng):
    """DC (n,) and alpha-PSC current (n, T). Sizes scale with rheobase so PSPs are a few mV in every cell."""
    sc = rheo / 83.8
    dc = rng.uniform(0.0, 0.55, n) * rheo
    I = np.zeros((n, T))
    t = np.arange(T, dtype=float)
    for i in range(n):
        for sign, rate, wlo, whi in ((1, rng.uniform(10, 50), 10, 70), (-1, rng.uniform(3, 20), 10, 60)):
            for ts in np.sort(rng.uniform(0, T, rng.poisson(rate * T / 1000))):
                tau = rng.choice(TAUS); w = sign * rng.uniform(wlo, whi) * sc
                k = t - ts
                I[i] += np.where(k >= 0, w * (k / tau) * np.exp(1 - k / tau), 0.0)
    return dc.astype(np.float32), I.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--model-dir", help="fit folder to load instead of the default for --cell")
    ap.add_argument("--tag", help="results subfolder name (default: --cell)")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    t_start = time.time()
    out = os.path.join(a.out, a.tag or a.cell); os.makedirs(out, exist_ok=True)
    log = lambda *m: print(f"[{a.cell} {time.time()-t_start:5.0f}s]", *m, flush=True)
    S = load_cell(a.cell, a.model_dir)
    rheo = rheobase(S)
    log(f"rheobase {rheo:.0f} pA")
    rng = np.random.default_rng(int(a.cell) % 2**31)
    dc, I = make_inputs(N_TRAIN + N_VAL + N_TEST, rheo, rng)
    V_tgt, s_tgt = S.simulate(dc[:, None] + I)                       # S's own response, current through the front end
    x_dc = S.current(np.repeat(dc[:, None], T, 1))                     # front end sees the DC only
    K = SK.KernelLayer(1); K.on.assign(0.0)
    V_rest = np.full((len(dc), T), -82.0, np.float32)
    Sm, M = sm.gt_masks(V_rest)
    keep = np.ones((len(dc), T), np.float32); keep[:, :T0] = 0
    for i, t in zip(*np.nonzero(s_tgt)):
        keep[i, max(t - 5, 0):t + 30] = 0
    v = dict(ln_k0=tf.Variable(0.0), c1=tf.Variable(0.0), c2=tf.Variable(0.0),
             ln_t1=tf.Variable(np.log(4.0), dtype=tf.float32), ln_t2=tf.Variable(np.log(40.0), dtype=tf.float32))

    def corr():
        t1 = 1.0 + tf.exp(v["ln_t1"])
        return dict(k0=tf.exp(v["ln_k0"]), c1=v["c1"], tau1=t1, c2=v["c2"], tau2=t1 + 2.0 + tf.exp(v["ln_t2"]))

    C = {k: tf.constant(z) for k, z in dict(x=x_dc, V=V_rest, Sm=Sm, M=M, I=I).items()}

    def run(idx, c):
        g = lambda k: tf.gather(C[k], idx)
        return SK.run_syn(S, K, g("x"), tf.zeros((tf.shape(idx)[0], T, 2)), g("V"), g("Sm"), g("M"), tf.constant(1), False,
                          i_ext=g("I"), corr=c)[0]

    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(a.lr, a.iters, alpha=0.1))

    @tf.function(reduce_retracing=True)
    def step(idx):
        with tf.GradientTape() as tape:
            vm = run(idx, corr())
            e = tf.minimum(vm, -40.0) - tf.minimum(tf.gather(tf.constant(V_tgt), idx), -40.0)
            kp = tf.gather(tf.constant(keep), idx)
            ms = tf.cast(vm > -20.0, tf.float32)                       # also mask 30 ms after each model spike
            ms = tf.nn.max_pool1d(tf.pad(ms, [[0, 0], [30, 0]])[:, :, None], 31, 1, "VALID")[:, :, 0]
            kp = kp * tf.stop_gradient(1.0 - ms)
            loss = tf.reduce_sum(kp * e * e) / tf.reduce_sum(kp)
        g = tape.gradient(loss, list(v.values()))
        g = [tf.where(tf.math.is_finite(z), z, tf.zeros_like(z)) for z in g]
        opt.apply_gradients(zip(g, list(v.values())))
        return loss

    val_idx = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    test = np.arange(N_TRAIN + N_VAL, N_TRAIN + N_VAL + N_TEST)
    run_tf = tf.function(run, reduce_retracing=True)

    def score(c, test=test):
        vm = run_tf(tf.constant(test), c).numpy()
        e = np.minimum(vm, -40) - np.minimum(V_tgt[test], -40)
        kp = keep[test] > 0
        for i, t in zip(*np.nonzero((vm[:, 1:] > -20) & (vm[:, :-1] <= -20))):
            kp[i, max(t - 5, 0):t + 30] = False
        d2 = lambda X: X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
        k2 = kp[:, 2:] & kp[:, 1:-1] & kp[:, :-2]
        nsp = int(((vm[:, 1:] > -20) & (vm[:, :-1] <= -20)).sum())
        return dict(rmse_mV=float(np.sqrt(np.mean(e[kp] ** 2))),
                    rough_ratio=float(np.sqrt(np.mean(d2(vm)[k2] ** 2)) / np.sqrt(np.mean(d2(V_tgt[test])[k2] ** 2))),
                    spikes_model=nsp, spikes_target=int(s_tgt[test].sum()), sd_target_mV=float(np.std(V_tgt[test][kp]))), vm

    NONE = dict(k0=tf.constant(1.0), c1=tf.constant(0.0), tau1=tf.constant(2.0), c2=tf.constant(0.0), tau2=tf.constant(20.0))
    val = lambda c: score(c, val_idx)[0]["rmse_mV"]                     # validation traces pick the iterate
    best = (val(NONE), None, -1)
    log(f"no correction: validation RMSE {best[0]:.3f} mV")
    for it in range(a.iters):
        idx = rng.choice(N_TRAIN, BATCH, replace=False)
        l = float(step(tf.constant(idx)))
        if it % 25 == 24 or it == a.iters - 1:
            r = val(corr())
            if r < best[0]:
                best = (r, {k: float(x) for k, x in corr().items()}, it)
            log(f"it {it} loss {l:.4f} val {r:.3f}", {k: round(float(x), 3) for k, x in corr().items()})

    c_fit = best[1] or {k: float(x.numpy()) for k, x in NONE.items()}   # no iterate beat 'no correction' -> identity
    res_fit, vm_fit = score({k: tf.constant(x) for k, x in c_fit.items()})
    res_none, vm_none = score(NONE)
    log("fit", res_fit, "no correction", res_none)
    json.dump(dict(cell=a.cell, model_dir=a.model_dir, rheobase_pA=rheo, correction=c_fit, best_iter=best[2], test=res_fit, test_no_correction=res_none,
                   iters=a.iters, wall_s=time.time() - t_start), open(os.path.join(out, "calibration.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out, "test_traces.npz"), V_target=V_tgt[test], vm_fit=vm_fit, vm_none=vm_none,
                        I_psc=I[test], dc=dc[test])


if __name__ == "__main__":
    main()
