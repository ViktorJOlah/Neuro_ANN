"""Fit the hardcoded synaptic kernel layer (synaptic_kernel.py) on evoked data, the frozen Standard fit S underneath.

    clean: stim -> depression -> difference-of-exponentials g -> I = g (E_rev - Vm) into the Izhikevich layer.
           The CNN-LSTM front end only sees the somatic DC current.
    soft:  same, plus a 64 ms CNN over the stimulus that sets each event's conductance gain (exp, zero-init).

Same training recipe as the tcn64 D (fit_input_transform.py, peak-fix recipe): spike-free batches, 5x loss weight
in the 25 ms after each pulse, one-step (teacher-forced) warm-up, then free-running loss; Vm clipped at -40 mV.
The kernel state needs history, so each segment carries CTX ms of stimulus in front, run with Vm clamped.

    EVOKED_SET=ampa python fit_kernel.py --cell 487661754 --variant clean
Results: ../results_ampa/<cell>/kernel_<variant><tag>/ (params, metrics.json, test_predictions.npz).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

import fit_input_transform as F          # data, splits, scoring, S loading (sets TF threads from --threads)
import synaptic_kernel as SK
import standard_model as sm
import peak_metrics as PM

CTX = 300                 # ms of clamped history in front of each training segment (kernel + depression state)
SEG, SEED_MS = F.SEG, F.SEED_MS


def front_dc(S, dc, T):
    """Front-end current for a constant somatic DC (B,) over T steps."""
    I = tf.repeat(tf.constant(dc, tf.float32)[:, None], T + 63, axis=1)
    return F.front_current(S, I)


_RUN = {}
CORR = None      # membrane correction from experiments/kernel_calibration (--corr), applied to the synaptic current


def simulate(S, K, G, stim, dc, V, seed_ms=F.EVAL_SEED, chunk=10, isyn=False):
    key = (id(S), id(K), id(G))
    if key not in _RUN:
        _RUN[key] = tf.function(lambda x, ev, Vg, Sm, M, cu: SK.run_syn(S, K, x, ev, Vg, Sm, M, cu, False, return_isyn=True, corr=CORR),
                                reduce_retracing=True)
    vms, iss = [], []
    for k in range(0, len(stim), chunk):
        s = tf.constant(stim[k:k + chunk]); T = s.shape[1]
        x = front_dc(S, dc[k:k + chunk], T)
        Sm, M = sm.gt_masks(V[k:k + chunk])
        out = _RUN[key](x, SK.events(K, s, G), tf.constant(V[k:k + chunk]), tf.constant(Sm), tf.constant(M),
                        tf.constant(int(seed_ms)))
        vms.append(out[0].numpy()); iss.append(out[2].numpy())
    return (np.concatenate(vms), np.concatenate(iss)) if isyn else np.concatenate(vms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--variant", choices=("clean", "soft"), default="clean")
    ap.add_argument("--ncomp", type=int, default=1, help="difference-of-exponential components per pathway")
    ap.add_argument("--corr", default="", help="calibration.json from experiments/kernel_calibration: fixed membrane correction from S alone")
    ap.add_argument("--signed", action="store_true", help="components 2.. enter with a minus sign (membrane correction)")
    ap.add_argument("--nmda", action="store_true", help="add an NMDA-like slow kernel with Mg block")
    ap.add_argument("--delay", type=int, nargs=2, default=[2, 2], help="L1, L2/3 delay (ms) from pulse to conductance onset")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--one-step-iters", type=int, default=150)
    ap.add_argument("--one-step-gain", type=float, default=100.0)
    ap.add_argument("--lr-k", type=float, default=0.03, help="Adam lr for the kernel parameters")
    ap.add_argument("--lr-g", type=float, default=1e-3, help="Adam lr for the soft variant's gain CNN")
    ap.add_argument("--sub-frac", type=float, default=1.0)
    ap.add_argument("--event-weight", type=float, default=5.0)
    ap.add_argument("--event-win", type=int, default=25)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--init", default="", help="start from this results folder (kernel.json, gain weights)")
    ap.add_argument("--minutes", type=float, default=0, help="training data budget (0 = all training traces)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(F.HERE, "..", "results" + F.SET))
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(0)
    t0 = time.time()
    name = f"kernel_{args.variant}{'_nmda' if args.nmda else ''}{f'_k{args.ncomp}' if args.ncomp > 1 else ''}{'s' if args.signed else ''}{'_corr' if args.corr else ''}{f'_{args.minutes:g}min' if args.minutes else ''}{args.tag}"
    out = os.path.join(args.out, args.cell, name)
    os.makedirs(out, exist_ok=True)
    log = lambda *a: print(f"[{args.cell} {name} {time.time()-t0:5.0f}s]", *a, flush=True)

    global CORR
    if args.corr:
        CORR = {k: tf.constant(float(v)) for k, v in json.load(open(args.corr))["correction"].items()}
    data = F.load_data(args.cell)
    n, T = data["V"].shape
    test, train = F.split(n)
    if args.minutes:
        train = train[:max(1, int(round(args.minutes * 60000 / T)))]
    S = F.load_somatic(args.cell)
    init = json.load(open(os.path.join(args.init, "kernel.json")))["init"] if args.init else None
    K = SK.KernelLayer(args.ncomp, args.nmda, args.delay, init, args.signed)
    G = SK.build_gain_net() if args.variant == "soft" else None
    if G is not None and args.init and os.path.exists(os.path.join(args.init, "gain.weights.h5")):
        G.load_weights(os.path.join(args.init, "gain.weights.h5"))
    stim, dc, V = data["stim"].astype(np.float32), data["dc"].astype(np.float32), data["V"].astype(np.float32)
    kvars = list(K.trainable_variables)
    gvars = list(G.trainable_variables) if G is not None else []
    opt_k = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(args.lr_k, args.iters, alpha=0.1))
    opt_g = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(args.lr_g, args.iters, alpha=0.05))
    rng = np.random.default_rng(1)
    L = CTX + SEG
    clamp_ms = CTX + SEED_MS

    @tf.function(reduce_retracing=True)
    def step(s, x, Vg, Sm, M, W, one_step):
        with tf.GradientTape() as tape:
            vm, _ = SK.run_syn(S, K, x, SK.events(K, s, G), Vg, Sm, M, tf.constant(clamp_ms), one_step, corr=CORR)
            e = tf.minimum(vm[:, clamp_ms:], F.TRAIN_CLIP) - tf.minimum(Vg[:, clamp_ms:], F.TRAIN_CLIP)
            Mi = M[:, clamp_ms:]
            if one_step:
                keep = 1.0 - tf.minimum(Mi + M[:, clamp_ms - 1:-1], 1.0)
                e = e * keep * tf.sqrt(args.one_step_gain)
            Wi = W[:, clamp_ms:]
            loss = tf.reduce_mean(Wi * tf.square(e)) / tf.reduce_mean(Wi)
        gr = tape.gradient(loss, kvars + gvars)
        gr = [tf.where(tf.math.is_finite(z), z, tf.zeros_like(z)) for z in gr]
        gk, _ = tf.clip_by_global_norm(gr[:len(kvars)], 1.0)
        opt_k.apply_gradients(zip(gk, kvars))
        if gvars:
            gg, _ = tf.clip_by_global_norm(gr[len(kvars):], 1.0)
            opt_g.apply_gradients(zip(gg, gvars))
        return loss

    hist = []
    for it in range(args.iters):
        B = args.batch
        tr = rng.choice(train, B)
        st = rng.integers(CTX, T - SEG, B)
        for b in range(int(round(args.sub_frac * B))):
            for _ in range(50):
                if not (V[tr[b], st[b]:st[b] + SEG] > -20).any():
                    break
                tr[b] = rng.choice(train); st[b] = rng.integers(CTX, T - SEG)
        s = np.stack([stim[i, a - CTX:a + SEG] for i, a in zip(tr, st)])
        Vg = np.stack([V[i, a - CTX:a + SEG] for i, a in zip(tr, st)])
        Sm, M = sm.gt_masks(Vg)
        W = np.ones_like(Vg)
        if args.event_weight:
            pul = (s.sum(-1) > 0).astype(np.float32)
            hit = np.stack([np.convolve(p, np.ones(args.event_win, np.float32))[:L] for p in pul]) > 0
            W += args.event_weight * hit
        x = front_dc(S, dc[tr], L)
        loss = float(step(tf.constant(s), x, tf.constant(Vg), tf.constant(Sm), tf.constant(M), tf.constant(W),
                          tf.constant(it < args.one_step_iters)))
        hist.append(loss)
        if it % 25 == 0 or it == args.iters - 1:
            log(f"it {it} loss {loss:.4f} (mean last 25: {np.mean(hist[-25:]):.4f})",
                json.dumps({k: np.round(v, 3).tolist() for k, v in K.as_json().items()}))

    vm, isyn = simulate(S, K, G, stim[test], dc[test], V[test], isyn=True)
    res = F.score(vm, V[test], stim[test])
    K.on.assign(0.0)
    res["no_synaptic_input"] = F.score(simulate(S, K, G, stim[test], dc[test], V[test]), V[test], stim[test])
    K.on.assign(1.0)
    res["peaks"] = PM.summarize(PM.events(V, stim, vm, test))
    log(json.dumps(res["peaks"]))
    log(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}))
    kj = K.as_json()
    init_out = dict(tau_r=kj["tau_r"], tau_d=kj["tau_d"], w=kj["w"], E_rev=kj["E_rev"],
                    lg_U=np.asarray(K.lg_U.numpy()).tolist(), tau_rec=kj["tau_rec"])
    json.dump(dict(values=kj, init=init_out, ncomp=args.ncomp, nmda=args.nmda, delay=args.delay, signed=args.signed, corr=args.corr), open(os.path.join(out, "kernel.json"), "w"), indent=1)
    if G is not None:
        G.save_weights(os.path.join(out, "gain.weights.h5"))
    np.savez_compressed(os.path.join(out, "test_predictions.npz"), vm=vm.astype(np.float32), test=test, isyn=isyn.astype(np.float32))
    json.dump(dict(cell=args.cell, model=name, args=vars(args), kernel=kj, train_traces=[int(i) for i in train],
                   test_traces=[int(i) for i in test], eval=res, loss_history=hist, iters=args.iters,
                   n_params=int(sum(np.prod(v.shape) for v in kvars + gvars)), wall_s=time.time() - t0),
              open(os.path.join(out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
