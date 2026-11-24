"""Fit a per-cell input transform D (stimulus pulses -> somatic-equivalent current) in front of a frozen somatic model S.

    stim (L1, L2/3 pulse intensities, 1 ms) --D--> I_syn (pA) --+--> S = Standard model fitted to the same cell's
                                       somatic DC current -----+     somatic current injections (front end + Izh + branches)

S is the cell's data-budget fit (experiments/data_budget/results/<cell>/<budget>/) and stays frozen; only D is trained,
end to end through S, on the recorded somatic Vm (clipped at -20 mV). So D's output is defined as "the current that,
injected at the soma of this cell's model, reproduces the evoked response": dendritic integration, NMDA, short-term
plasticity and release statistics all end up in D, the somatic transform stays in S.

Variants of D (--model):
  linear     causal FIR filter, 256 ms per pathway (what a fixed PSC kernel layer can do at best)
  tcn64      dilated causal CNN (kernel 2, dilations 1..32), receptive field 64 ms
  tcn256     same, dilations 1..128, receptive field 256 ms
  cnnlstm64  the Standard front-end architecture (Conv1D 32x5 x2 -> LSTM 32 -> Dense) on 64 ms windows of the stimulus
  gru        GRU(32) over the whole trace (unbounded memory)

Run from this folder:
    python fit_input_transform.py --cell 521409057 --model tcn256 [--minutes 10] [--somatic 513531471]
Results: ../results/<cell>/<model>_<minutes>min/ (weights, metrics.json, test predictions).
"""
from __future__ import annotations

SET = ('_' + __import__('os').environ['EVOKED_SET']) if __import__('os').environ.get('EVOKED_SET') else ''   # e.g. EVOKED_SET=ampa -> data_ampa/, results_ampa/
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
import keras

_th = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 2
tf.config.threading.set_intra_op_parallelism_threads(_th)      # before standard_model creates tensors
tf.config.threading.set_inter_op_parallelism_threads(1)
if "--no-grappler" in sys.argv:   # some graphs (cnnlstm64) hit a grappler AddN rewrite that fails with AlreadyExists tmp_var
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(HERE, "..", "data" + SET)
BUDGET = os.path.join(ROOT, "experiments", "data_budget", "results")
sys.path.insert(0, os.path.join(ROOT, "Standard"))
import standard_model as sm  # noqa: E402

CTX, SEG, SEED_MS, BATCH, CLIP = 320, 500, 150, 32, -20.0
TRAIN_CLIP = -40.0            # training loss clips Vm here: a missed or extra spike costs a bounded error
EVAL_SEED = 250               # test traces: recorded Vm for the first 250 ms (before any stimulus), then free running
N_TEST = 20
OUT_SCALE = 100.0             # D output in units of 100 pA


# ------------------------------------------------------------------ models
def build_D(kind):
    inp = keras.Input((None, 2))
    if kind == "linear":
        y = keras.layers.Conv1D(1, 256, padding="causal", use_bias=True, kernel_initializer="zeros")(inp)
    elif kind in ("tcn64", "tcn256"):
        dil = [1, 2, 4, 8, 16, 32] + ([64, 128] if kind == "tcn256" else [])
        x = keras.layers.Conv1D(32, 1)(inp)
        for d in dil:
            r = keras.layers.Conv1D(32, 2, dilation_rate=d, padding="causal", activation="relu")(x)
            r = keras.layers.Conv1D(32, 1)(r)
            x = keras.layers.Add()([x, r])
        x = keras.layers.Activation("relu")(x)
        y = keras.layers.Conv1D(1, 1, kernel_initializer="zeros")(x)
    elif kind == "gru":
        x = keras.layers.GRU(32, return_sequences=True)(inp)
        y = keras.layers.Conv1D(1, 1, kernel_initializer="zeros")(x)
    elif kind == "cnnlstm64":
        net = keras.Sequential([keras.layers.Conv1D(32, 5, padding="causal", activation="relu"),
                                keras.layers.Conv1D(32, 5, padding="causal", activation="relu"),
                                keras.layers.LSTM(32), keras.layers.Dense(1, kernel_initializer="zeros")])
        net.build((None, 64, 2))

        class Windowed(keras.Model):
            def __init__(self):
                super().__init__()
                self.net = net

            def call(self, s):
                B, T = tf.shape(s)[0], tf.shape(s)[1]
                pad = tf.zeros([B, 63, 2])
                w = tf.signal.frame(tf.concat([pad, s], 1), 64, 1, axis=1)       # (B, T, 64, 2)
                return tf.reshape(self.net(tf.reshape(w, (-1, 64, 2))), (B, T, 1))
        m = Windowed()
        m(tf.zeros((1, 8, 2)))
        return m
    else:
        raise ValueError(kind)
    return keras.Model(inp, y)


def load_somatic(cell, budget="full"):
    if cell == "487661754":          # the Standard's own cell: S is the Standard model itself
        return sm.load()
    d = os.path.join(BUDGET, cell, budget)
    m = json.load(open(os.path.join(d, "metrics.json")))
    c = sm.load()
    w = np.load(os.path.join(d, "front_end.npz"))
    c.net.set_weights([w[f"w{i}"] for i in range(len(w.files))])
    c.p = {k: tf.constant(float(m["izhikevich"].get(k, float(c.p[k].numpy()))), tf.float32) for k in c.p}
    c.set(m["branches"])
    return c


def front_current(S, I):
    """Differentiable S.current: I (B, T) pA with >= 63 ms of history in front; returns (B, T-63)."""
    B, T = tf.shape(I)[0], tf.shape(I)[1]
    w = tf.signal.frame(I, 64, 1, axis=1)                       # (B, T-63, 64)
    y = S.net(tf.reshape(w, (-1, 64, 1)) / sm.I_SCALE)
    return S.p["g"] * (I[:, 63:] + S.p["I0"]) + tf.reshape(y, (B, T - 63))


# ------------------------------------------------------------------ data
def load_data(tag):
    z = np.load(os.path.join(DATA, tag, "evoked.npz"))
    return {k: z[k] for k in z.files}


def split(n, seed=0):
    idx = np.arange(n)
    test = idx[np.arange(N_TEST) * (n // N_TEST)] if n > N_TEST else idx
    train = np.setdiff1d(idx, test)
    np.random.default_rng(seed).shuffle(train)
    return test, train


def spikes(V):
    return (V[:, 1:] > CLIP) & (V[:, :-1] <= CLIP)


# ------------------------------------------------------------------ scoring
def gamma(tt, tm, T, win=4):
    if len(tt) == 0 and len(tm) == 0:
        return np.nan
    nc = sum(np.any(np.abs(np.asarray(tm) - t) <= win) for t in tt) if len(tm) else 0
    r = len(tm) / T
    exp = 2 * r * win * len(tt)
    norm = 1 - 2 * r * win
    return (nc - exp) / (0.5 * (len(tt) + len(tm)) * norm) if len(tt) + len(tm) else np.nan


def score(vm, V, stim, t0=EVAL_SEED):
    """Free-running part after t0. Vm RMSE (clipped), RMSE away from spikes, evoked-response RMSE per pathway
    (0-150 ms after pulses of one pathway with no pulse of the other within 150 ms), spike Γ / counts."""
    vm, V = vm[:, t0:], V[:, t0:]
    st = stim[:, t0:]
    T = V.shape[1]
    e = np.minimum(vm, CLIP) - np.minimum(V, CLIP)
    away = np.ones_like(V, bool)
    for S in (spikes(V), spikes(vm)):
        for i, t in zip(*np.nonzero(S)):
            away[i, max(t - 5, 0):t + 30] = False
    out = dict(rmse_mV=float(np.sqrt(np.mean(e ** 2))), rmse_away_mV=float(np.sqrt(np.mean(e[away] ** 2))))
    for j, p in enumerate(("L1", "L23")):
        m = np.zeros_like(V, bool)
        for i, t in zip(*np.nonzero(st[..., j] > 0)):
            other = st[i, max(t - 150, 0):t + 150, 1 - j].sum() > 0
            if not other:
                m[i, t:t + 150] = True
        m &= away
        out[f"rmse_{p}_mV"] = float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan
        # response size: variance of recorded Vm in these windows (for a relative error)
        out[f"sd_{p}_mV"] = float(np.std(V[m])) if m.any() else np.nan
    gs, nt, nm = [], [], []
    for i in range(len(V)):
        tt, tm = np.nonzero(spikes(V[i:i + 1])[0])[0], np.nonzero(spikes(vm[i:i + 1])[0])[0]
        nt.append(len(tt)); nm.append(len(tm))
        g = gamma(tt, tm, T)
        if not np.isnan(g):
            gs.append(g)
    out.update(gamma=float(np.mean(gs)) if gs else np.nan, n_true=int(sum(nt)), n_model=int(sum(nm)),
               count_abs_err=float(np.mean(np.abs(np.array(nt) - np.array(nm)))),
               count_r=float(np.corrcoef(nt, nm)[0, 1]) if np.std(nt) > 0 and np.std(nm) > 0 else np.nan)
    return out


def simulate(S, D, stim, dc, V, seed_ms=EVAL_SEED, chunk=10):
    vms = []
    for k in range(0, len(stim), chunk):
        s = tf.constant(stim[k:k + chunk]); n = s.shape[0]; T = s.shape[1]
        I = OUT_SCALE * D(s)[..., 0] + dc[k:k + chunk, None]
        I = tf.concat([tf.repeat(I[:, :1], 63, axis=1), I], 1)
        x = front_current(S, I)
        Vg = tf.constant(V[k:k + chunk])
        Sm, M = sm.gt_masks(V[k:k + chunk])
        vm, _ = S.run(x, Vg, tf.constant(Sm), tf.constant(M), tf.constant(int(seed_ms)), tf.constant(False))
        vms.append(vm.numpy())
    return np.concatenate(vms)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="evoked dataset tag (cell id)")
    ap.add_argument("--somatic", default=None, help="cell whose fitted somatic model S to use (default: --cell)")
    ap.add_argument("--somatic-budget", default="full")
    ap.add_argument("--model", default="tcn256")
    ap.add_argument("--minutes", type=float, default=0, help="training data budget (0 = all training traces)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--no-grappler", action="store_true")
    ap.add_argument("--sub-frac", type=float, default=0.0, help="share of each batch drawn from spike-free segments (small EPSPs otherwise drown in spike-timing error)")
    ap.add_argument("--tag", default="", help="suffix for the results folder")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results" + SET))
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(0)
    t0 = time.time()
    tagm = f"{args.model}_{'all' if not args.minutes else f'{args.minutes:g}min'}{args.tag}"
    out = os.path.join(args.out, args.cell, tagm)
    os.makedirs(out, exist_ok=True)
    log = lambda *a: print(f"[{args.cell} {tagm} {time.time()-t0:5.0f}s]", *a, flush=True)

    data = load_data(args.cell)
    n, T = data["V"].shape
    test, train = split(n)
    if args.minutes:
        train = train[:max(1, int(round(args.minutes * 60000 / T)))]
    log(f"train {len(train)} traces ({len(train)*T/60000:.1f} min), test {len(test)}")
    S = load_somatic(args.somatic or args.cell, args.somatic_budget)
    D = build_D(args.model)
    stim, dc, V = data["stim"].astype(np.float32), data["dc"].astype(np.float32), data["V"].astype(np.float32)
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(args.lr, args.iters, alpha=0.05))
    rng = np.random.default_rng(1)

    @tf.function(reduce_retracing=True)
    def step(s, d, Vg, Sm, M):
        with tf.GradientTape() as tape:
            I = OUT_SCALE * D(s, training=True)[..., 0] + d[:, None]
            x = front_current(S, I[:, CTX - 63:])
            vm, _ = type(S).run.python_function(S, x, Vg, Sm, M, tf.constant(SEED_MS), False)   # inlined: a nested tf.function hits AlreadyExists on the 2nd step
            e = tf.minimum(vm[:, SEED_MS:], TRAIN_CLIP) - tf.minimum(Vg[:, SEED_MS:], TRAIN_CLIP)
            loss = tf.reduce_mean(tf.square(e))
        g = tape.gradient(loss, D.trainable_variables)
        g = [tf.where(tf.math.is_finite(x), x, tf.zeros_like(x)) for x in g]
        g, _ = tf.clip_by_global_norm(g, 1.0)
        opt.apply_gradients(zip(g, D.trainable_variables))
        return loss

    hist = []
    for it in range(args.iters):
        tr = rng.choice(train, BATCH)
        st = rng.integers(CTX, T - SEG, BATCH)
        for b in range(int(round(args.sub_frac * BATCH))):   # resample until the segment holds no recorded spike
            for _ in range(50):
                if not (V[tr[b], st[b]:st[b] + SEG] > -20).any():
                    break
                tr[b] = rng.choice(train); st[b] = rng.integers(CTX, T - SEG)
        s = np.stack([stim[i, a - CTX:a + SEG] for i, a in zip(tr, st)])
        Vg = np.stack([V[i, a:a + SEG] for i, a in zip(tr, st)])
        Sm, M = sm.gt_masks(Vg)
        loss = float(step(tf.constant(s), tf.constant(dc[tr]), tf.constant(Vg), tf.constant(Sm), tf.constant(M)))
        hist.append(loss)
        if it % 25 == 0 or it == args.iters - 1:
            log(f"it {it} loss {loss:.3f} (mean last 25: {np.mean(hist[-25:]):.3f})")

    vm = simulate(S, D, stim[test], dc[test], V[test])
    res = score(vm, V[test], stim[test])
    zero = lambda s, training=False: tf.zeros(tf.concat([tf.shape(s)[:2], [1]], 0))
    res["no_synaptic_input"] = score(simulate(S, zero, stim[test], dc[test], V[test]), V[test], stim[test])
    # repeat ceiling: same pulses, new release draws (only for control traces in the test set)
    ctrl = list(data.get("ctrl_idx", []))
    rep = [k for k, i in enumerate(test) if i in ctrl]
    if rep and len(data.get("V_repeat", [])):
        Vr = np.stack([data["V_repeat"][ctrl.index(test[k])] for k in rep])
        res["repeat_ceiling"] = score(Vr, V[test][rep], stim[test][rep])
        res["model_on_repeat_subset"] = score(vm[rep], V[test][rep], stim[test][rep])
    log(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)}))
    D.save_weights(os.path.join(out, "D.weights.h5"))
    np.savez_compressed(os.path.join(out, "test_predictions.npz"), vm=vm.astype(np.float32), test=test)
    json.dump(dict(cell=args.cell, somatic=args.somatic or args.cell, somatic_budget=args.somatic_budget,
                   model=args.model, train_traces=[int(i) for i in train], train_minutes=len(train) * T / 60000,
                   test_traces=[int(i) for i in test], eval=res, loss_history=hist, iters=args.iters,
                   n_params=int(sum(np.prod(v.shape) for v in D.trainable_variables)), wall_s=time.time() - t0),
              open(os.path.join(out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
