"""Can the frozen S produce the fast L2/3 EPSP peak at all? Free-current ceiling on isolated events.

Takes isolated, spike-free pulses of one pathway (same criteria as peak_metrics.py), cuts 350 ms segments
(recorded-Vm seed up to 50 ms before the pulse), and optimises a free current u(t) per segment through the
frozen S on the recorded Vm (plain MSE, or with --event-weight like fit_input_transform). Reports the EPSP
peak ratio model/NEURON. If S reaches ~1, the under-prediction is D's (training), not S's.
    python oracle_events.py --cell 487661754 [--path 1 --n 48 --iters 400]
"""
import argparse, json, os, sys, time
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
os.environ.setdefault("EVOKED_SET", "ampa")
import fit_input_transform as F
import peak_metrics as PM

ap = argparse.ArgumentParser()
ap.add_argument("--cell", default="487661754"); ap.add_argument("--path", type=int, default=1)
ap.add_argument("--n", type=int, default=48); ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--event-weight", type=float, default=0.0)
a = ap.parse_args()
t0 = time.time()
z = F.load_data(a.cell); V, stim, dc = z["V"].astype(np.float32), z["stim"].astype(np.float32), z["dc"].astype(np.float32)
PRE, POST, SEED = 200, 150, 150
ev = []
for i in range(len(V)):
    for t in np.nonzero(stim[i, :, a.path] > 0)[0]:
        if t < 400 or t + POST > V.shape[1]:
            continue
        w = stim[i, t - 100:t + 150].copy(); w[100, a.path] = 0
        if w.sum() == 0 and not (V[i, t - PRE:t + POST] > -20).any():
            ev.append((i, t))
rng = np.random.default_rng(0)
ev = [ev[k] for k in rng.choice(len(ev), min(a.n, len(ev)), replace=False)]
Vg = np.stack([V[i, t - PRE:t + POST] for i, t in ev])
d = np.array([dc[i] for i, _ in ev], np.float32)
S = F.load_somatic(a.cell)
Sm, M = [tf.constant(x) for x in F.sm.gt_masks(Vg)]
u = tf.Variable(tf.zeros(Vg.shape))
W = np.ones_like(Vg); W[:, PRE:PRE + 25] += a.event_weight
W = tf.constant(W)
opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(0.05, a.iters, alpha=0.05))
Vgt = tf.constant(Vg)


def fwd():
    I = F.OUT_SCALE * u + d[:, None]
    I = tf.concat([tf.repeat(I[:, :1], 63, axis=1), I], 1)
    vm, _ = type(S).run.python_function(S, F.front_current(S, I), Vgt, Sm, M, tf.constant(SEED), False)
    return vm


@tf.function
def step():
    with tf.GradientTape() as tape:
        vm = fwd()
        e = tf.minimum(vm[:, SEED:], F.TRAIN_CLIP) - tf.minimum(Vgt[:, SEED:], F.TRAIN_CLIP)
        loss = tf.reduce_mean(W[:, SEED:] * tf.square(e)) / tf.reduce_mean(W[:, SEED:])
    g = tape.gradient(loss, [u])
    opt.apply_gradients(zip(g, [u]))
    return loss


for it in range(a.iters):
    l = float(step())
    if it % 50 == 0 or it == a.iters - 1:
        print(f"[oracle {a.cell} {time.time()-t0:4.0f}s] it {it} loss {l:.4f}", flush=True)
vm = fwd().numpy()
win = PM.WIN[a.path]
b, bm = Vg[:, PRE - 10:PRE].mean(1), vm[:, PRE - 10:PRE].mean(1)
pk = Vg[:, PRE:PRE + win].max(1) - b
pm = np.minimum(vm[:, PRE:PRE + win], -20).max(1) - bm
res = dict(cell=a.cell, path=a.path, n=len(ev), peak_mean=float(pk.mean()), peak_model_mean=float(pm.mean()),
           ratio=float(pm.mean() / pk.mean()), tpk=float(np.median(Vg[:, PRE:PRE + win].argmax(1))),
           tpk_model=float(np.median(vm[:, PRE:PRE + win].argmax(1))),
           rmse_event_mV=float(np.sqrt(np.mean((vm[:, PRE:PRE + 60] - Vg[:, PRE:PRE + 60]) ** 2))),
           u_peak_pA=float((F.OUT_SCALE * u.numpy()[:, PRE:PRE + 20]).max(1).mean()), event_weight=a.event_weight)
print(json.dumps(res))
out = os.path.join(F.HERE, "..", "results_ampa", a.cell, f"oracle_events_p{a.path}" + (f"_ew{a.event_weight:g}" if a.event_weight else ""))
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out, "oracle.npz"), u=F.OUT_SCALE * u.numpy(), vm=vm, V=Vg, dc=d, ev=np.array(ev))
