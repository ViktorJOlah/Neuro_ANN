"""Ceiling for any input transform: the best somatic current a cell's fitted somatic model S can be driven with.

For a few test traces, a free current u(t) (one value per ms, 4000 per trace) is optimised directly through the
frozen S to reproduce the recorded evoked Vm. Whatever error is left is S's own limit (a D can only do worse);
the gap between this and a trained D is what D still misses.
    python oracle_current.py --cell 521409057 [--n 6 --iters 300]   ->  ../results/<cell>/oracle/metrics.json
"""
SET = ('_' + __import__('os').environ['EVOKED_SET']) if __import__('os').environ.get('EVOKED_SET') else ''   # e.g. EVOKED_SET=ampa -> data_ampa/, results_ampa/
import argparse, json, os, sys, time
import numpy as np
_th = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 2
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(_th)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})   # grappler rewrites AddN in the while-loop gradient into tmp_var ops that clash (AlreadyExists)
import fit_input_transform as F

ap = argparse.ArgumentParser()
ap.add_argument("--cell", default="521409057"); ap.add_argument("--somatic", default=None)
ap.add_argument("--n", type=int, default=6); ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
t0 = time.time()
data = F.load_data(a.cell)
test, _ = F.split(len(data["V"]))
idx = test[np.linspace(0, len(test) - 1, a.n).astype(int)]
stim, dc, V = data["stim"][idx].astype(np.float32), data["dc"][idx].astype(np.float32), data["V"][idx].astype(np.float32)
S = F.load_somatic(a.somatic or a.cell)
u = tf.Variable(tf.zeros(V.shape))
opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(0.05, a.iters, alpha=0.05))
Sm, M = [tf.constant(x) for x in F.sm.gt_masks(V)]
Vg = tf.constant(V)


@tf.function
def step():
    with tf.GradientTape() as tape:
        I = F.OUT_SCALE * u + dc[:, None]
        I = tf.concat([tf.repeat(I[:, :1], 63, axis=1), I], 1)
        x = F.front_current(S, I)
        vm, _ = type(S).run.python_function(S, x, Vg, Sm, M, tf.constant(F.EVAL_SEED), False)
        e = tf.minimum(vm[:, F.EVAL_SEED:], F.TRAIN_CLIP) - tf.minimum(Vg[:, F.EVAL_SEED:], F.TRAIN_CLIP)
        loss = tf.reduce_mean(tf.square(e))
    g = tape.gradient(loss, [u])
    opt.apply_gradients(zip(g, [u]))
    return loss


for it in range(a.iters):
    l = float(step())
    if it % 25 == 0 or it == a.iters - 1:
        print(f"[{a.cell} oracle {time.time()-t0:5.0f}s] it {it} loss {l:.3f}", flush=True)
vm = F.simulate(S, lambda s, training=False: u[:, :, None], stim, dc, V)
res = F.score(vm, V, stim)
print(json.dumps({k: round(v, 3) for k, v in res.items() if isinstance(v, float)}))
out = os.path.join(F.HERE, "..", "results" + SET, a.cell, "oracle" if not a.somatic else f"oracle_S{a.somatic}")
os.makedirs(out, exist_ok=True)
json.dump(dict(cell=a.cell, somatic=a.somatic or a.cell, traces=[int(i) for i in idx], eval=res, iters=a.iters),
          open(os.path.join(out, "metrics.json"), "w"), indent=1)
np.savez_compressed(os.path.join(out, "oracle.npz"), u=u.numpy() * F.OUT_SCALE, vm=vm, idx=idx)
