"""Ablation of the fast-K branches on the square test steps (continuous): first-AHP trough and spikes
per trace with Kv3.1 and/or BK edges zeroed, for the results_fast_k fits. Run from "Isolated L2_3 circuit/"."""
import sys, json, numpy as np, tensorflow as tf
sys.path.insert(0,'experiments/conductance_branches')
import spiking_branches as sb
sp, fs, base = sb.sp, sb.fs, sb.base
sets = fs.load_split(); s = sets[("square","test")]
V = s["V"]; S, M = sp.gt_masks(V); S=S.astype(bool)
rows = np.flatnonzero(S.any(1))
def trough(v, spk, f):
    w = v[f+2:f+60][~spk[f+2:f+60]]; return w.min() if w.size else np.nan
gt = np.nanmean([trough(V[i], S[i]|M[i].astype(bool), np.flatnonzero(S[i])[0]) for i in rows])
for m in ("dense3","cnnlstm","dense_cnnlstm"):
    r = json.load(open(f"experiments/conductance_branches/results_fast_k/{m}/metrics.json"))
    x = np.load(f"{sp.CACHE}/front_end_{m}.npz")["square_test"][rows]
    cell = sb.BranchCell(r["frozen"], r["trained"]); f = tf.function(cell.run)
    for lab, kv, bk in (("full",1,1),("no Kv3.1",0,1),("no BK",1,0),("neither",0,0)):
        cell.kv_on.assign(kv); cell.bk_on.assign(bk)
        vm, spk = [a.numpy() for a in f(tf.constant(x), tf.constant(V[rows]), tf.constant(S[rows].astype(np.float32)), tf.constant(M[rows]), tf.constant(base.SEED_MS), tf.constant(False))]
        spk = spk > 0.5
        tr = [trough(vm[k], spk[k], np.flatnonzero(spk[k])[0]) for k in range(len(rows)) if spk[k].any()]
        print(m, lab, "trough %.1f (GT %.1f)  spikes/trace %.1f (GT %.1f)" % (np.nanmean(tr), gt, spk[:, base.SEED_MS:].sum(1).mean(), S[rows][:, base.SEED_MS:].sum(1).mean()), flush=True)
