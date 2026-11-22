"""NEURON reference runs of Allen model 488462783 for the conductance-branch layer.

For selected traces of the existing dataset, replays the same 1 ms current (linear
interpolation, dt as in the dataset) and records at 1 ms on soma(0.5):
  v (recording convention, +14 mV), NaTs ina, Nap ina, SK ik, Ca_HVA ica, Ca_LVA ica, cai, SK z.
Conditions: control, and SK blocked (gbar_SK = 0 on the soma; mimics apamin).
Usage: python neuron_reference.py --model /path/to/compiled/model --out DIR --which test|train100
"""
import argparse, os, sys, time, json
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "simulations", "code"))
VARS = ("v", "ina_NaTs", "ina_Nap", "ik_SK", "ica_Ca_HVA", "ica_Ca_LVA", "cai", "z_SK")
_r = None


def _init(model_dir, dt):
    global _r
    from neuron import h
    from allen_cell import Runner
    _r = Runner(model_dir, dt=dt)
    seg = h.soma[0](0.5)
    refs = dict(ina_NaTs=seg.NaTs._ref_ina, ina_Nap=seg.Nap._ref_ina, ik_SK=seg.SK._ref_ik,
                ica_Ca_HVA=seg.Ca_HVA._ref_ica, ica_Ca_LVA=seg.Ca_LVA._ref_ica, cai=seg._ref_cai,
                z_SK=seg.SK._ref_z)
    _r.recs = {}
    for k, ref in refs.items():
        vec = h.Vector(); vec.record(ref, 1.0); _r.recs[k] = vec
    _r.area_um2 = seg.area()
    _r.gsk = seg.SK.gbar


def _sim(task):
    from neuron import h
    proto, idx, cond, i_pA = task
    h.soma[0](0.5).SK.gbar = 0.0 if cond == "sk_block" else _r.gsk
    v = _r.run(i_pA, 1.0)[: len(i_pA)]
    out = dict(v=v.astype(np.float32))
    for k, vec in _r.recs.items():
        out[k] = np.array(vec)[: len(i_pA)].astype(np.float32)
    return proto, idx, cond, out, _r.area_um2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="test")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(ROOT, "experiments", "full_spiking"))
    import common as fs
    sets = fs.load_split()
    conds = ("control", "sk_block") if a.which == "test" else ("control",)
    split = "test" if a.which == "test" else "train"
    tasks = []
    sel = {}
    rng = np.random.default_rng(7)
    for p in fs.PROTOS:
        s = sets[(p, split)]
        rows = np.flatnonzero(s["n_spikes"] > 0)
        if a.which == "train100":
            rows = np.sort(rng.choice(rows, 100, replace=False))
        if a.limit:
            rows = rows[: a.limit]
        sel[p] = rows
        for c in conds:
            for r in rows:
                tasks.append((p, int(r), c, s["I"][r].astype(float)))
    print(len(tasks), "simulations", flush=True)
    res = {}
    t0 = time.time()
    area = None
    with mp.get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.model, a.dt)) as pool:
        for n, (p, r, c, out, area) in enumerate(pool.imap_unordered(_sim, tasks, chunksize=2)):
            res[(p, r, c)] = out
            if n % 50 == 0:
                print(n, "%.0fs" % (time.time() - t0), flush=True)
    os.makedirs(a.out, exist_ok=True)
    for p in fs.PROTOS:
        d = dict(rows=sel[p], ids=sets[(p, split)]["ids"][sel[p]])
        for c in conds:
            for k in VARS:
                d[f"{c}_{k}"] = np.stack([res[(p, int(r), c)][k] for r in sel[p]])
        np.savez_compressed(os.path.join(a.out, f"{a.which}_{p}.npz"), **d)
    json.dump(dict(soma_area_um2=area, split=split, conds=conds, dt=a.dt),
              open(os.path.join(a.out, f"{a.which}_meta.json"), "w"))
    print("total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
