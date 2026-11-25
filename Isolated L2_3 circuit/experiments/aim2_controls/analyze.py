"""Aim 2 controls: rate distributions, rank test, depth placement and interneuron diversity.

    python analyze.py --runs runs.json --prep PREP --pops populations --invivo npx_l23_rs_rates_all.npy --out results

runs.json maps a condition name to {"spikes": path, "population": "shuffled" | "depth" | ... | null,
"interneurons": "diverse" | "homogeneous"}. Rates are taken over the grating (0.5-3 s), as in the earlier
in vivo comparison (isolated_l23/invivo_comparison/compare.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from circuit import sonata_io as sio  # noqa: E402
from circuit.hybrid import _read_spikes  # noqa: E402

T0, T1 = 500.0, 3000.0
N_L23 = 65949
SILENT = 0.4   # Hz, = one spike in the 2.5 s window


def rates_all(path, t0=T0, t1=T1):
    ids, ts = _read_spikes(path, "l23")
    m = (ts >= t0) & (ts < t1)
    return np.bincount(ids[m], minlength=N_L23)[:N_L23] / ((t1 - t0) / 1000)


def dist_stats(r):
    r = np.asarray(r, float)
    a = np.sort(r)[::-1]
    k = max(1, int(round(0.1 * len(a))))
    act = r[r >= SILENT]
    lg = np.log10(act)
    return dict(n=int(len(r)), mean=float(r.mean()), median=float(np.median(r)), frac_silent=float((r < SILENT).mean()),
                p90=float(np.percentile(r, 90)), p99=float(np.percentile(r, 99)), max=float(r.max()),
                top10_share=float(a[:k].sum() / max(a.sum(), 1e-12)), top1_share=float(a[:max(1, len(a) // 100)].sum() / max(a.sum(), 1e-12)),
                sd_log10_active=float(lg.std()), skew_log10_active=float(((lg - lg.mean()) ** 3).mean() / lg.std() ** 3))


def boot_stats(r, n, reps=2000, seed=0):
    """Model statistics at the in vivo sample size (n cells drawn without replacement), 95% intervals."""
    rng = np.random.default_rng(seed)
    keys = ("top10_share", "p99", "max", "frac_silent", "mean")
    vals = {k: [] for k in keys}
    for _ in range(reps):
        s = dist_stats(rng.choice(r, n, replace=False))
        for k in keys:
            vals[k].append(s[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def r2(y, X):
    X = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return float(1 - e.var() / y.var())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True); p.add_argument("--prep", required=True)
    p.add_argument("--pops", required=True); p.add_argument("--invivo", required=True)
    p.add_argument("--excitability", default=os.path.join(HERE, "results", "models", "excitability.json"))
    p.add_argument("--models", default=os.path.join(HERE, "models.json"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = json.load(open(a.runs))
    spec = json.load(open(a.models))["models"]
    names = [m["name"] for m in spec]
    exc = json.load(open(a.excitability))
    with h5py.File(os.path.join(a.prep, "ids.h5")) as f:
        pc_to_l23, in_to_l23 = f["pc_to_l23"][()], f["in_to_l23"][()]
    depth = np.load(os.path.join(a.pops, "pc_depth_um.npy"))
    invivo = np.load(a.invivo)
    R, IN = {}, {}
    for k, v in runs.items():
        if not os.path.exists(v["spikes"]):
            print("missing", k); continue
        r = rates_all(v["spikes"])
        R[k] = r[pc_to_l23]
        IN[k] = r[in_to_l23]
    res = {"window_ms": [T0, T1], "silent_below_hz": SILENT, "invivo": dist_stats(invivo), "conditions": {}}
    for k, r in R.items():
        s = dist_stats(r)
        s["at_invivo_n"] = boot_stats(r, len(invivo))
        from scipy.stats import ks_2samp
        act, iv = r[r >= SILENT], invivo[invivo >= SILENT]
        s["KS_shape_active_vs_invivo"] = float(ks_2samp(np.log10(act / np.median(act)), np.log10(iv / np.median(iv))).statistic)
        s["interneuron_rate_hz_grating"] = float(IN[k].mean())
        res["conditions"][k] = s

    # ---- rank test (b) and depth (c) per heterogeneous run --------------------------------------
    ou_mu = np.array(exc["ou_mu_pa"])
    ex = exc["models"]
    rheo = np.array([ex[n]["rheobase_pa"] for n in names])
    homog = R.get("homogeneous")
    rank = {}
    from scipy.stats import spearmanr
    for k, v in runs.items():
        if k not in R or not v.get("population") or v["population"] == "homogeneous":
            continue
        asg = np.load(os.path.join(a.pops, f"assign_{'depth' if v['population'] == 'depth' else 'shuffled'}.npy"))
        r = R[k]
        top = r >= np.sort(r)[::-1][int(round(0.1 * len(r))) - 1]
        per = []
        for j, n in enumerate(names):
            m = asg == j
            per.append(dict(model=n, n=int(m.sum()), mean_hz=float(r[m].mean()), frac_silent=float((r[m] < SILENT).mean()),
                            share_of_top10_cells=float(top[m].sum() / top.sum()), share_of_spikes=float(r[m].sum() / r.sum()),
                            rheobase_pa=float(rheo[j])))
        y = np.log10(r + 0.2)
        D = np.eye(len(names))[asg][:, 1:]
        out = dict(per_model=per,
                   spearman_rate_vs_minus_rheobase=float(spearmanr(r, -rheo[asg]).correlation),
                   r2_model_identity=r2(y, D))
        if homog is not None:
            h = np.log10(homog + 0.2)
            out.update(spearman_rate_vs_homogeneous_rate=float(spearmanr(r, homog).correlation),
                       r2_input_drive=r2(y, h[:, None]), r2_both=r2(y, np.c_[D, h]))
            # input drive of the top 10%: their rank in the homogeneous network
            out["top10_cells_homog_percentile_median"] = float(np.median((np.argsort(np.argsort(homog)) / len(homog))[top]))
        out["spearman_rate_vs_depth"] = float(spearmanr(r, depth).correlation)
        rank[k] = out
    res["rank_test"] = rank
    if homog is not None:
        I_ext = np.load(os.path.join(a.prep, "pc_external_current.npy"), mmap_mode="r")
        ie = np.asarray(I_ext[int(T0):int(T1)]).mean(0)
        res["depth_vs_input"] = dict(spearman_external_current_vs_depth=float(spearmanr(ie, depth).correlation),
                                     spearman_homog_rate_vs_depth=float(spearmanr(homog, depth).correlation))
        np.save(os.path.join(a.out, "pc_external_current_grating.npy"), ie.astype(np.float32))
    import pandas as pd
    nn = os.path.join(a.prep, "nest_network")
    with h5py.File(os.path.join(nn, "l23i_nodes.h5")) as f:
        tid = f["nodes/l23i/node_type_id"][()]
    pop = pd.read_csv(os.path.join(nn, "l23i_node_types.csv"), sep=" ").set_index("node_type_id")["pop_name"]
    pn = pop.loc[tid].to_numpy()
    cls = {c: np.flatnonzero(pn == f"i23{c}").tolist() for c in ("Pvalb", "Sst", "Htr3a")}
    json.dump(cls, open(os.path.join(a.out, "interneuron_classes.json"), "w"))
    res["interneuron_rates_by_class"] = {k: {c: float(v[np.array(ix)].mean()) for c, ix in cls.items()} for k, v in IN.items()}
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "pc_rates_grating.npz"), invivo=invivo, depth=depth, **R)
    np.savez_compressed(os.path.join(a.out, "in_rates_grating.npz"), **IN)
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "at_invivo_n"}
                      for k, v in res["conditions"].items()}, indent=1))
    print(json.dumps(res["rank_test"], indent=1)[:4000])
    print(json.dumps(res.get("depth_vs_input"), indent=1))


if __name__ == "__main__":
    main()
