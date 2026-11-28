"""Step 2 of the kernel-layer rerun: L2/3 PC rate distribution of the hybrid circuit (kernel synapses) vs in vivo.

Same statistics as the 2026-11-23 in vivo check (isolated_l23/invivo_comparison): 595 Neuropixels VISp L2/3 RS units
(Allen Visual Coding, drifting gratings), and every model PC's rate over 0.5-3 s of the grating run.

    python invivo_compare.py --ids H/ids.h5 --invivo npx_l23_rs_rates_all.npy --run "label=spikes.h5" ... --out DIR
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from circuit import sonata_io as sio  # noqa: E402


def model_rates(path, pc, t0=500, t1=3000):
    (_, i, t), = list(sio.iter_spikes(path))
    m = (t >= t0) & (t < t1)
    return (np.bincount(i[m], minlength=pc.max() + 1) / ((t1 - t0) / 1000))[pc]


def stats(r):
    a = np.sort(r)[::-1]
    k = max(1, int(round(0.1 * len(a))))
    act = a[a >= 0.4]
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)), frac_below_0_4Hz=float((a < 0.4).mean()),
                top10_share=float(a[:k].sum() / a.sum()), p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                max=float(a.max()), sd_log10_active=float(np.log10(act).std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True); ap.add_argument("--invivo", required=True)
    ap.add_argument("--run", action="append", required=True, help='"label=spikes.h5"')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pc = h5py.File(a.ids)["pc_to_l23"][()]
    vivo = np.load(a.invivo)
    runs = {"in vivo (Neuropixels L2/3 RS)": vivo}
    for r in a.run:
        lab, path = r.split("=", 1)
        runs[lab] = model_rates(path, pc)
    out = {}
    for k, r in runs.items():
        s = stats(r)
        if k != list(runs)[0]:
            # matched mean: scale model rates to the in vivo mean, a pure shape comparison
            rs = r * vivo.mean() / r.mean()
            s["matched_mean"] = stats(rs)
            act, av = r[r >= 0.4], vivo[vivo >= 0.4]
            s["KS_shape_active_vs_invivo"] = float(ks_2samp(np.log10(act / np.median(act)), np.log10(av / np.median(av))).statistic)
        out[k] = s
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "invivo_stats.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "rates.npz"), **{k.split(" (")[0].replace(" ", "_"): v for k, v in runs.items()})
    for k, s in out.items():
        print(f"{k:40s} mean {s['mean']:.2f} top10 {s['top10_share']:.2f} p99 {s['p99']:.1f} max {s['max']:.1f} <0.4Hz {s['frac_below_0_4Hz']:.2f}",
              "" if "matched_mean" not in s else f"| matched: top10 {s['matched_mean']['top10_share']:.2f} p99 {s['matched_mean']['p99']:.1f} max {s['matched_mean']['max']:.1f}  KS {s['KS_shape_active_vs_invivo']:.2f}")


if __name__ == "__main__":
    main()
