"""Step 5a reference check: isolated L2/3 rates vs L2/3 rates in the full network.

Reads the L2/3 reference spikes (from ``simulate streams``) and the isolated run's spikes and
reports, per cell class (L2/3 E, Pvalb, Sst, Htr3a):
  * mean rate in each run and their ratio,
  * per-cell rate correlation,
  * the fraction of spikes that match to within ``--tol`` ms (1.0 = spike-for-spike replay),
  * the KS distance between the two per-cell rate distributions.
Also plots per-cell rates and the rate distributions.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import sonata_io as sio
from .extract_l23 import classify


def per_cell_rates(spikes_h5: str, population: str, n: int, t0: float, t1: float) -> np.ndarray:
    for pop, ids, ts in sio.iter_spikes(spikes_h5):
        if pop == population:
            m = (ts >= t0) & (ts < t1)
            return np.bincount(ids[m], minlength=n)[:n] / ((t1 - t0) / 1000.0)
    return np.zeros(n)


def spikes_of(spikes_h5: str, population: str):
    for pop, ids, ts in sio.iter_spikes(spikes_h5):
        if pop == population:
            return ids, ts
    return np.zeros(0, int), np.zeros(0)


def matched_fraction(ref, iso, n, tol, t0, t1) -> float:
    (ri, rt), (ii, it) = ref, iso
    mr, mi = (rt >= t0) & (rt < t1), (it >= t0) & (it < t1)
    ri, rt, ii, it = ri[mr], rt[mr], ii[mi], it[mi]
    hits = 0
    for c in range(n):
        a = np.sort(rt[ri == c])
        b = np.sort(it[ii == c])
        if len(a) == 0 or len(b) == 0:
            continue
        j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
        d = np.minimum(np.abs(b[j] - a), np.abs(b[np.maximum(j - 1, 0)] - a))
        hits += int((d <= tol).sum())
    total = max(len(rt), len(it))
    return hits / total if total else 1.0


def ks(a, b) -> float:
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic) if len(a) and len(b) else float("nan")


def compare(extract_dir: str, reference_spikes: str, isolated_spikes: str, out_dir: str,
            t0: float = 0.0, t1: float | None = None, tol: float = 0.5, plot: bool = True) -> dict:
    net = os.path.join(extract_dir, "network")
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    n = len(nodes)
    ref, iso = spikes_of(reference_spikes, "l23"), spikes_of(isolated_spikes, "l23")
    if t1 is None:
        t1 = float(max(ref[1].max(initial=0), iso[1].max(initial=0))) + 1e-9
    r_ref = per_cell_rates(reference_spikes, "l23", n, t0, t1)
    r_iso = per_cell_rates(isolated_spikes, "l23", n, t0, t1)
    cls = nodes["pop_name"].map(classify).to_numpy()
    rows = []
    for c in ["all"] + sorted(set(cls)):
        m = np.ones(n, bool) if c == "all" else cls == c
        a, b = r_ref[m], r_iso[m]
        rows.append({
            "class": c, "n": int(m.sum()), "rate_full_hz": float(a.mean()), "rate_isolated_hz": float(b.mean()),
            "ratio": float(b.mean() / a.mean()) if a.mean() > 0 else float("nan"),
            "per_cell_r": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan"),
            "ks_rate_distribution": ks(a, b),
        })
    df = pd.DataFrame(rows)
    res = {"window_ms": [t0, t1], "spike_match_tol_ms": tol,
           "matched_spike_fraction": matched_fraction(ref, iso, n, tol, t0, t1),
           "by_class": df.to_dict(orient="records")}
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "rates_by_class.csv"), index=False)
    pd.DataFrame({"l23_id": np.arange(n), "pop_name": nodes["pop_name"], "class": cls,
                  "rate_full_hz": r_ref, "rate_isolated_hz": r_iso}).to_csv(
        os.path.join(out_dir, "rates_per_cell.csv"), index=False)
    with open(os.path.join(out_dir, "reference_check.json"), "w") as f:
        json.dump(res, f, indent=2)
    if plot:
        _plot(r_ref, r_iso, cls, out_dir)
    return res


def _plot(r_ref, r_iso, cls, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(set(cls))
    colors = {"L2/3 E": "#2a6fdb", "L2/3 Pvalb": "#d1495b", "L2/3 Sst": "#edae49", "L2/3 Htr3a": "#3c9d5d"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    hi = max(r_ref.max(initial=1), r_iso.max(initial=1)) * 1.05
    for c in classes:
        m = cls == c
        ax[0].scatter(r_ref[m], r_iso[m], s=6, alpha=0.6, color=colors.get(c), label=f"{c} (n={m.sum()})")
    ax[0].plot([0, hi], [0, hi], color="0.5", lw=1)
    ax[0].set(xlabel="rate in full network (Hz)", ylabel="rate in isolated L2/3 (Hz)", title="Per-cell rates")
    ax[0].legend(frameon=False, fontsize=8)
    bins = np.logspace(-2, np.log10(hi + 1), 40)
    for c in classes:
        m = cls == c
        ax[1].hist(np.maximum(r_ref[m], 1e-2), bins=bins, histtype="step", color=colors.get(c), lw=1.5)
        ax[1].hist(np.maximum(r_iso[m], 1e-2), bins=bins, histtype="step", color=colors.get(c), lw=1.5, ls="--")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="rate (Hz, 0 shown at 0.01)", ylabel="cells", title="Rate distributions (solid full, dashed isolated)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_check.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract-dir", required=True)
    p.add_argument("--reference-spikes", required=True)
    p.add_argument("--isolated-spikes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float)
    p.add_argument("--tol", type=float, default=0.5)
    a = p.parse_args(argv)
    print(json.dumps(compare(a.extract_dir, a.reference_spikes, a.isolated_spikes, a.out, a.t0, a.t1, a.tol), indent=2))


if __name__ == "__main__":
    main()
