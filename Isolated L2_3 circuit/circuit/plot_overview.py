"""Overview figure for one isolated-L2/3 condition.

Panels: population rate over time for each L2/3 class (full network, isolated replay and the
rate-matched Poisson control), and the per-cell rate distribution of L2/3 E cells in each run,
with the share of spikes fired by the top 10% of cells (the long-tail readout of step 6).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import sonata_io as sio
from .extract_l23 import classify

CLASSES = ["L2/3 E", "L2/3 Pvalb", "L2/3 Sst", "L2/3 Htr3a"]
CLASS_COLORS = {"L2/3 E": "#2a78d6", "L2/3 Pvalb": "#eb6834", "L2/3 Sst": "#1baf7a", "L2/3 Htr3a": "#eda100"}
RUN_STYLES = [dict(color="#222222", lw=2.0), dict(color="#2a78d6", lw=1.2, ls="--"), dict(color="#eb6834", lw=1.2)]


def load(spikes_h5):
    for pop, ids, ts in sio.iter_spikes(spikes_h5):
        if pop == "l23":
            return ids, ts
    raise ValueError(f"no l23 spikes in {spikes_h5}")


def top_share(rates, frac=0.1):
    r = np.sort(rates)[::-1]
    k = max(1, int(round(frac * len(r))))
    return float(r[:k].sum() / r.sum()) if r.sum() > 0 else float("nan")


def overview(extract_dir, runs, out_png, tstop=3000.0, gray_until=500.0, bin_ms=20.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = os.path.join(extract_dir, "network")
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    cls = nodes["pop_name"].map(classify).to_numpy()
    n = len(nodes)
    data = [(label, *load(path)) for label, path in runs]

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.45, wspace=0.3)
    bins = np.arange(0, tstop + bin_ms, bin_ms)
    stats = {}
    for j, c in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, j])
        m = cls == c
        ax.axvspan(0, gray_until, color="0.92", lw=0)
        for (label, ids, ts), st in zip(data, RUN_STYLES):
            sel = m[ids]
            h, _ = np.histogram(ts[sel], bins=bins)
            ax.plot(bins[:-1] + bin_ms / 2, h / m.sum() / (bin_ms / 1000.0), label=label, **st)
        ax.set_title(f"{c} (n={m.sum():,})", fontsize=10, color=CLASS_COLORS[c], loc="left")
        ax.set_xlabel("time (ms)")
        if j == 0:
            ax.set_ylabel("population rate (Hz)")
            ax.text(gray_until / 2, ax.get_ylim()[1] * 0.92, "gray", ha="center", fontsize=8, color="0.4")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.01))

    ax = fig.add_subplot(gs[1, :2])
    e = cls == "L2/3 E"
    lines = []
    for (label, ids, ts), st in zip(data, RUN_STYLES):
        m = (ts >= gray_until) & (ts < tstop)
        r = np.bincount(ids[m], minlength=n)[:n] / ((tstop - gray_until) / 1000.0)
        re = r[e]
        silent = float((re == 0).mean())
        share = top_share(re)
        stats[label] = {"mean_rate_E_hz": float(re.mean()), "silent_fraction_E": silent, "top10_share_E": share,
                        "median_rate_E_hz": float(np.median(re))}
        rs = np.sort(re[re > 0])
        ccdf = (len(rs) - np.arange(len(rs))) / len(re)  # fraction of all E cells at or above each rate
        ax.step(rs, ccdf, where="post", **st)
        lines.append(f"{label}: mean {re.mean():.2f} Hz, silent {silent:.0%}, top 10% fire {share:.0%}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="L2/3 E rate during grating (Hz)", ylabel="fraction of E cells at or above rate",
           title="L2/3 E rate distribution (500-3000 ms)")
    ax.title.set_fontsize(10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.03, "\n".join(lines), transform=ax.transAxes, va="bottom", ha="left", fontsize=8)

    ax = fig.add_subplot(gs[1, 2:])
    ids, ts = data[0][1], data[0][2]
    rng = np.random.default_rng(0)
    shown = []
    y0 = 0
    for c in CLASSES:
        cell_ids = np.flatnonzero(cls == c)
        k = {"L2/3 E": 160}.get(c, 40)
        pick = np.sort(rng.choice(cell_ids, size=min(k, len(cell_ids)), replace=False))
        lut = np.full(n, -1)
        lut[pick] = np.arange(len(pick)) + y0
        sel = lut[ids] >= 0
        ax.scatter(ts[sel], lut[ids[sel]], s=1.2, color=CLASS_COLORS[c], lw=0)
        shown.append((y0 + len(pick) / 2, c))
        y0 += len(pick) + 8
    ax.set_yticks([p for p, _ in shown], [c for _, c in shown], fontsize=8)
    ax.axvspan(0, gray_until, color="0.92", lw=0, zorder=-1)
    ax.set(xlim=(0, tstop), xlabel="time (ms)", title=f"{data[0][0]}: random L2/3 cells")
    ax.title.set_fontsize(10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.splitext(out_png)[0] + ".json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extract-dir", required=True)
    p.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "SPIKES_H5"), required=True,
                   help="label and l23 spikes file; first is the reference (up to 3)")
    p.add_argument("--out", required=True)
    p.add_argument("--tstop", type=float, default=3000.0)
    p.add_argument("--gray-until", type=float, default=500.0)
    a = p.parse_args(argv)
    print(json.dumps(overview(a.extract_dir, a.run, a.out, a.tstop, a.gray_until), indent=2))


if __name__ == "__main__":
    main()
