"""Why the hybrid circuit's interneurons fired too little: open-loop replay tests.

Each test feeds a fixed set of PC spikes into the GLIF interneurons (``hybrid run --replay-pc``,
no feedback from the interneurons to the PCs) with a chosen gain on the PC -> interneuron
synapses, and compares each interneuron class's rate with the GLIF network. This separates
the gain on PC -> interneuron synapses from the PC spike pattern itself.

    python -m circuit.diagnose_interneurons --extract-dir X --reference R/streams/l23_reference_spikes.h5 \\
        --test "GLIF PC spikes, PC→IN 0.82" O/glif_g082/spikes.h5 ... --out results/interneuron_rate_fix
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import sonata_io as sio
from .extract_l23 import classify
from .plot_overview import CLASSES, CLASS_COLORS


def class_rates(path, cls, t1):
    for pop, ids, ts in sio.iter_spikes(path):
        if pop == "l23":
            break
    m = ts < t1
    r = np.bincount(ids[m].astype(np.int64), minlength=len(cls))[:len(cls)] / (t1 / 1000)
    return {c: float(r[cls == c].mean()) for c in CLASSES}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract-dir", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--test", nargs=2, action="append", metavar=("LABEL", "SPIKES_H5"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tstop", type=float, default=3000.0)
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net = os.path.join(a.extract_dir, "network")
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    cls = nodes.sort_values("node_id")["pop_name"].map(classify).to_numpy()
    ref = class_rates(a.reference, cls, a.tstop)
    res = {"GLIF network": ref}
    for label, path in a.test:
        res[label] = class_rates(path, cls, a.tstop)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "open_loop_tests.json"), "w") as f:
        json.dump(res, f, indent=2)

    ins = CLASSES[1:]
    tests = [lb for lb, _ in a.test]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    wd = 0.8 / len(ins)
    y = np.arange(len(tests))[::-1]
    for j, c in enumerate(ins):
        v = [100 * (res[t][c] / ref[c] - 1) for t in tests]
        ax.barh(y + (1 - j) * wd, v, wd, color=CLASS_COLORS[c], label=c.replace("L2/3 ", ""))
        for yi, vi in zip(y + (1 - j) * wd, v):
            ax.text(vi - 1 if vi < 0 else vi + 1, yi, f"{vi:+.0f}%", va="center", ha="right" if vi < 0 else "left", fontsize=7)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_yticks(y, tests, fontsize=8)
    ax.set(xlabel=f"interneuron rate vs GLIF network, 0-{a.tstop:.0f} ms (%)",
           title="Open loop: fixed PC spikes into the GLIF interneurons")
    ax.title.set_fontsize(10)
    ax.set_xlim(-50, 10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(a.out, "open_loop_tests.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
