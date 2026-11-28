"""Step 2, trial-to-trial variability: Fano factor of L2/3 PC spike counts (1-3 s) over 10 grating trials
(trial 0 = the step 1 run, trials 1-9 from run_trials.sh, each with its own GLIF-network noise seed), for the GLIF
network and the kernel-mode ANN-PC network. In vivo and earlier numbers from isolated_l23/invivo_comparison/fano_stats.json.

    python fano.py  -> results/step2_invivo/fano_kernel.json and fig_step2_fano.png
"""
import json
import os
import sys

import h5py
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from circuit.hybrid import _read_spikes  # noqa: E402

V1 = os.environ.get("V1W", "/home/user/v1")
PREV = "/mnt/project-files/isolated_l23/invivo_comparison/fano_stats.json"
pc = h5py.File(f"{V1}/H/ids.h5")["pc_to_l23"][()]


def counts(path, a=1000, b=3000):
    i, t = _read_spikes(path, "l23"); m = (t >= a) & (t < b)
    return np.bincount(i[m], minlength=65949)[:65949][pc]


files = {"GLIF network": [f"{V1}/runs/l23_reference_spikes.h5"] + [f"{V1}/trials/t{k}/streams/l23_reference_spikes.h5" for k in range(1, 10)],
         "ANN-PC network, kernel layer": [f"{V1}/K/standard_inadj3/spikes.h5"] + [f"{V1}/trials/t{k}/hyb/spikes.h5" for k in range(1, 10)]}
ff, res = {}, dict(fano_median={}, fano_iqr={}, n_cells={}, n_trials={}, silent_per_trial_0_5_3s={}, silent_all_trials={})
for k, fl in files.items():
    C = np.stack([counts(f) for f in fl]); mu = C.mean(0); act = mu >= 1
    ff[k] = C.var(0, ddof=1)[act] / mu[act]
    C25 = np.stack([counts(f, 500, 3000) for f in fl])
    res["fano_median"][k] = float(np.median(ff[k])); res["fano_iqr"][k] = np.percentile(ff[k], [25, 75]).tolist()
    res["n_cells"][k] = int(act.sum()); res["n_trials"][k] = len(fl)
    res["silent_per_trial_0_5_3s"][k] = float((C25 == 0).mean()); res["silent_all_trials"][k] = float((C25 == 0).all(0).mean())
prev = json.load(open(PREV)); res["previous (PR 14 current path)"] = prev
out = os.path.join(HERE, "results", "step2_invivo"); json.dump(res, open(os.path.join(out, "fano_kernel.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("fano_median", "silent_per_trial_0_5_3s", "silent_all_trials")}, indent=1))

fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
bins = np.logspace(-1.5, 1.5, 49)
for k, c in (("GLIF network", "#9a9a9a"), ("ANN-PC network, kernel layer", "#2a6fdb")):
    v = np.clip(ff[k], bins[0], bins[-1])
    ax.hist(v, bins, weights=np.ones(len(v)) / len(v), histtype="step", lw=2, color=c, label=f"{k}: median {np.median(ff[k]):.2f}")
iv = prev["fano_median"]["in vivo (Neuropixels L2/3 RS)"]; lo, hi = prev["fano_iqr"]["in vivo (Neuropixels L2/3 RS)"]
ax.axvspan(lo, hi, color="#222222", alpha=0.12); ax.axvline(iv, color="#222222", lw=2, label=f"in vivo median {iv:.2f} (IQR shaded)")
ax.axvline(1, color="#888", lw=1, ls="--")
ax.set_xscale("log"); ax.set_xlabel("Fano factor (count var / mean, 1-3 s, 10 trials)"); ax.set_ylabel("fraction of cells")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Trial-to-trial variability, L2/3 PCs", loc="left")
fig.savefig(os.path.join(HERE, "results", "fig_step2_fano.png"), dpi=150)
