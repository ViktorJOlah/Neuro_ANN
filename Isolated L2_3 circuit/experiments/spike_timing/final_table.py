"""Before/after table over the 5 reliable cells from experiments/real_cells/results/metrics_summary.csv
(run real_cells/code/compare.py first). Writes results_table.md next to this script."""
import csv, os
import numpy as np
import timing_lib as tl

rows = list(csv.DictReader(open(os.path.join(tl.RC, "results", "metrics_summary.csv"))))
get = lambda c, m, p, k: next((float(r[k]) if r.get(k) not in ("", None) else np.nan for r in rows
                               if r["cell"] == c and r["model"] == m and r["protocol"] == p), np.nan)
MODELS = [("standard", "Standard (PR 11)"), ("standard_branches", "branch refit"), ("standard_timing", "**timing refit**"),
          ("branches_v2", "branch refit v2"), ("allactive", "all-active"), ("glif_best", "GLIF best")]
METRICS = [("noise", "gamma", "Noise 2 Γ ↑"), ("noise", "gamma_norm", "Noise 2 Γ / repeat Γ ↑"),
           ("noise", "spike_ev_ratio", "Noise 2 spike-train EV / ceiling ↑"),
           ("noise", "count_bias", "Noise 2 spike-count bias (model − data)"), ("noise", "count_abs_err", "Noise 2 |count error| ↓"),
           ("noise", "vm_ev_sub", "Noise Vm EV away from spikes ↑"), ("noise", "rmse_sub_mV", "Noise Vm RMSE away from spikes (mV) ↓"),
           ("square", "vm_ev_sub", "Square Vm EV away from spikes ↑"), ("square", "ss_err_mV", "Square steady-state error (mV) ↓"),
           ("square", "count_abs_err", "Square |count error| ↓"), ("square", "latency_err_ms", "Square first-spike latency error (ms) ↓")]
out = ["| metric (median of 5 cells) | " + " | ".join(n for _, n in MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
for p, k, name in METRICS:
    out.append(f"| {name} | " + " | ".join(f"{np.nanmedian([get(c, m, p, k) for c in tl.CELLS]):.2f}" for m, _ in MODELS) + " |")
out += ["", "Noise 2 Γ per cell:", "", "| cell | " + " | ".join(n for _, n in MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
for c in tl.CELLS:
    out.append(f"| {c} | " + " | ".join(f"{get(c, m, 'noise', 'gamma'):.2f}" for m, _ in MODELS) + " |")
out += ["", "Noise 2 spike-count bias per sweep, per cell:", "", "| cell | " + " | ".join(n for _, n in MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
for c in tl.CELLS:
    out.append(f"| {c} | " + " | ".join(f"{get(c, m, 'noise', 'count_bias'):+.1f}" for m, _ in MODELS) + " |")
open(os.path.join(tl.HERE, "results_table.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
