"""Figure: L2/3 Fano and noise correlation, in vivo (all / still trials) vs model (Poisson LGN / measured LGN state drive)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
iv = json.load(open(f"{H}/results/invivo/fano_invivo.json"))["VISp L2/3"]
nc = json.load(open(f"{H}/results/invivo/noisecorr_invivo.json"))
m = json.load(open(f"{H}/results/model_fano_state_756029989.json"))["results"]
lab = ["in vivo\nall", "in vivo\nstill", "GLIF\nPoisson", "GLIF\nstate", "ANN-PC\nPoisson", "ANN-PC\nstate"]
fano = [iv["fano_all"]["median"], iv["fano_still"]["median"]] + [m[k]["fano_median"] for k in
        ("GLIF, Poisson LGN", "GLIF, LGN state drive", "ANN-PC, Poisson LGN", "ANN-PC, LGN state drive")]
ncs = [nc["all"]["mean"], nc["still"]["mean"]] + [m[k]["noise_corr_mean"] for k in
       ("GLIF, Poisson LGN", "GLIF, LGN state drive", "ANN-PC, Poisson LGN", "ANN-PC, LGN state drive")]
col = ["#222222", "#666666", "#bbbbbb", "#9a9a9a", "#9cc0f0", "#2a6fdb"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for a, v, t in ((ax[0], fano, "Fano factor (median, 2 s counts)"), (ax[1], ncs, "noise correlation (mean pair)")):
    a.bar(range(6), v, color=col); a.set_xticks(range(6)); a.set_xticklabels(lab, fontsize=8); a.set_title(t, loc="left", fontsize=10)
    for i, x in enumerate(v): a.text(i, x, f"{x:.2f}", ha="center", va="bottom", fontsize=8)
    a.spines[["top", "right"]].set_visible(False)
ax[0].axhline(1, color="#888", lw=1, ls="--")
fig.suptitle("L2/3 PCs: measured LGN state drive (session 756029989, gain SD 0.16) vs in vivo", x=0.01, ha="left", fontsize=10)
fig.savefig(f"{H}/results/fig_fano_state.png", dpi=150)
