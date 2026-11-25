"""Summary figure for the population library: fit quality and parameters vs soma depth."""
import os, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
x = pd.read_csv(os.path.join(H, "library", "index.csv"), dtype={"cell": str})
u = x.use == 1
cols = [("noise_gamma", "Noise Γ (±4 ms)"), ("noise_rmse_sub_mV", "Noise subthreshold Vm RMSE (mV)"),
        ("square_count_abs_err", "Square |spike-count error|"), ("rheobase_pA", "Rheobase (pA, recorded)"),
        ("V_th", "Izh V_th (mV)"), ("ln_s_sk", "ln SK scale")]
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for a, (c, lab) in zip(ax.flat, cols):
    a.scatter(x.loc[~u, "depth_norm"], x.loc[~u, c], s=18, c="0.75", label="flagged")
    a.scatter(x.loc[u, "depth_norm"], x.loc[u, c], s=22, c="C0", label="use = 1")
    a.set_xlabel("normalized soma depth (0 = pia)"); a.set_ylabel(lab)
ax[0, 0].legend(frameon=False)
fig.suptitle(f"Standard fits on {len(x)} Allen VISp L2/3 PCs ({u.sum()} usable)")
fig.tight_layout(); fig.savefig(os.path.join(H, "summary_vs_depth.png"), dpi=130)
