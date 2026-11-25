"""Figures for the Aim 2 controls, from analyze.py's outputs (summary.json, pc_rates_grating.npz, in_rates_grating.npz).

    python plot_results.py --analysis results --out results
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = "#222222", "#6b6b6b", "#dddddd"
COL = {"in vivo": INK, "homogeneous": "#2a78d6", "heterogeneous, shuffled": "#eb6834",
       "heterogeneous, depth-placed": "#1baf7a", "heterogeneous, rheobase-matched": "#6250d6",
       "GLIF network": "#9a9a9a",
       "homogeneous, homogeneous interneurons": "#2a78d6", "heterogeneous, homogeneous interneurons": "#eb6834"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.title.set_fontsize(10)
    ax.grid(color=GRID, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def ccdf(ax, r, label, color, ls="-"):
    x = np.sort(r)
    y = 1 - np.arange(len(x)) / len(x)
    ax.step(x, y, where="post", color=color, lw=2, ls=ls, label=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True); p.add_argument("--out", required=True)
    p.add_argument("--excitability", default=os.path.join(HERE, "results", "models", "excitability.json"))
    a = p.parse_args()
    S = json.load(open(os.path.join(a.analysis, "summary.json")))
    R = dict(np.load(os.path.join(a.analysis, "pc_rates_grating.npz")))
    IN = dict(np.load(os.path.join(a.analysis, "in_rates_grating.npz")))
    C = S["conditions"]
    iv = R["invivo"]

    # ---------- Figure 1: distributions vs in vivo -----------------------------------------------
    main_conds = [c for c in ("homogeneous", "heterogeneous, shuffled", "heterogeneous, depth-placed",
                              "heterogeneous, rheobase-matched") if c in C]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(wspace=0.32, width_ratios=[1.3, 1, 1]))
    ax = axs[0]
    ccdf(ax, iv, f"in vivo, Neuropixels L2/3 RS (n={len(iv)})", INK)
    for c in main_conds:
        ccdf(ax, R[c], c, COL[c], "--" if "depth" in c else "-")
    ax.set(xscale="log", yscale="log", xlim=(0.3, 40), ylim=(1e-4, 1.05), xlabel="rate during grating (Hz)",
           ylabel="fraction of cells with rate ≥ x", title="A  Upper tail of the PC rate distribution")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for ax, key, lab in ((axs[1], "top10_share", "share of spikes fired by the top 10% of cells"),
                         (axs[2], "p99", "99th percentile rate (Hz)")):
        xs = np.arange(len(main_conds) + 1)
        ax.bar(0, S["invivo"][key], 0.6, color=INK)
        for i, c in enumerate(main_conds, 1):
            lo, md, hi = C[c]["at_invivo_n"][key]
            ax.bar(i, C[c][key], 0.6, color=COL[c])
            ax.errorbar(i, md, yerr=[[md - lo], [hi - md]], color=INK, capsize=3, lw=1)
        ax.set_xticks(xs, ["in vivo"] + [{"homogeneous": "homog.", "heterogeneous, shuffled": "het.\nshuffled",
                                         "heterogeneous, depth-placed": "het.\ndepth", "heterogeneous, rheobase-matched": "het.\nrheo-\nmatched"}.get(c, c)
                                        for c in main_conds], fontsize=7.5)
        ax.set(ylabel=lab)
    axs[1].set_title("B  Top-10% spike share")
    axs[2].set_title("C  99th percentile")
    for ax in axs:
        style(ax)
    fig.text(0.5, -0.04, "Error bars: model statistic in random subsamples of the in vivo size (n=%d), 95%% interval. "
             "All PCs (silent included), rates over 0.5-3 s; matched mean PC rate." % len(iv), ha="center", fontsize=8, color=MUTED)
    fig.savefig(os.path.join(a.out, "fig1_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: rank test ----------------------------------------------------------------
    rk = S.get("rank_test", {})
    key = "heterogeneous, shuffled"
    if key in rk:
        T = rk[key]
        per = T["per_model"]
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(wspace=0.32))
        ax = axs[0]
        rh = [q["rheobase_pa"] for q in per]
        ax.scatter(rh, [q["mean_hz"] for q in per], s=[600 * q["share_of_top10_cells"] + 10 for q in per], color=COL[key],
                   edgecolor="white", lw=2, zorder=3)
        for q in per:
            if q["share_of_top10_cells"] >= 0.05:
                ax.annotate(f"{q['model']}: {100 * q['share_of_top10_cells']:.0f}% of top 10%", (q["rheobase_pa"], q["mean_hz"]),
                            xytext=(12, -3), textcoords="offset points", fontsize=7, color=INK)
        low = [q for q in per if q["share_of_top10_cells"] < 0.05]
        ax.text(0.97, 0.22, f"{len(low)} other models:\n<0.5 Hz, ≤1% of top 10%", transform=ax.transAxes, ha="right",
                fontsize=7, color=MUTED)
        ax.set_ylim(-0.3, max(q["mean_hz"] for q in per) * 1.15)
        ax.set(xlabel="model rheobase, 1 s steps (pA)", ylabel="mean rate in the network (Hz)",
               title="A  Rate by model (area = share of top-10% cells)")
        ax = axs[1]
        pops = np.load(os.path.join(HERE, "populations", "assign_shuffled.npy"))
        h, r = R["homogeneous"], R[key]
        jit = np.random.default_rng(0).uniform(-0.2, 0.2, (2, len(h)))
        order = np.argsort(rh)
        for lab, js, col in (("most excitable model (%s)" % per[order[0]]["model"], [order[0]], "#eb6834"),
                             ("other models", list(order[1:]), "#9a9a9a")):
            m = np.isin(pops, js)
            ax.scatter(h[m] + 0.2 + jit[0][m], r[m] + 0.2 + jit[1][m], s=2, alpha=0.25, color=col, lw=0, label=lab)
        ax.set(xscale="log", yscale="log", xlabel="same PC in the homogeneous network + 0.2 (Hz)",
               ylabel="heterogeneous network + 0.2 (Hz)", title="B  Input drive (homogeneous rate) vs heterogeneous rate")
        ax.legend(frameon=False, fontsize=7.5, markerscale=5, loc="upper left")
        ax = axs[2]
        vals = [T["r2_model_identity"], T["r2_input_drive"], T["r2_both"]]
        ax.bar(range(3), vals, 0.6, color=[COL[key], "#2a78d6", INK])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(3), ["model identity\n(intrinsic)", "input drive\n(homog. rate)", "both"], fontsize=8)
        ax.set(ylabel="variance of log rate explained (R²)", ylim=(0, 1), title="C  What sets a PC's rate?")
        for ax in axs:
            style(ax)
        fig.savefig(os.path.join(a.out, "fig2_rank_test.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Figure 3: depth ----------------------------------------------------------------------
    if "heterogeneous, depth-placed" in R and "heterogeneous, shuffled" in R:
        d = R["depth"]
        bins = np.linspace(d.min(), d.max() + 1e-6, 9)
        b = np.digitize(d, bins) - 1
        mid = 0.5 * (bins[1:] + bins[:-1])
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw=dict(wspace=0.32))
        ax = axs[0]
        for c in ("homogeneous", "heterogeneous, shuffled", "heterogeneous, depth-placed"):
            ax.plot(mid, [R[c][b == i].mean() for i in range(8)], "o-", color=COL[c], lw=2, ms=6, label=c)
        ax.set(xlabel="PC soma depth (µm below pia)", ylabel="mean rate (Hz)", title="A  Rate vs depth")
        ax.legend(frameon=False, fontsize=7.5)
        ax = axs[1]
        ie = np.load(os.path.join(a.analysis, "pc_external_current_grating.npy"))
        ax.plot(mid, [ie[b == i].mean() for i in range(8)], "o-", color=INK, lw=2, ms=6)
        ax.set(xlabel="PC soma depth (µm below pia)", ylabel="mean replayed input current (pA)",
               title="B  Input from outside L2/3 vs depth (Allen model)", ylim=(0, 1.25 * ie.mean()))
        ax = axs[2]
        ccdf(ax, iv, "in vivo", INK)
        for c in ("heterogeneous, shuffled", "heterogeneous, depth-placed"):
            ccdf(ax, R[c], c, COL[c], "--" if "depth" in c else "-")
        ax.set(xscale="log", yscale="log", xlim=(0.3, 40), ylim=(1e-4, 1.05), xlabel="rate during grating (Hz)",
               ylabel="fraction of cells with rate ≥ x", title="C  Depth-placed vs shuffled")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")
        for ax in axs:
            style(ax)
        fig.savefig(os.path.join(a.out, "fig3_depth.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Figure 4: interneurons ---------------------------------------------------------------
    pairs = [(c, c + ", homogeneous interneurons") for c in ("homogeneous", "heterogeneous, shuffled")]
    pairs = [(x, y) for x, y in pairs if x.split(",")[0] + ", homogeneous interneurons" in R or y in R]
    pairs = [(x, y if y in R else x.split(",")[0] + ", homogeneous interneurons") for x, y in pairs]
    pairs = [(x, y) for x, y in pairs if x in R and y in R]
    if pairs:
        import h5py
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(wspace=0.32, width_ratios=[1, 1, 1.1]))
        cls = json.load(open(os.path.join(a.analysis, "interneuron_classes.json")))
        names = ["Pvalb", "Sst", "Htr3a"]
        ax = axs[0]
        w = 0.8 / (2 * len(pairs))
        k = 0
        for x, y in pairs:
            for c, hatch in ((x, None), (y, "//")):
                ax.bar(np.arange(3) + (k - len(pairs) + 0.5) * w, [IN[c][np.array(cls[n])].mean() for n in names], w * 0.92,
                       color=COL[c], hatch=hatch, edgecolor="white", label=c)
                k += 1
        ax.set_xticks(range(3), names)
        ax.set(ylabel="mean rate during grating (Hz)", title="A  Interneuron rates (hatched = homogeneous IN)")
        ax.set_ylim(0, 1.45 * max(IN[c][np.array(cls[n])].mean() for xy in pairs for c in xy for n in names))
        ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=1)
        ax = axs[1]
        ccdf(ax, iv, "in vivo", INK)
        for x, y in pairs:
            short = "homog. PCs" if x == "homogeneous" else "het. PCs"
            ccdf(ax, R[x], short + ", diverse IN", COL[x])
            ccdf(ax, R[y], short + ", homog. IN", COL[x], ":")
        ax.set(xscale="log", yscale="log", xlim=(0.3, 40), ylim=(1e-4, 1.05), xlabel="PC rate during grating (Hz)",
               ylabel="fraction of PCs with rate ≥ x", title="B  PC tail with diverse vs homogeneous interneurons")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
        ax = axs[2]
        rows = [c for xy in pairs for c in xy]
        cols = ["mean", "frac_silent", "top10_share", "p99", "max"]
        txt = [[f"{C[c][q]:.2f}" for q in cols] for c in rows]
        ax.axis("off")
        tb = ax.table(cellText=txt, rowLabels=[r.replace("heterogeneous", "het.").replace("homogeneous interneurons", "homog. IN")
                                               for r in rows], colLabels=["mean", "silent", "top10%", "p99", "max"],
                      bbox=[0.38, 0.25, 0.62, 0.5], cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.5)
        ax.set_title("C  PC distribution (Hz; shares as fractions)", fontsize=10)
        for ax in axs[:2]:
            style(ax)
        fig.savefig(os.path.join(a.out, "fig4_interneurons.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Figure 0: the 8 models ---------------------------------------------------------------
    E = json.load(open(a.excitability))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(wspace=0.3))
    amps = np.array(E["amps_pa"])
    mus = np.array(E["ou_mu_pa"])
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for i, (n, m) in enumerate(E["models"].items()):
        axs[0].plot(amps, m["fi_hz"], color=pal[i], lw=2, label=n)
        sg = sorted(m["ou_rates_hz"], key=float)[-1]
        axs[1].plot(mus, m["ou_rates_hz"][sg], color=pal[i], lw=2, label=n)
    axs[0].set(xlabel="1 s step (pA)", ylabel="rate (Hz)", title="A  f-I of the 8 fitted models")
    axs[1].set(xlabel="mean of fluctuating input (pA), SD %s pA, τ 5 ms" % sg, ylabel="rate (Hz)",
               title="B  Rate under network-like fluctuating input")
    axs[0].legend(frameon=False, fontsize=7.5, title="specimen", title_fontsize=8)
    for ax in axs:
        style(ax)
    fig.savefig(os.path.join(a.out, "fig0_models.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
