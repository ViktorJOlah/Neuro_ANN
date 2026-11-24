"""Score the fitted Standard ANN against Allen's GLIF and all-active NEURON models on held-out recordings.

Run from "Isolated L2_3 circuit/":
    python experiments/real_cells/code/compare.py

Reads data/<cell>/ and results/<cell>/<model>/predictions.npz; writes results/metrics_per_sweep.csv,
results/metrics_summary.csv, results/summary.json and the figures in results/figures/.

Everything is scored from 450 ms after the start of each cropped test sweep (the Standard's recorded-Vm seed
covers only the pre-stimulus baseline, so it gets no information about the response).

Spike metrics (data spike times at 50 kHz; model spike times as the model gives them):
  gamma              coincidence factor Γ, ±4 ms (Kistler 1997; 1 = perfect, 0 = chance)
  gamma_norm         (noise only) Γ / the cell's own repeat-to-repeat Γ over the identical Noise 2 repeats
  spike_ev           (noise only) explained variance of the data PSTH (all Noise 2 repeats, Gaussian σ = 10 ms)
                     by the model's spike train, 1 - var(PSTH - model)/var(PSTH)
  spike_ev_ratio     spike_ev / the mean explained variance of one repeat by the PSTH of the other repeats
                     (the ceiling a deterministic model can reach; this is the form of Allen's GLIF score)
  count_abs_err, count_bias   spikes per sweep, model - data
  vr_dist            van Rossum distance (τ = 10 ms) per data spike
  latency_err_ms     |first-spike latency error| on Long Square steps where both spike
  isi_mean_rel_err, isi_cv_err   on sweeps where both have >= 3 spikes
Voltage metrics (1 ms):
  vm_ev_sub, rmse_sub_mV      Vm explained variance and RMSE away from spikes (5 ms before to 30 ms after
                              every data or model spike removed)
  rmse_clip_mV                RMSE with both traces clipped at -20 mV
  ss_err_mV                   (Long Square, no spikes in the data) steady-state Vm error over the last 300 ms
                              of the step: input resistance and sag
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
RES = os.path.join(RC, "results")
CELLS = ("513531471", "485468180", "502269786", "318808427", "521409057", "382982932")
SEED = 450
WIN = 4.0
SIG = 10.0
CLIP = -20.0
MODELS = ("standard", "allactive", "glif1_LIF", "glif2_LIF-R", "glif3_LIF-ASC", "glif4_LIF-R-ASC",
          "glif5_LIF-R-ASC-A", "standard_unfitted", "standard_timing", "standard_branches", "branches_v2")
MAIN = ("standard", "standard_timing", "allactive", "glif_best", "glif3_LIF-ASC")
LABEL = {"standard": "Standard ANN (fitted)", "allactive": "Allen all-active NEURON", "glif_best": "Allen GLIF (best level)",
         "glif3_LIF-ASC": "Allen GLIF3 (LIF-ASC)", "standard_unfitted": "Standard ANN (not fitted)",
         "standard_timing": "Standard ANN (timing refit)", "standard_branches": "Standard ANN (branch refit only)",
         "branches_v2": "Standard ANN (branch refit v2)"}
COL = {"standard": "#d95f02", "allactive": "#1b9e77", "glif_best": "#7570b3", "glif3_LIF-ASC": "#a6a2d6",
       "standard_unfitted": "#bbbbbb", "standard_timing": "#e7298a", "standard_branches": "#f4a6cf", "branches_v2": "#c994c7", "data": "k"}


# ------------------------------------------------------------------ metrics
def gamma(td, tm, T):
    nd, nm = len(td), len(tm)
    if nd == 0 and nm == 0:
        return np.nan
    if nd == 0 or nm == 0:
        return 0.0
    used, co = np.zeros(nm, bool), 0
    for t in td:
        k = np.flatnonzero(~used & (np.abs(tm - t) <= WIN))
        if len(k):
            used[k[np.argmin(np.abs(tm[k] - t))]] = True
            co += 1
    nu = nm / T
    norm = 1 - 2 * nu * WIN
    return float((co - 2 * nu * WIN * nd) / (0.5 * (nd + nm)) / norm)


def rate(t, T, sig=SIG):
    r = np.zeros(T)
    for s in t:
        if 0 <= s < T:
            r[int(s)] += 1
    k = np.arange(-int(4 * sig), int(4 * sig) + 1)
    g = np.exp(-0.5 * (k / sig) ** 2)
    return np.convolve(r, g / g.sum(), "same") * 1000.0


def ev(pred, ref):
    v = np.var(ref)
    return float(1 - np.var(pred - ref) / v) if v > 0 else np.nan


def van_rossum(td, tm, T, tau=10.0):
    def filt(t):
        x = np.zeros(T)
        for s in t:
            if 0 <= s < T:
                x[int(s)] += 1
        k = np.exp(-np.arange(int(6 * tau)) / tau)
        return np.convolve(x, k)[:T]
    d = np.sqrt(np.sum((filt(td) - filt(tm)) ** 2) / tau)
    return float(d / max(len(td), 1))


def away_mask(T, *spike_lists):
    m = np.ones(T, bool)
    for sl in spike_lists:
        for s in sl:
            s = int(s)
            m[max(0, s - 5):s + 30] = False
    return m


def isi_stats(t):
    if len(t) < 3:
        return None
    i = np.diff(t)
    return i.mean(), i.std() / i.mean()


def sweep_metrics(r, V, td, vm, tm):
    T = len(V)
    td, tm = td[td >= SEED] - SEED, tm[tm >= SEED] - SEED
    V, vm = V[SEED:], vm[SEED:T]
    T = len(V)
    out = dict(gamma=gamma(td, tm, T), n_data=len(td), n_model=len(tm), count_err=len(tm) - len(td),
               vr_dist=van_rossum(td, tm, T))
    m = away_mask(T, td, tm)
    if m.sum() > 100:
        out["vm_ev_sub"] = ev(vm[m], V[m])
        out["rmse_sub_mV"] = float(np.sqrt(np.mean((vm[m] - V[m]) ** 2)))
    out["rmse_clip_mV"] = float(np.sqrt(np.mean((np.minimum(vm, CLIP) - np.minimum(V, CLIP)) ** 2)))
    if r["stimulus"] == "Long Square":
        on = r["stim_on_ms"] - SEED
        if len(td) and len(tm):
            out["latency_err_ms"] = float(abs(tm[0] - td[0]))
        if len(td) == 0 and r["stim_off_ms"] - r["stim_on_ms"] > 500:
            a, b = r["stim_off_ms"] - 300 - SEED, r["stim_off_ms"] - SEED
            out["ss_err_mV"] = float(abs(vm[a:b].mean() - V[a:b].mean()))
        out["onset"] = on
    sd, sm_ = isi_stats(td), isi_stats(tm)
    if sd and sm_:
        out["isi_mean_rel_err"] = float(abs(sm_[0] - sd[0]) / sd[0])
        out["isi_cv_err"] = float(abs(sm_[1] - sd[1]))
    return out


# ------------------------------------------------------------------ load
def load_cell(cell):
    meta = json.load(open(os.path.join(RC, "data", cell, "dataset.json")))
    z = np.load(os.path.join(RC, "data", cell, "sweeps_1ms.npz"))
    test = [r for r in meta["sweeps"] if r["split"] == "test"]
    data = {r["sweep"]: dict(I=z[f"s{r['sweep']}_I"], V=z[f"s{r['sweep']}_V"], spk=z[f"s{r['sweep']}_spk_ms"]) for r in test}
    preds = {}
    for m in MODELS:
        f = os.path.join(RES, cell, m, "predictions.npz")
        if os.path.exists(f):
            p = np.load(f)
            preds[m] = {r["sweep"]: dict(vm=p[f"s{r['sweep']}_vm"], spk=np.asarray(p[f"s{r['sweep']}_spk"], float)) for r in test}
    return meta, test, data, preds


def noise_metrics(test, data, preds):
    """Γ normalised by the repeat-to-repeat Γ, and the spike-train explained variance, over Noise 2 repeats."""
    rows = [r for r in test if r["stimulus"] == "Noise 2"]
    if len(rows) < 2:
        return {}, {}
    T = min(len(data[r["sweep"]]["V"]) for r in rows) - SEED
    tr = [data[r["sweep"]]["spk"] for r in rows]
    tr = [t[(t >= SEED) & (t < SEED + T)] - SEED for t in tr]
    gi = [gamma(tr[i], tr[j], T) for i in range(len(tr)) for j in range(len(tr)) if i != j]
    rates = np.array([rate(t, T) for t in tr])
    psth = rates.mean(0)
    ev_int = np.nanmean([ev(rates[np.arange(len(tr)) != i].mean(0), rates[i]) for i in range(len(tr))])
    intrinsic = dict(gamma_repeat=float(np.nanmean(gi)), spike_ev_repeat=float(ev_int),
                     n_repeats=len(tr), stim_identical=bool(all(np.allclose(data[r["sweep"]]["I"][:T], data[rows[0]["sweep"]]["I"][:T], atol=1.0) for r in rows)))
    out = {}
    for m, P in preds.items():
        tm = P[rows[0]["sweep"]]["spk"]
        tm = tm[(tm >= SEED) & (tm < SEED + T)] - SEED
        e = ev(rate(tm, T), psth)
        g = np.nanmean([gamma(t, tm, T) for t in tr])
        out[m] = dict(spike_ev=e, spike_ev_ratio=e / ev_int if ev_int > 0 else np.nan,
                      gamma_norm=g / intrinsic["gamma_repeat"] if intrinsic["gamma_repeat"] > 0 else np.nan)
    return intrinsic, out


# ------------------------------------------------------------------ main
def main():
    fig_dir = os.path.join(RES, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    per_sweep, summary, intr = [], {}, {}
    for cell in CELLS:
        if not os.path.exists(os.path.join(RC, "data", cell, "dataset.json")):
            continue
        meta, test, data, preds = load_cell(cell)
        intrinsic, nm = noise_metrics(test, data, preds)
        intr[cell] = intrinsic
        for m, P in preds.items():
            for r in test:
                d = data[r["sweep"]]
                x = sweep_metrics(r, d["V"], d["spk"], P[r["sweep"]]["vm"], P[r["sweep"]]["spk"])
                per_sweep.append(dict(cell=cell, model=m, sweep=r["sweep"], stimulus=r["stimulus"], protocol=r["protocol"],
                                      amp_pA=round(r["amp_pA"], 1), **x))
            summary.setdefault(cell, {})[m] = dict(noise_extra=nm.get(m, {}))
    # aggregate
    keys = ("gamma", "count_abs_err", "count_bias", "vr_dist", "vm_ev_sub", "rmse_sub_mV", "rmse_clip_mV",
            "latency_err_ms", "isi_mean_rel_err", "isi_cv_err", "ss_err_mV")
    agg = []
    for cell in summary:
        for m in summary[cell]:
            for p in ("square", "noise", "ramp", "all"):
                rr = [x for x in per_sweep if x["cell"] == cell and x["model"] == m and (p == "all" or x["protocol"] == p)]
                row = dict(cell=cell, model=m, protocol=p, n_sweeps=len(rr))
                for k in keys:
                    if k == "count_abs_err":
                        v = [abs(x["count_err"]) for x in rr]
                    elif k == "count_bias":
                        v = [x["count_err"] for x in rr]
                    elif k == "gamma":
                        v = [x["gamma"] for x in rr if x["n_data"] > 0 or x["n_model"] > 0]
                    else:
                        v = [x[k] for x in rr if k in x and x[k] is not None and np.isfinite(x[k])]
                    row[k] = float(np.nanmean(v)) if len(v) else np.nan
                if p == "noise":
                    row.update(summary[cell][m]["noise_extra"])
                agg.append(row)
    # GLIF best level per cell: highest noise Γ (generous to GLIF: chosen on the test sweeps)
    for cell in summary:
        gl = [a for a in agg if a["cell"] == cell and a["model"].startswith("glif") and a["protocol"] == "noise"]
        if not gl:
            continue
        best = max(gl, key=lambda a: a["gamma"])["model"]
        for a in [a for a in agg if a["cell"] == cell and a["model"] == best]:
            agg.append(dict(a, model="glif_best", glif_level=best))
    with open(os.path.join(RES, "metrics_per_sweep.csv"), "w", newline="") as f:
        fields = sorted({k for x in per_sweep for k in x}, key=lambda k: (k not in ("cell", "model", "sweep", "stimulus", "protocol", "amp_pA"), k))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_sweep)
    with open(os.path.join(RES, "metrics_summary.csv"), "w", newline="") as f:
        fields = sorted({k for x in agg for k in x}, key=lambda k: (k not in ("cell", "model", "protocol", "n_sweeps"), k))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(agg)
    json.dump(dict(intrinsic=intr, summary=agg), open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
    plots(agg, per_sweep, intr, fig_dir)
    print_table(agg)


def get(agg, cell, m, p, k):
    for a in agg:
        if a["cell"] == cell and a["model"] == m and a["protocol"] == p:
            return a.get(k, np.nan)
    return np.nan


def print_table(agg):
    cells = sorted({a["cell"] for a in agg})
    rows = (("noise", "gamma"), ("noise", "gamma_norm"), ("noise", "spike_ev_ratio"), ("square", "gamma"), ("ramp", "gamma"),
            ("all", "count_abs_err"), ("all", "vm_ev_sub"), ("all", "rmse_sub_mV"), ("all", "rmse_clip_mV"),
            ("square", "latency_err_ms"), ("all", "isi_mean_rel_err"), ("square", "ss_err_mV"), ("all", "vr_dist"))
    for p, k in rows:
        line = f"{p:6s} {k:18s}"
        for m in MAIN + ("standard_unfitted",):
            v = [get(agg, c, m, p, k) for c in cells]
            line += f" | {m[:12]:12s} {np.nanmedian(v):7.3f}"
        print(line)


def plots(agg, per_sweep, intr, fig_dir):
    cells = sorted({a["cell"] for a in agg})
    panels = (("noise", "gamma", "Noise: Γ (±4 ms)", True), ("noise", "spike_ev_ratio", "Noise: spike-train EV / repeat ceiling", True),
              ("square", "gamma", "Long square: Γ", True), ("ramp", "gamma", "Ramp: Γ", True),
              ("all", "count_abs_err", "|spike-count error| per sweep", False), ("square", "latency_err_ms", "first-spike latency error (ms)", False),
              ("all", "isi_mean_rel_err", "mean ISI relative error", False), ("all", "vm_ev_sub", "Vm explained variance away from spikes", True),
              ("all", "rmse_sub_mV", "Vm RMSE away from spikes (mV)", False), ("all", "rmse_clip_mV", "Vm RMSE, clipped at -20 mV (mV)", False),
              ("square", "ss_err_mV", "subthreshold steady-state error (mV)", False), ("all", "vr_dist", "van Rossum distance per spike", False))
    fig, ax = plt.subplots(3, 4, figsize=(18, 11))
    rng = np.random.default_rng(0)
    for A, (p, k, title, higher) in zip(ax.flat, panels):
        for j, m in enumerate(MAIN):
            v = np.array([get(agg, c, m, p, k) for c in cells], float)
            A.bar(j, np.nanmedian(v), color=COL[m], alpha=0.35, width=0.7)
            A.scatter(j + rng.uniform(-0.15, 0.15, len(v)), v, color=COL[m], s=18, zorder=3)
        for i, c in enumerate(cells):
            A.plot(range(len(MAIN)), [get(agg, c, m, p, k) for m in MAIN], color="0.8", lw=0.6, zorder=1)
        if k == "gamma" and p == "noise":
            rep = [intr[c].get("gamma_repeat", np.nan) for c in cells if intr.get(c)]
            A.axhline(np.nanmedian(rep), color="k", ls="--", lw=0.8, label="cell's own repeat Γ")
            A.legend(fontsize=7)
        A.set_xticks(range(len(MAIN)))
        A.set_xticklabels(["Standard\n(PR 11)", "Standard\n(timing)", "all-active", "GLIF best", "GLIF3"], fontsize=7)
        A.set_title(title + ("  (higher = better)" if higher else "  (lower = better)"), fontsize=9)
    fig.suptitle("Held-out Allen recordings, 6 VISp L2/3 pyramidal cells: fitted Standard ANN vs Allen all-active NEURON and GLIF "
                 "(bars = median, dots = cells, grey lines join a cell)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "model_comparison.png"), dpi=120)
    plt.close(fig)

    # per-cell example traces and Noise 2 rasters
    for cell in cells:
        meta, test, data, preds = load_cell(cell)
        show = [m for m in ("standard", "allactive", "glif_best") if m in preds or m == "glif_best"]
        best = next((a.get("glif_level") for a in agg if a["cell"] == cell and a["model"] == "glif_best"), None)
        mods = [("standard", "standard"), ("standard_timing", "standard_timing"), ("allactive", "allactive"), (best, "glif_best")]
        mods = [(m, lab) for m, lab in mods if m in preds]
        ls = [r for r in test if r["stimulus"] == "Long Square" and r["n_spikes"] > 0]
        ls = ls[len(ls) // 2] if ls else None
        nz = next((r for r in test if r["stimulus"] == "Noise 2"), None)
        rp = next((r for r in test if r["protocol"] == "ramp"), None)
        cols = [r for r in (ls, nz, rp) if r is not None]
        fig, ax = plt.subplots(len(mods) + 1, len(cols), figsize=(6 * len(cols), 2.3 * (len(mods) + 1)), squeeze=False)
        for j, r in enumerate(cols):
            d = data[r["sweep"]]
            t = np.arange(len(d["V"])) / 1000.0
            lo, hi = (0, len(t)) if r["protocol"] != "noise" else (r["stim_on_ms"] + 2000, r["stim_on_ms"] + 5000)
            ax[0, j].plot(t[lo:hi], d["I"][lo:hi], "k", lw=0.6)
            ax[0, j].set_title(f"{r['stimulus']} sweep {r['sweep']} ({r['amp_pA']:.0f} pA): injected current (pA)", fontsize=9)
            for i, (m, lab) in enumerate(mods, start=1):
                A = ax[i, j]
                A.plot(t[lo:hi], d["V"][lo:hi], "k", lw=0.6, label="recording")
                A.plot(t[lo:hi], preds[m][r["sweep"]]["vm"][lo:hi], color=COL[lab], lw=0.6, alpha=0.9, label=LABEL[lab] if lab != "glif_best" else f"Allen {m}")
                x = sweep_metrics(r, d["V"], d["spk"], preds[m][r["sweep"]]["vm"], preds[m][r["sweep"]]["spk"])
                A.set_title(f"spikes {x['n_data']} → {x['n_model']}   Γ {x['gamma']:.2f}", fontsize=8)
                A.set_ylim(-100, 50)
                if j == 0:
                    A.legend(fontsize=7, loc="upper left")
        for A in ax[-1]:
            A.set_xlabel("time (s)")
        fig.suptitle(f"cell {cell}: held-out sweeps (noise column: 3 s zoom)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"traces_{cell}.png"), dpi=100)
        plt.close(fig)

    # rasters
    fig, ax = plt.subplots(len(cells), 1, figsize=(16, 2.6 * len(cells)), squeeze=False)
    for A, cell in zip(ax[:, 0], cells):
        meta, test, data, preds = load_cell(cell)
        best = next((a.get("glif_level") for a in agg if a["cell"] == cell and a["model"] == "glif_best"), None)
        rows = [r for r in test if r["stimulus"] == "Noise 2"]
        y, labels = 0, []
        for i, r in enumerate(rows):
            A.vlines(data[r["sweep"]]["spk"] / 1000, y, y + 0.8, color="k", lw=0.6)
            y += 1
            labels.append(f"recording {i+1}")
        for m, lab, name in (("standard", "standard", "Standard"), ("standard_timing", "standard_timing", "Standard timing"), ("allactive", "allactive", "all-active"),
                             (best, "glif_best", f"GLIF ({(best or '')[:5]})")):
            if m in preds and rows:
                A.vlines(preds[m][rows[0]["sweep"]]["spk"] / 1000, y, y + 0.8, color=COL[lab], lw=0.6)
                y += 1
                labels.append(name)
        A.set_yticks(np.arange(y) + 0.4)
        A.set_yticklabels(labels, fontsize=7)
        A.set_title(f"cell {cell}: Noise 2 (identical repeats)", fontsize=9)
    ax[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "noise_rasters.png"), dpi=100)
    plt.close(fig)

    # f-I on the test long squares
    fig, ax = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 3.2), squeeze=False)
    for A, cell in zip(ax[0], cells):
        best = next((a.get("glif_level") for a in agg if a["cell"] == cell and a["model"] == "glif_best"), None)
        rr = sorted([x for x in per_sweep if x["cell"] == cell and x["stimulus"] == "Long Square" and x["model"] == "standard"], key=lambda x: x["amp_pA"])
        A.plot([x["amp_pA"] for x in rr], [x["n_data"] for x in rr], "k-o", ms=3, label="recording")
        for m, lab in (("standard", "standard"), ("standard_timing", "standard_timing"), ("allactive", "allactive"), (best, "glif_best")):
            q = sorted([x for x in per_sweep if x["cell"] == cell and x["stimulus"] == "Long Square" and x["model"] == m], key=lambda x: x["amp_pA"])
            A.plot([x["amp_pA"] for x in q], [x["n_model"] for x in q], "-o", ms=3, color=COL[lab], label=LABEL[lab] if lab != "glif_best" else "GLIF best")
        A.set_title(cell, fontsize=9)
        A.set_xlabel("step (pA)")
    ax[0, 0].set_ylabel("spikes in 1 s")
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fI_test_steps.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
