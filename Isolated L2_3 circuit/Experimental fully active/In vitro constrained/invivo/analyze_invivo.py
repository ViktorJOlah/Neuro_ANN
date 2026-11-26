"""Summaries and figures for the in vivo-like runs (results/invivo/*.npz from run_invivo.py).

Per run: somatic rate, Vm (median, SD), and dendritic Ca2+ / plateau events by region:
  * Ca2+ event: the region-mean [Ca]i rises > CA_THR_NM above its running baseline (10th percentile in a
    500 ms window); events are merged if < 20 ms apart. Each event is "coupled" if a somatic AP falls
    within -10..+30 ms of its onset, otherwise "isolated" (dendrite-only).
  * Spatial extent: fraction of all dendritic segments whose own [Ca]i is elevated at the event peak
    (local branch event vs cell-wide wave).
  * Plateau: a segment held above -40 mV for > 10 ms while the soma is below -40 mV (NMDA/Ca plateau).
Run: python invivo/analyze_invivo.py results/invivo
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

CA_THR_NM = 30.0
REGIONS = ("basal_prox", "basal_dist", "trunk", "oblique", "tuft")


def regions(meta):
    out = {r: [] for r in REGIONS}
    for i, (name, x, d, kind, tuft, height, trunk) in enumerate(meta):
        d = float(d)
        if kind == "soma":
            continue
        if kind == "basal":
            out["basal_prox" if d <= 70 else "basal_dist"].append(i)
        elif int(tuft):
            out["tuft"].append(i)
        elif trunk in (True, "True", 1):
            out["trunk"].append(i)
        else:
            out["oblique"].append(i)
    return {k: np.array(v, int) for k, v in out.items()}


def baseline(x, w=500):
    n = len(x); b = np.empty(n)
    for i in range(0, n, 50):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2)
        b[i:i + 50] = np.percentile(x[lo:hi], 10)
    return b


def events(trace_nM, spikes_ms, thr=CA_THR_NM):
    d = trace_nM - baseline(trace_nM)
    above = d > thr
    idx = np.nonzero(np.diff(above.astype(int)) == 1)[0] + 1
    ends = np.nonzero(np.diff(above.astype(int)) == -1)[0] + 1
    ev = []
    for s in idx:
        e = ends[ends > s][0] if np.any(ends > s) else len(d)
        if ev and s - ev[-1]["end"] < 20:
            ev[-1]["end"] = e; ev[-1]["peak"] = max(ev[-1]["peak"], d[s:e].max()); continue
        ev.append(dict(onset=int(s), end=int(e), peak=float(d[s:e].max()), t_peak=int(s + np.argmax(d[s:e]))))
    for x in ev:
        x["coupled"] = bool(np.any((spikes_ms >= x["onset"] - 10) & (spikes_ms <= x["onset"] + 30)))
    return ev, d


def analyse(f):
    z = np.load(f, allow_pickle=True)
    meta = z["seg_meta"]; ca = z["cadend"].astype(float) * 1e6; v = z["vdend"].astype(float)
    sp = z["spikes"]; vs = z["vsoma"]; T = ca.shape[1]
    reg = regions(meta)
    soma_i = [i for i, m in enumerate(meta) if m[3] == "soma"][0]
    vsoma_1ms = v[soma_i]
    dend = np.array([i for i, m in enumerate(meta) if m[3] != "soma"])
    base_all = np.array([baseline(c) for c in ca[dend]])
    out = dict(file=os.path.basename(f), rate_hz=len(sp) / (T / 1000), vm_median=float(np.median(vs)),
               vm_sd=float(np.std(vs[vs < -45])), n_spikes=int(len(sp)))
    for r, ii in reg.items():
        if len(ii) == 0:
            continue
        ev, _ = events(ca[ii].mean(0), sp)
        ext = []
        for e in ev:
            el = (ca[dend, e["t_peak"]] - base_all[:, e["t_peak"]]) > CA_THR_NM
            ext.append(el.mean())
        pl = 0
        for i in ii:
            m = (v[i] > -40) & (vsoma_1ms < -40)
            runs = np.split(np.nonzero(m)[0], np.nonzero(np.diff(np.nonzero(m)[0]) > 1)[0] + 1)
            pl += sum(1 for rr in runs if len(rr) > 10)
        out[r] = dict(n_seg=int(len(ii)), ca_events_per_s=len(ev) / (T / 1000),
                      frac_isolated=float(np.mean([not e["coupled"] for e in ev])) if ev else np.nan,
                      mean_peak_nM=float(np.mean([e["peak"] for e in ev])) if ev else np.nan,
                      mean_extent=float(np.mean(ext)) if ext else np.nan,
                      plateaus_per_seg_per_s=pl / len(ii) / (T / 1000))
    return out


def spacetime_figure(f, out_png, t0=0, t1=3000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(f, allow_pickle=True)
    meta = z["seg_meta"]; ca = z["cadend"].astype(float) * 1e6; v = z["vdend"].astype(float)
    signed = np.array([(-1 if m[3] == "basal" else 1) * float(m[2]) for m in meta])
    keep = np.array([m[3] != "soma" for m in meta])
    o = np.argsort(signed[keep]); idx = np.nonzero(keep)[0][o]
    dca = np.array([c - baseline(c) for c in ca[idx]])
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2, 2.2]))
    t = np.arange(len(z["vsoma"])) * 0.1
    ax[0].plot(t, z["vsoma"], lw=0.5, color="k"); ax[0].set_ylabel("soma Vm (mV)")
    cb0 = plt.colorbar(plt.cm.ScalarMappable(cmap="Greys"), ax=ax[0], pad=0.01); cb0.ax.set_visible(False)
    ext = [0, ca.shape[1], signed[idx][0], signed[idx][-1]]
    im = ax[1].imshow(v[idx], aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=-80, vmax=0,
                      interpolation="nearest")
    ax[1].set_ylabel("basal (−) ← path distance (µm) → apical (+)", fontsize=8)
    plt.colorbar(im, ax=ax[1], label="Vm (mV)", pad=0.01)
    im2 = ax[2].imshow(dca, aspect="auto", origin="lower", extent=ext, cmap="viridis", vmin=0,
                       vmax=np.percentile(dca, 99.8), interpolation="nearest")
    plt.colorbar(im2, ax=ax[2], label="Δ[Ca]i (nM)", pad=0.01)
    ax[2].set_xlabel("time (ms)"); ax[2].set_ylabel("path distance (µm)", fontsize=8)
    for a in ax[1:]:
        a.axhline(0, color="w", lw=0.5)
    ax[0].set_xlim(t0, t1)
    fig.suptitle(os.path.basename(f).replace(".npz", ""), fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/invivo"
    rows = [analyse(f) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
    json.dump(rows, open(os.path.join(d, "summary.json"), "w"), indent=1, default=float)
    for r in rows:
        tu = r.get("tuft", {}); bp = r.get("basal_dist", {})
        print("%-34s %5.2f Hz  Vm %.1f±%.1f | tuft Ca ev/s %.2f iso %.2f ext %.2f pk %.0f | basal-dist ev/s %.2f iso %.2f | plateau/seg/s tuft %.3f basal %.3f" % (
            r["file"][:34], r["rate_hz"], r["vm_median"], r["vm_sd"], tu.get("ca_events_per_s", np.nan),
            tu.get("frac_isolated", np.nan), tu.get("mean_extent", np.nan), tu.get("mean_peak_nM", np.nan),
            bp.get("ca_events_per_s", np.nan), bp.get("frac_isolated", np.nan),
            tu.get("plateaus_per_seg_per_s", np.nan), bp.get("plateaus_per_seg_per_s", np.nan)))


if __name__ == "__main__":
    main()
