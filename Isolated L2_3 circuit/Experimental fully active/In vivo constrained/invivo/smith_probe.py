"""Dendritic 'patch-pipette' recordings during in vivo-like input, compared with Smith et al. 2013 (Nature 503:115).

Runs the same in vivo-like input as run_invivo.py, records Vm at 0.05 ms at the soma and at dendritic sites
75-300 um from the soma (thin apical/oblique branches, trunk, tuft, basal), then:
  * detects spikes at each site (local maxima with prominence > PROM_MV and half-width < 6 ms),
  * splits them into isolated spikes (>= 50 ms from any other; Smith's presumed bAPs) and spikes in bursts,
  * reports bAP amplitude/FWHM vs distance (Smith ED Fig 1e,f), max instantaneous rate, mean rate and
    variance-to-mean ratio (Fig 2a), onset phase slope of isolated vs burst spikes (Fig 2b-d),
  * draws Smith-style traces (20 mV / 200 ms bars) for every site.

Run: python invivo/smith_probe.py --cond control --td-rate 0 --out results/smith
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from model.l23_cell import L23Cell, load_params  # noqa: E402
from neuron import h  # noqa: E402
import run_invivo as R  # noqa: E402
import visual_drive as VD  # noqa: E402

DT_REC = 0.05
PROM_MV = 6.0
PEAK_MIN_MV = -45.0     # spikes must reach this; at -95 mV (dendritic -50 pA) large fast AMPA EPSPs otherwise pass
MIN_DVDT = 15.0          # mV/ms; EPSPs rise at < ~2 mV/ms, Smith's spikes in bursts at 5-30 mV/ms

# Smith et al. 2013 targets (mouse V1 L2/3, in vivo, distal = >75 um)
SMITH = {
    "bap_amp_mV": {"soma": 80, "100-150": 27, "175-300": 11},          # ED Fig 1e (read off bars)
    "bap_fwhm_ms": {"soma": 0.9, "100-150": 2.0, "175-300": 3.0},      # ED Fig 1f
    "max_inst_rate_hz": {"soma": 80, "proximal": 60, "distal": 220},  # Fig 2a (group means, read off)
    "mean_rate_hz": {"soma": 1.5, "proximal": 1.0, "distal": 3.5},
    "fano": {"soma": 1.7, "proximal": 1.6, "distal": 2.4},
    "phase_slope_ms-1": {"isolated_bAP": 10.0, "spikes_in_bursts": 3.0},  # Fig 2d (means, read off)
    "rin_dend_MOhm": "100-300 (doubling every ~300 um); soma ~110",
    "burst_envelope": "20-40 mV slow depolarisation lasting 50-300 ms with riding Na+ spikelets of variable size",
}


def pick_sites(cell):
    """(label, sec, x, kind) at fixed path distances on the main dendrite types."""
    want = []
    def at(sec, d):
        d0, d1 = h.distance(0, sec=sec), h.distance(1, sec=sec)
        return min(max((d - d0) / (d1 - d0), 0.02), 0.98)
    def longest(cands, d):
        c = [s for s in cands if h.distance(0, sec=s) <= d <= h.distance(1, sec=s)]
        return max(c, key=lambda s: s.diam * s.L) if c else None
    obl = [s for s in cell.apical if s not in cell.trunk and not cell._is_tuft(s, 0.5)]
    tuft = [s for s in cell.apical if cell._is_tuft(s, 0.5) and s not in cell.trunk]
    bas = [s for s in cell.dends if s not in cell.apical]
    for d in (100, 150):
        s = longest(cell.trunk, d)
        if s is not None:
            want.append((f"trunk {d}", s, at(s, d), "trunk"))
    for d in (80, 100, 150, 200):
        s = longest(obl, d)
        if s is not None:
            want.append((f"oblique {d}", s, at(s, d), "oblique"))
    for d in (250, 300):
        s = longest(tuft, d)
        if s is not None:
            want.append((f"tuft {d}", s, at(s, d), "tuft"))
    for d in (40, 100, 150):
        s = longest(bas, d)
        if s is not None:
            want.append((f"basal {d}", s, at(s, d), "basal"))
    return want


def detect(v, dt=DT_REC, prom=PROM_MV):
    """Spike peaks: local maxima with prominence > prom (relative to the min in the preceding 4 ms) and
    falling back by prom/2 within 6 ms."""
    w4 = int(4 / dt); w6 = int(6 / dt)
    dv = np.diff(v)
    pk = np.nonzero((dv[:-1] > 0) & (dv[1:] <= 0))[0] + 1
    out = []
    for p in pk:
        if p < w4 or p + w6 >= len(v):
            continue
        base = v[p - w4:p].min()
        if v[p] - base < prom or v[p] < PEAK_MIN_MV:
            continue
        if v[p] - v[p:p + w6].min() < prom / 2:
            continue
        if np.max(np.diff(v[p - int(2 / dt):p + 1])) / dt < MIN_DVDT:   # spikes, not EPSPs
            continue
        if out and (p - out[-1]) * dt < 1.0:      # keep the larger of two peaks < 1 ms apart
            if v[p] > v[out[-1]]:
                out[-1] = p
            continue
        out.append(p)
    return np.array(out, int)


def spike_shape(v, p, dt=DT_REC):
    """amplitude from the pre-spike minimum (3 ms window), FWHM, max dV/dt in the first 5 mV (phase slope
    in 1/ms, the slope of dV/dt vs V at the foot)."""
    w = int(3 / dt)
    seg = v[p - w:p + 1]; i0 = np.argmin(seg); base = seg[i0]
    amp = v[p] - base; half = base + amp / 2
    a = p
    while a > p - w and v[a] > half:
        a -= 1
    b = p
    while b < len(v) - 1 and v[b] > half and (b - p) * dt < 15:
        b += 1
    fwhm = (b - a) * dt
    # phase slope: d(dV/dt)/dV between 10% and 30% of the rise (Smith Fig 2b-d)
    rise = v[p - w + i0:p + 1]
    dvdt = np.gradient(rise, dt)
    m = (rise > base + 0.1 * amp) & (rise < base + 0.3 * amp)
    ps = np.polyfit(rise[m], dvdt[m], 1)[0] if m.sum() >= 3 else np.nan
    return float(amp), float(fwhm), float(ps)


def classify(t_ms):
    iso = np.zeros(len(t_ms), bool)
    for i, t in enumerate(t_ms):
        others = np.delete(t_ms, i)
        iso[i] = not np.any(np.abs(others - t) < 50)
    return iso


def bap_at_site(v, soma_v, dt=DT_REC):
    """Amplitude/FWHM at this site of isolated somatic APs (>= 50 ms from other somatic APs): peak within
    0-4 ms after the somatic AP peak minus the mean Vm 3-1 ms before it (Smith ED Fig 1e,f)."""
    sp = detect(soma_v, dt, prom=30.0) * dt
    if len(sp) == 0:
        return np.nan, np.nan, 0
    iso = sp[classify(sp)]
    amps, fws = [], []
    for t in iso:
        i = int(t / dt)
        if i < int(5 / dt) or i + int(15 / dt) >= len(v):
            continue
        base = v[i - int(3 / dt):i - int(1 / dt)].mean()
        w = v[i:i + int(4 / dt)]; p = i + int(np.argmax(w)); amp = v[p] - base; half = base + amp / 2
        a = p
        while a > i - int(3 / dt) and v[a] > half:
            a -= 1
        b = p
        while b < len(v) - 1 and v[b] > half and (b - p) * dt < 15:
            b += 1
        amps.append(amp); fws.append((b - a) * dt)
    return (float(np.median(amps)) if amps else np.nan, float(np.median(fws)) if fws else np.nan, len(amps))


def site_stats(v, dt=DT_REC, T=3000.0, soma_v=None):
    pk = detect(v, dt); t = pk * dt
    iso = classify(t) if len(t) else np.zeros(0, bool)
    shapes = [spike_shape(v, p, dt) for p in pk]
    amps = np.array([s[0] for s in shapes]); fw = np.array([s[1] for s in shapes]); ps = np.array([s[2] for s in shapes])
    isi = np.diff(t)
    counts = np.histogram(t, bins=np.arange(0, T + 1, 250))[0]
    return dict(n=int(len(t)), mean_rate_hz=len(t) / (T / 1000), max_inst_rate_hz=float(1000 / isi.min()) if len(isi) else 0.0,
                fano=float(counts.var() / counts.mean()) if counts.mean() > 0 else np.nan,
                n_isolated=int(iso.sum()), n_burst=int((~iso).sum()),
                iso_amp_mV=float(np.median(amps[iso])) if iso.any() else np.nan,
                iso_fwhm_ms=float(np.median(fw[iso])) if iso.any() else np.nan,
                burst_amp_mV=float(np.median(amps[~iso])) if (~iso).any() else np.nan,
                iso_phase_slope=float(np.nanmedian(ps[iso])) if iso.any() else np.nan,
                burst_phase_slope=float(np.nanmedian(ps[~iso])) if (~iso).any() else np.nan,
                vm_median=float(np.median(v)),
                **dict(zip(("bap_amp_mV", "bap_fwhm_ms", "n_bap"), bap_at_site(v, soma_v if soma_v is not None else v, dt))))


def run(cell_id, cond, td_rate, seed, params=None, cal=None, hyper_nA=0.0, T=3000.0, ori=None, sites_on_clusters=False, hyper_site="basal 150"):
    cal = cal or json.load(open(os.path.join(HERE, "calibration.json")))
    cell = L23Cell(params or load_params())
    inputs = dict(np.load(os.path.join(HERE, "inputs", f"cell_{cell_id}.npz")))
    keep, _ = R.build(cell, inputs, cal["g_scale"], cal["gi_scale"], td_rate, 2.0, seed, cluster=cal.get("cluster"))
    if ori is not None and cal.get("visual") is not None:
        kv, info = VD.add(cell, ori, cal["visual"], seed=seed, T=T)
        keep += kv
    else:
        info = None
    for b in cond.split("+"):
        if b != "control":
            cell.block(b)
    sites = [("soma", cell.soma[0], 0.5, "soma")] + pick_sites(cell)
    if info:
        for k, (sec, x) in enumerate(info["centres"][:3]):
            sites.append((f"cluster{k + 1} {h.distance(x, sec=sec):.0f}", sec, x, "cluster"))
    recs = []
    for lab, sec, x, kind in sites:
        r = h.Vector(); r.record(sec(x)._ref_v, DT_REC); recs.append(r)
    clamps = []
    if hyper_nA:
        for lab, sec, x, kind in sites[1:]:
            if lab == hyper_site:              # steady current through the 'dendritic pipette' (Smith Fig 4)
                clamps.append(cell.iclamp(sec, x, 0, T, hyper_nA))
    cell.init()
    for ncs, times in keep:
        for tt in times:
            for nc in ncs:
                nc.event(tt)
    h.continuerun(T)
    meta = [(lab, sec.name(), float(x), float(h.distance(x, sec=sec)), kind, float(sec(x).diam)) for lab, sec, x, kind in sites]
    return meta, np.array([np.array(r)[:int(T / DT_REC)] for r in recs], np.float32)


def rin_profile(params=None):
    """Steady-state input resistance at each site (-50 pA, 300 ms), Smith ED Fig 1a."""
    cell = L23Cell(params or load_params())
    sites = [("soma", cell.soma[0], 0.5, "soma")] + pick_sites(cell)
    out = []
    for lab, sec, x, kind in sites:
        v = h.Vector(); v.record(sec(x)._ref_v)
        ic = cell.iclamp(sec, x, 50, 300, -0.05)
        cell.init(); h.continuerun(400)
        a = np.array(v); tt = np.arange(len(a)) * cell.p["dt"]
        vr = a[(tt > 20) & (tt < 50)].mean(); vss = a[(tt > 320) & (tt < 350)].mean(); vmin = a[(tt > 50) & (tt < 350)].min()
        out.append(dict(site=lab, d=float(h.distance(x, sec=sec)), rin=float((vss - vr) / -0.05), sag_mV=float(vss - vmin)))
        ic.amp = 0
    return out


def figure(meta, V, stats, out_png, title, t0=0, t1=3000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(meta)
    fig, ax = plt.subplots(n, 1, figsize=(9, 0.9 * n + 0.8), sharex=True)
    t = np.arange(V.shape[1]) * DT_REC
    m = (t >= t0) & (t <= t1)
    for i, (lab, name, x, d, kind, diam) in enumerate(meta):
        col = "#2b6cb0" if kind == "soma" else "#2f855a"
        ax[i].plot(t[m], V[i][m], lw=0.5, color=col)
        ax[i].set_ylim(-80, 30 if kind == "soma" else 0)
        ax[i].set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax[i].spines[s].set_visible(False)
        st = stats[i]
        ax[i].text(t0 - 0.01 * (t1 - t0), -40, f"{lab} µm" if kind != "soma" else "soma", ha="right", va="center", fontsize=8)
        ax[i].text(t1, -10, f"{st['mean_rate_hz']:.1f} Hz, max {st['max_inst_rate_hz']:.0f} Hz", ha="right", fontsize=6, color="0.4")
    ax[-1].plot([t1 - 200, t1], [-78, -78], color="k", lw=1.5)
    ax[-1].plot([t1, t1], [-78, -58], color="k", lw=1.5)
    ax[-1].text(t1 - 100, -76, "200 ms", ha="center", va="bottom", fontsize=7)
    ax[-1].text(t1 + 10, -68, "20 mV", fontsize=7, va="center")
    ax[-1].set_xticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="82141")
    ap.add_argument("--cond", default="control")
    ap.add_argument("--td-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hyper-nA", type=float, default=0.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "smith"))
    ap.add_argument("--rin", action="store_true")
    ap.add_argument("--set", default="{}", help="JSON param overrides (dotted keys), e.g. for fitting")
    ap.add_argument("--cal", default="{}", help="JSON overrides of calibration.json (g_scale, cluster, ...)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ori", type=float, default=None, help="grating orientation (deg) for the visual drive")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    P = load_params(**json.loads(a.set))
    cal = json.load(open(os.path.join(HERE, "calibration.json"))); cal.update(json.loads(a.cal))
    os.makedirs(a.out, exist_ok=True)
    if a.rin:
        r = rin_profile(P)
        json.dump(r, open(os.path.join(a.out, "rin_profile.json"), "w"), indent=1)
        for x in r:
            print("%-12s d=%5.0f  Rin %6.1f MOhm  sag %.2f mV" % (x["site"], x["d"], x["rin"], x["sag_mV"]))
        return
    meta, V = run(a.cell, a.cond, a.td_rate, a.seed, params=P, cal=cal, hyper_nA=a.hyper_nA, ori=a.ori)
    stats = [site_stats(v.astype(float), soma_v=V[0].astype(float)) for v in V]
    tag = f"cell{a.cell}_{a.cond.replace('+', '-')}_td{a.td_rate:g}_s{a.seed}" + (f"_hyp{abs(a.hyper_nA):g}" if a.hyper_nA else "") + (f"_ori{a.ori:g}" if a.ori is not None else "") + a.tag
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), V=V, meta=np.array(meta, dtype=object))
    json.dump([dict(site=m[0], d=m[3], diam=m[5], **s) for m, s in zip(meta, stats)],
              open(os.path.join(a.out, tag + "_stats.json"), "w"), indent=1)
    if not a.no_fig:
        figure(meta, V, stats, os.path.join(a.out, tag + "_traces.png"), tag)
    for m, s in zip(meta, stats):
        print("%-12s d=%5.0f diam %.2f | n %3d rate %5.1f maxinst %5.0f fano %.2f | iso %2d amp %5.1f fwhm %4.1f ps %5.1f | burst %3d amp %5.1f ps %5.1f | Vm %.1f" % (
            m[0], m[3], m[5], s["n"], s["mean_rate_hz"], s["max_inst_rate_hz"], s["fano"], s["n_isolated"], s["iso_amp_mV"],
            s["iso_fwhm_ms"], s["iso_phase_slope"], s["n_burst"], s["burst_amp_mV"], s["burst_phase_slope"], s["vm_median"]))


if __name__ == "__main__":
    main()
