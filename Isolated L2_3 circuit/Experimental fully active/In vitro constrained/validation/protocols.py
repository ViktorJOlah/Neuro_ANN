"""Validation protocols for the Experimental fully active L2/3 model.

Each protocol builds what it needs on an existing L23Cell, runs, and returns (metrics, traces).
Target values and their sources are in targets.json; run_validation.py runs them all.
"""
from __future__ import annotations

import numpy as np
from neuron import h


def _rec(seg):
    v = h.Vector(); v.record(seg._ref_v)
    return v


def _run(cell, tstop, settle_ms=None):
    t = h.Vector(); t.record(h._ref_t)
    cell.init(settle_ms=settle_ms)
    h.continuerun(tstop)
    return np.array(t)


def halfwidth(t, v, t0, base=None):
    """Half-width (ms) and amplitude (mV) of the response after t0."""
    if base is None:
        base = v[(t > t0 - 20) & (t < t0)].mean()
    m = t >= t0
    tt, vv = t[m], v[m] - base
    k = int(np.argmax(vv)); amp = vv[k]
    above = np.nonzero(vv >= amp / 2)[0]
    if amp <= 0 or len(above) == 0:
        return np.nan, amp
    i0, i1 = above[0], above[-1]
    # contiguous run around the peak
    run = np.split(above, np.nonzero(np.diff(above) > 1)[0] + 1)
    for r in run:
        if r[0] <= k <= r[-1]:
            i0, i1 = r[0], r[-1]
    return tt[i1] - tt[i0], amp


# ---------------------------------------------------------------------------- passive
def passive(cell):
    out = {}
    for cond in ("control", "HCN"):
        cell.unblock()
        if cond == "HCN":
            cell.block("HCN")
        res = {}
        for amp in (-0.02, -0.1):
            ic = cell.iclamp(cell.soma[0], 0.5, 100, 400, amp)
            v = _rec(cell.soma[0](0.5))
            t = _run(cell, 600, settle_ms=800)
            v = np.array(v); ic.amp = 0
            vr = v[(t > 50) & (t < 100)].mean(); vss = v[(t > 450) & (t < 500)].mean()
            if amp == -0.02:
                res["vrest"] = vr; res["rin"] = (vss - vr) / amp
                m = (t > 100) & (t < 250); x = (v[m] - vr) / (vss - vr)
                res["tau"] = float(t[m][np.argmax(x >= 1 - np.exp(-1))] - 100)
            else:
                vmin = v[(t > 100) & (t < 250)].min()
                res["sag_pct"] = 100 * (vss - vmin) / (vr - vmin)
        out[cond] = res
    cell.unblock()
    return dict(rin_ctrl=out["control"]["rin"], rin_hcn_block=out["HCN"]["rin"],
                vrest_ctrl=out["control"]["vrest"], dvrest_hcn_block=out["HCN"]["vrest"] - out["control"]["vrest"],
                tau_ctrl=out["control"]["tau"], tau_hcn_block=out["HCN"]["tau"],
                sag_pct=out["control"]["sag_pct"]), out


# ---------------------------------------------------------------------------- EPSP kinetics vs location
def pick_sites(cell, which, n, rng, dmin, dmax, tuft_only=False):
    pool = [(s, x, d) for s, x, d in cell.segments(which) if dmin <= d <= dmax and s.diam < 2.5
            and (not tuft_only or cell._is_tuft(s, x))]
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def epsp_location(cell, n_sites=8, n_syn=8, target_mv=3.0, seed=1, conds=("control", "HCN", "NMDA", "HCN+NMDA")):
    """'Focal stimulation' near a dendritic site: n_syn synapses on the site's segment activated together.
    Synaptic strength per site is scaled so the control somatic EPSP is ~target_mv (like setting the
    stimulus intensity). Proximal = basal or apical sites 20-100 um from the soma; distal = apical > 100 um."""
    rng = np.random.default_rng(seed)
    groups = {"proximal": pick_sites(cell, "dend", n_sites, rng, 20, 100),
              "distal": pick_sites(cell, "apical", n_sites, rng, 120, 400)}
    vs = _rec(cell.soma[0](0.5))
    res = {g: {c: [] for c in conds} for g in groups}
    amps = {g: {c: [] for c in conds} for g in groups}
    traces = {}
    for g, sites in groups.items():
        for si, (sec, x, d) in enumerate(sites):
            cell.clear_synapses()
            syns = [cell.add_synapse(sec, x) for _ in range(n_syn)]
            for s in syns:
                cell.drive(s)
            scale = 1.0
            for c in conds:
                cell.unblock()
                for b in c.split("+"):
                    if b != "control":
                        cell.block(b)
                for it in range(6 if c == "control" else 1):
                    for s in syns:
                        s["g"] = cell.p["synapse"]["g_ampa_uS"] * scale
                    cell._apply_syn_weights(syns)
                    t = _run_with_events(cell, syns, 300.0)
                    v = np.array(vs)
                    hw, amp = halfwidth(t, v, 100.0)
                    if c != "control" or abs(amp - target_mv) < 0.15:
                        break
                    scale *= (target_mv / max(amp, 0.05)) ** 0.9
                res[g][c].append(hw); amps[g][c].append(amp)
                if si == 0:
                    traces[(g, c)] = (t, v.copy())
    cell.unblock(); cell.clear_synapses()
    m = {}
    for g in groups:
        for c in conds:
            m[f"hw_{g}_{c}"] = float(np.nanmean(res[g][c]))
            m[f"amp_{g}_{c}"] = float(np.nanmean(amps[g][c]))
    m["sites"] = {g: [float(d) for _, _, d in s] for g, s in groups.items()}
    return m, traces


def _run_with_events(cell, syns, tstop, t_ev=100.0):
    t = h.Vector(); t.record(h._ref_t)
    cell.init()
    for s in syns:
        for nc in s["netcons"]:
            nc.event(t_ev)
    h.continuerun(tstop)
    return np.array(t)


# ---------------------------------------------------------------------------- pathway EPSPs (Olah 2025 Fig 3e / 4h)
CONDS4 = ("control", "HCN", "HCN+NMDA", "NMDA")


def _apply(cell, cond):
    cell.unblock()
    for b in cond.split("+"):
        if b != "control":
            cell.block(b)


def pathway_pools(cell):
    """L4 stimulation -> basal + apical within 100 um path distance (bottom-up, proximal);
    L1 stimulation -> apical tuft (segments within tuft_depth_um of the top of the apical arbor)."""
    prox = [(s, x, d) for s, x, d in cell.segments("dend") if 20 <= d <= 100]
    tuft = [(s, x, d) for s, x, d in cell.segments("apical") if cell._is_tuft(s, x)]
    return {"L4": prox, "L1": tuft}


def epsp_pathways(cell, n_draws=6, n_syn=5, target_mv=0.6, seed=2, conds=CONDS4, clustered=False):
    """Minimal-stimulation-like unitary EPSPs: n_syn synapses drawn from the pathway's pool (one axon's
    contacts), strength set once per draw so the control somatic EPSP is ~target_mv, then the same
    synapses in every pharmacological condition. Returns mean half-widths/amplitudes."""
    rng = np.random.default_rng(seed)
    pools = pathway_pools(cell)
    vs = _rec(cell.soma[0](0.5))
    res, traces = {}, {}
    for path, pool in pools.items():
        hw = {c: [] for c in conds}; am = {c: [] for c in conds}
        for k in range(n_draws):
            cell.clear_synapses()
            if clustered:
                i0 = rng.integers(len(pool)); sec = pool[i0][0]
                sites = [(sec, rng.uniform(0.2, 0.8)) for _ in range(n_syn)]
            else:
                sites = [pool[i][:2] for i in rng.choice(len(pool), n_syn, replace=False)]
            syns = [cell.add_synapse(s, x) for s, x in sites]
            for s in syns:
                cell.drive(s)
            g0 = cell.p["synapse"]["g_ampa_uS"]
            for it in range(5):
                _apply(cell, "control")
                for s in syns:
                    s["g"] = g0
                cell._apply_syn_weights(syns)
                t = _run_with_events(cell, syns, 250.0)
                w, a = halfwidth(t, np.array(vs), 100.0)
                if abs(a - target_mv) < 0.05 * target_mv:
                    break
                g0 *= (target_mv / max(a, 0.01)) ** 0.9
            for c in conds:
                _apply(cell, c)
                cell._apply_syn_weights(syns)
                t = _run_with_events(cell, syns, 250.0)
                v = np.array(vs)
                w, a = halfwidth(t, v, 100.0)
                hw[c].append(w); am[c].append(a)
                if k == 0:
                    traces[(path, c)] = (t, v.copy())
        for c in conds:
            res[f"hw_{path}_{c}"] = float(np.nanmean(hw[c]))
            res[f"amp_{path}_{c}"] = float(np.nanmean(am[c]))
    cell.unblock(); cell.clear_synapses()
    return res, traces


# ---------------------------------------------------------------------------- firing and back-propagation
def fi_curve(cell, amps=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5), dur=500.0):
    vs = _rec(cell.soma[0](0.5)); out = {}
    for a in amps:
        ic = cell.iclamp(cell.soma[0], 0.5, 100, dur, a)
        t = _run(cell, 100 + dur + 50); ic.amp = 0
        v = np.array(vs)
        out[a] = int(((v[1:] > -10) & (v[:-1] <= -10)).sum())
    rheo = next((a for a in amps if out[a] > 0), np.nan)
    return dict(counts={str(k): v for k, v in out.items()}, rheobase_nA=rheo), None


def apical_path(cell):
    """Segments along the longest apical path (soma -> most distal apical tip)."""
    tips = [s for s in cell.apical if len(s.children()) == 0]
    tip = max(tips, key=lambda s: h.distance(1.0, sec=s))
    path = []
    sec = tip
    while sec is not None and sec in cell.apical:
        path.append(sec)
        sr = h.SectionRef(sec=sec)
        sec = sr.parent if sr.has_parent() else None
    path = path[::-1]
    segs = [(s, seg.x, h.distance(seg.x, sec=s)) for s in path for seg in s]
    return segs


def bap(cell, n_ap=1, freq=100.0, amp=2.0, width=2.0, path=None, conds=("control",)):
    """Somatic brief current pulses (n_ap at freq) -> bAP amplitude, latency and peak [Ca]i along the
    main apical path. Returns per condition arrays (distance, amplitude, delay, dCa)."""
    path = path or apical_path(cell)
    out = {}
    for c in conds:
        _apply(cell, c)
        vrec = [_rec(s(x)) for s, x, _ in path]
        carec = []
        for s, x, _ in path:
            r = h.Vector()
            if h.ismembrane("ca_ion", sec=s):
                r.record(s(x)._ref_cai)
            carec.append(r)
        vs = _rec(cell.soma[0](0.5))
        ics = [cell.iclamp(cell.soma[0], 0.5, 100 + k * 1000.0 / freq, width, amp) for k in range(n_ap)]
        t = _run(cell, 100 + n_ap * 1000.0 / freq + 250)
        for ic in ics:
            ic.amp = 0
        vsa = np.array(vs)
        nsp = int(((vsa[1:] > -10) & (vsa[:-1] <= -10)).sum())
        t0 = t[np.argmax(vsa)] if nsp else np.nan
        d = np.array([p[2] for p in path]); A = []; D = []; CA = []
        for (s, x, _), vr, cr in zip(path, vrec, carec):
            va = np.array(vr); base = va[(t > 80) & (t < 99)].mean()
            k = np.argmax(va[t < 100 + n_ap * 1000.0 / freq + 30]); A.append(va[k] - base); D.append(t[k] - t0)
            ca = np.array(cr)
            CA.append((ca[t >= 100].max() - ca[(t > 80) & (t < 99)].mean()) * 1e6 if len(ca) else np.nan)  # nM
        out[c] = dict(d=d, amp=np.array(A), delay=np.array(D), dca_nM=np.array(CA), n_spikes=nsp, t=t, vsoma=vsa)
    cell.unblock()
    return out


def apical_site(cell, target_um=200.0):
    """Segment on the main apical path closest to target_um path distance."""
    path = apical_path(cell)
    return min(path, key=lambda p: abs(p[2] - target_um))


def dendritic_injection(cell, site_um=200.0, amps=np.arange(0.2, 3.01, 0.2), tau_r=0.5, tau_d=5.0, conds=("control",)):
    """EPSC-shaped current (double exponential) into the apical dendrite at ~site_um (Larkum 2007,
    Ledergerber & Larkum 2012). Threshold = smallest amplitude giving a regenerative dendritic event
    (dendritic dV/dt > 20 mV/ms not explained by a somatic AP arriving first). Reports dendritic spike
    half-width and number of somatic APs at threshold and at 2x threshold."""
    sec, x, d = apical_site(cell, site_um)
    res = {}
    for c in conds:
        _apply(cell, c)
        vd = _rec(sec(x)); vs = _rec(cell.soma[0](0.5))
        tt = np.arange(0, 60, h.dt if h.dt > 0 else 0.025)
        rows = []
        for a in amps:
            wave = a * (np.exp(-tt / tau_d) - np.exp(-tt / tau_r))
            wave /= (np.exp(-tt / tau_d) - np.exp(-tt / tau_r)).max() / a if False else 1.0
            pk = (np.exp(-tt / tau_d) - np.exp(-tt / tau_r)).max()
            ic = h.IClamp(sec(x)); ic.delay = 0; ic.dur = 1e9
            iv = h.Vector(np.concatenate([[0, 0], a * (np.exp(-tt / tau_d) - np.exp(-tt / tau_r)) / pk, [0]]))
            tv = h.Vector(np.concatenate([[0, 100 - 1e-6], 100 + tt, [100 + tt[-1] + 1e-3]]))
            iv.play(ic._ref_amp, tv, 1)
            t = _run(cell, 200)
            iv.play_remove(); ic.amp = 0
            v1, v0 = np.array(vd), np.array(vs)
            m = (t > 99) & (t < 160)
            dvdt = np.diff(v1[m]) / np.diff(t[m])
            nsp = int(((v0[1:] > -10) & (v0[:-1] <= -10)).sum())
            t_sp = t[1:][(v0[1:] > -10) & (v0[:-1] <= -10)]
            k = int(np.argmax(dvdt)); t_dmax = t[m][k]
            regen = dvdt.max() > 20 and (nsp == 0 or t_dmax < t_sp[0])
            hw, amp = halfwidth(t, v1, 100.0)
            rows.append(dict(amp_nA=float(a), dvdt_max=float(dvdt.max()), dend_peak=float(v1[m].max()),
                             dend_hw=float(hw), dend_amp=float(amp), n_soma_ap=nsp, regen_first=bool(regen),
                             t=t.copy(), vd=v1.copy(), vs=v0.copy()))
        thr = next((r["amp_nA"] for r in rows if r["regen_first"]), np.nan)
        res[c] = dict(site_um=float(d), threshold_nA=thr, rows=rows)
    cell.unblock()
    return res


# ---------------------------------------------------------------------------- Ca2+ readouts
def trunk_tuft_path(cell):
    """Trunk sections followed by the longest tuft branch: the line used for Ca2+ space-time plots."""
    segs = [(s, seg.x, h.distance(seg.x, sec=s)) for s in cell.trunk for seg in s]
    sec = cell.trunk[-1]
    while sec.children():
        sec = max(sec.children(), key=lambda c: h.distance(1, sec=c))
        segs += [(sec, seg.x, h.distance(seg.x, sec=sec)) for seg in sec]
    return segs


def critical_frequency(cell, freqs=(20, 50, 80, 100, 120, 140, 160, 200), n_ap=5, conds=("control",)):
    """Larkum 2007 / 1999: 5 somatic APs at increasing frequency; tuft [Ca]i integral (and peak tuft V)
    vs frequency. A supralinear step marks the critical frequency (L2/3 rat: ~128 Hz, no burst)."""
    tuft = [(s, x) for s, x, d in trunk_tuft_path(cell) if cell._is_tuft(s, x)]
    s_t, x_t = tuft[len(tuft) // 2]
    res = {}
    for c in conds:
        _apply(cell, c)
        rows = []
        for f in freqs:
            ca = h.Vector(); ca.record(s_t(x_t)._ref_cai); vt = _rec(s_t(x_t)); vs = _rec(cell.soma[0](0.5))
            ics = [cell.iclamp(cell.soma[0], 0.5, 100 + k * 1000.0 / f, 2.0, 2.5) for k in range(n_ap)]
            t = _run(cell, 400)
            for ic in ics:
                ic.amp = 0
            cc = np.array(ca); v = np.array(vs)
            base = cc[(t > 80) & (t < 99)].mean()
            integ = np.trapezoid(np.clip(cc - base, 0, None), t) * 1e6 / 1000.0   # nM*s
            rows.append(dict(freq=f, ca_integral_nMs=float(integ), tuft_vmax=float(np.array(vt).max()),
                             n_spikes=int(((v[1:] > -10) & (v[:-1] <= -10)).sum())))
        res[c] = rows
    cell.unblock()
    return res


def ca_spacetime(cell, stim, conds=("control",), tstop=300.0):
    """Space-time [Ca]i and Vm along the trunk -> tuft path for a stimulus callable stim(cell) that
    installs IClamps/synapses (and returns objects to keep). Returns per condition (d, t, ca, v)."""
    path = trunk_tuft_path(cell)
    res = {}
    for c in conds:
        _apply(cell, c)
        keep = stim(cell)
        car, vr = [], []
        for s, x, _ in path:
            a = h.Vector(); a.record(s(x)._ref_cai, 0.25); car.append(a)
            b = h.Vector(); b.record(s(x)._ref_v, 0.25); vr.append(b)
        vs = h.Vector(); vs.record(cell.soma[0](0.5)._ref_v, 0.25)
        cell.init()
        if callable(getattr(keep, "queue", None)):
            keep.queue()
        h.continuerun(tstop)
        for ic in cell.iclamps:
            ic.amp = 0
        cell.iclamps = []
        ca = np.array([np.array(a) for a in car]); v = np.array([np.array(b) for b in vr])
        res[c] = dict(d=np.array([p[2] for p in path]), t=np.arange(ca.shape[1]) * 0.25, ca=ca, v=v,
                      vsoma=np.array(vs), tuft=np.array([cell._is_tuft(s, x) for s, x, _ in path]))
        cell.clear_synapses()
    cell.unblock()
    return res


class SynCluster:
    """n synapses on one section, activated together (tuft or basal NMDA spike)."""
    def __init__(self, cell, sec, n, g, t0=100.0, x0=0.3, x1=0.7):
        self.cell, self.t0 = cell, t0
        self.syns = [cell.add_synapse(sec, x, g_ampa=g) for x in np.linspace(x0, x1, n)]
        for s in self.syns:
            cell.drive(s)

    def queue(self):
        for s in self.syns:
            for nc in s["netcons"]:
                nc.event(self.t0)


def nmda_spike_io(cell, where="basal", n_max=20, g=0.4e-3, conds=("control", "NMDA"), dists=("proximal", "distal")):
    """Uncaging-like input-output curve (Branco & Hausser 2011): k = 1..n_max synapses on one thin
    branch, all at once; somatic EPSP peak vs k, measured vs expected (k x unitary). Proximal = segment
    near the branch base, distal = near its tip."""
    pool = cell.basal if where == "basal" else [s for s in cell.apical if s not in cell.trunk]
    terms = [s for s in pool if not s.children() and s.L > 60 and s.diam < 1.2]
    sec = max(terms, key=lambda s: s.L)
    vs = _rec(cell.soma[0](0.5)); vd = None
    res = {}
    for c in conds:
        _apply(cell, c)
        for dd in dists:
            xs = (0.1, 0.3) if dd == "proximal" else (0.75, 0.95)
            vd = _rec(sec(np.mean(xs)))
            peaks, dpk = [], []
            for k in range(1, n_max + 1):
                cell.clear_synapses()
                cl = SynCluster(cell, sec, k, g, x0=xs[0], x1=xs[1])
                cell.init(); cl.queue(); h.continuerun(250)
                v = np.array(vs); t = np.arange(len(v)) * h.dt
                peaks.append(v.max() - v[int(95 / h.dt)]); dpk.append(np.array(vd).max())
            peaks = np.array(peaks)
            res[(c, dd)] = dict(k=np.arange(1, n_max + 1), measured=peaks, expected=peaks[0] * np.arange(1, n_max + 1),
                                dend_peak=np.array(dpk), sec=sec.name(),
                                d=(h.distance(xs[0], sec=sec), h.distance(xs[1], sec=sec)))
    cell.clear_synapses(); cell.unblock()
    return res
