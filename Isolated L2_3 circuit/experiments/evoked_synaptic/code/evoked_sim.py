"""Simulated stimulating-electrode experiment on Allen all-active L2/3 PCs (feasibility test).

Two extracellular stimulation pathways, as in a slice experiment with one electrode in L1 and one in L2/3:
  * L1:   N_SYN synapses on the apical tuft (apical segments beyond TUFT_FRAC of the maximal apical path distance)
  * L2/3: N_SYN synapses on basal dendrites and proximal apical obliques (20-120 um from the soma)
Each synapse is AMPA (Exp2Syn 0.2/2 ms) + NMDA (NMDA_JS.mod: 2/50 ms, Jahr-Stevens Mg block), NMDA peak
conductance = NMDA_RATIO x AMPA. A stimulus pulse of intensity n (0..1) recruits the first round(n*N_SYN)
synapses of a fixed recruitment order (nearest fibres first); each releases with probability P0*R, where R is
a Tsodyks-Markram resource (depletes by U per pulse, recovers with TAU_REC), 2 ms latency. No inhibition
(the GABA-A-blocked configuration). The AMPA conductance per pathway is calibrated so that a full-intensity
single pulse gives a CAL_MV (15 mV) somatic EPSP at rest (like setting the stimulus strength in the rig).
A per-trace somatic DC current (0, or 0.4-1.1 x rheobase) sets the baseline, so some traces spike.

Run:  python evoked_sim.py --cell 521409057 --n 150 --out ../data            (4 workers)
      python evoked_sim.py --cell 521409057 --nmda-ratio 0 --out ../data_ampa  (AMPA only, as in the Billeh V1 model)
      python evoked_sim.py --cell 521409057 --chimera-soma 513531471 ...     (A's morphology and dendrites, B's soma+axon)
Output: <out>/<tag>/evoked.npz with stim (n, T, 2) intensity per ms, dc (n,), V (n, T) somatic Vm at 1 ms
(recording convention), plus, for the first --controls traces, an NMDA-blocked run and a repeat with new release draws; meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))           # Isolated L2_3 circuit/
MODELS = os.path.join(ROOT, "simulations", "additional_cells", "models")
sys.path.insert(0, os.path.join(ROOT, "simulations", "additional_cells", "code"))
MECH = os.environ.get("ALLEN_MECH", "/tmp/allen_model")

N_SYN, TUFT_FRAC, NMDA_RATIO = 40, 0.6, 1.0
P0, U, TAU_REC, LATENCY = 0.8, 0.4, 300.0, 2.0
CAL_MV = 15.0
T_MS, QUIET_MS, DT = 4000, 300, 0.025
PERI = {"487661754": os.environ.get("ALLEN_PERI", "/tmp/allen_peri")}   # the Standard's own cell: Allen perisomatic model
# 488462783 (multicompartmental models/L2_3 PC/NEURON/orig_file + modfiles + NMDA_JS.mod, compiled with nrnivmodl in that folder)
G_MAX = None                 # optional cap on the calibrated AMPA conductance per synapse (uS)
RHEO = {"487661754": 83.8, "521409057": 207.0, "513531471": 144.0, "318808427": 114.0, "502359001": 144.0, "486262299": 490.0}

_cell = None


def merged_dir(cell, soma_from):
    """Model dir with cell's morphology + apical/basal parameters and soma_from's somatic/axonal parameters."""
    a = json.load(open(os.path.join(MODELS, cell, "fit_parameters.json")))
    b = json.load(open(os.path.join(MODELS, soma_from, "fit_parameters.json")))
    g = [x for x in a["genome"] if x["section"] in ("apic", "dend", "glob")]
    g += [x for x in b["genome"] if x["section"] in ("soma", "axon")]
    a["genome"] = g
    d = tempfile.mkdtemp(prefix=f"chim_{cell}_{soma_from}_")
    json.dump(a, open(os.path.join(d, "fit_parameters.json"), "w"))
    os.symlink(os.path.join(MODELS, cell, "reconstruction.swc"), os.path.join(d, "reconstruction.swc"))
    return d


class EvokedCell:
    def __init__(self, cell_dir, seed):
        from neuron import h
        self.h = h
        if cell_dir in PERI.values():
            sys.path.insert(0, os.path.join(ROOT, "simulations", "code"))
            import allen_cell as pc
            self.run_ = pc.Runner(cell_dir, dt=DT)
        else:
            import allactive_cell as ac
            self.run_ = ac.Runner(cell_dir, MECH, dt=DT)
        rng = np.random.default_rng(seed)
        h.distance(0, 0.5, sec=h.soma[0])
        segs = {"apic": [], "dend": []}
        for sec in h.allsec():
            kind = sec.name().split("[")[0]
            if kind in segs:
                for s in sec:
                    segs[kind].append((sec, s.x, h.distance(s.x, sec=sec), sec.L / sec.nseg))
        dmax = max(d for _, _, d, _ in segs["apic"])
        tuft = [s for s in segs["apic"] if s[2] >= TUFT_FRAC * dmax]
        prox = [s for s in segs["dend"] + segs["apic"] if 20.0 <= s[2] <= 120.0]
        self.sites = {}
        self.syn = {}
        for name, pool in (("L1", tuft), ("L23", prox)):
            w = np.array([p[3] for p in pool]); w /= w.sum()
            idx = rng.choice(len(pool), N_SYN, p=w)
            sites = [pool[i] for i in idx]
            order = np.argsort([s[2] for s in sites]) if name == "L23" else np.argsort([-s[2] for s in sites])
            sites = [sites[i] for i in order]      # recruitment order: nearest to the electrode first
            self.sites[name] = [float(s[2]) for s in sites]
            lst = []
            for sec, x, _, _ in sites:
                am = h.Exp2Syn(sec(x)); am.tau1, am.tau2, am.e = 0.2, 2.0, 0.0
                nm = h.NMDA_JS(sec(x))
                nca, ncn = h.NetCon(None, am), h.NetCon(None, nm)
                lst.append((am, nm, nca, ncn))
            self.syn[name] = lst
        self.dmax = float(dmax)
        self.n_tuft, self.n_prox = len(tuft), len(prox)
        self.g_ampa = {"L1": 0.5e-3, "L23": 0.5e-3}     # uS, set by calibrate()
        self.ic = self.run_.stim

    def set_weights(self, nmda=True, ampa=True):
        for name, lst in self.syn.items():
            g = self.g_ampa[name]
            for am, nm, nca, ncn in lst:
                nca.weight[0] = g if ampa else 0.0
                ncn.weight[0] = g * NMDA_RATIO if nmda else 0.0

    def run(self, events, dc_pA, nmda=True, ampa=True):
        """events: {path: [(t_ms, k_syn_index), ...]} release events. Returns Vm at 1 ms (T_MS,)."""
        h = self.h
        self.set_weights(nmda, ampa)
        n_fine = int(T_MS / DT)
        i_fine = np.zeros(n_fine); i_fine[int(QUIET_MS / 3 / DT):] = dc_pA   # DC on at 100 ms

        def queue():
            for name, ev in events.items():
                lst = self.syn[name]
                for t, k in ev:
                    lst[k][2].event(t); lst[k][3].event(t)
        # Runner._init settles and then moves t to 0, so events are queued after initialisation
        self.run_.ivec.from_python(i_fine * 1e-3)
        self.run_.tvec.from_python(np.arange(n_fine) * DT)
        self.run_._init()
        queue()
        h.continuerun(T_MS)
        v = np.array(self.run_.vrec)[:T_MS] - self.run_.info["junction_potential"]
        return v.astype(np.float32)

    def calibrate(self):
        for name in ("L1", "L23"):
            g = 0.5e-3
            for _ in range(8):
                self.g_ampa[name] = g
                ev = {name: [(QUIET_MS + LATENCY, k) for k in range(N_SYN)]}
                v = self.run(ev, 0.0)
                amp = v[QUIET_MS:QUIET_MS + 200].max() - v[QUIET_MS - 10:QUIET_MS].mean()
                g *= (CAL_MV / max(amp, 0.05)) ** 0.8
                if G_MAX:
                    g = min(g, G_MAX)
            self.g_ampa[name] = g
        return dict(self.g_ampa)


def protocol(rng, kind):
    """Stimulus pulses per pathway: {path: [(t_ms, intensity)]}."""
    pulses = {"L1": [], "L23": []}
    t_end = T_MS - 50
    if kind == "poisson":
        for p in pulses:
            r = rng.uniform(1, 15)
            t = QUIET_MS + rng.exponential(1000 / r)
            while t < t_end:
                pulses[p].append((t, rng.uniform(0.1, 1.0)))
                t += rng.exponential(1000 / r)
    elif kind == "trains":
        t = QUIET_MS + rng.uniform(0, 100)
        while t < t_end - 600:
            f = rng.choice([10, 20, 50, 100]); n = rng.uniform(0.2, 1.0)
            which = rng.choice(["L1", "L23", "both"]); off = rng.uniform(-20, 20)
            for j in range(5):
                tj = t + j * 1000 / f
                if which in ("L1", "both"):
                    pulses["L1"].append((tj, n))
                if which in ("L23", "both"):
                    pulses["L23"].append((tj + (off if which == "both" else 0), n))
            t += 5 * 1000 / f + rng.uniform(400, 800)
    else:  # single / paired pulses
        t = QUIET_MS + rng.uniform(0, 100)
        while t < t_end - 300:
            which = rng.choice(["L1", "L23", "both"]); n = rng.uniform(0.1, 1.0); off = rng.uniform(-20, 20)
            if which in ("L1", "both"):
                pulses["L1"].append((t, n))
            if which in ("L23", "both"):
                pulses["L23"].append((t + (off if which == "both" else 0), n))
            t += rng.uniform(250, 500)
    return {p: sorted([x for x in v if QUIET_MS <= x[0] < t_end]) for p, v in pulses.items()}


def releases(rng, pulses):
    """Pulse list -> synaptic release events with stochastic, depressing release."""
    ev = {}
    for p, lst in pulses.items():
        R, t_last, out = 1.0, -1e9, []
        for t, n in lst:
            R = 1 - (1 - R) * np.exp(-(t - t_last) / TAU_REC)
            k = int(round(n * N_SYN))
            rel = rng.random(k) < P0 * R
            out += [(t + LATENCY, j) for j in np.nonzero(rel)[0]]
            R *= (1 - U)
            t_last = t
        ev[p] = out
    return ev


def stim_array(pulses):
    s = np.zeros((T_MS, 2), np.float32)
    for j, p in enumerate(("L1", "L23")):
        for t, n in pulses[p]:
            s[int(t), j] += n
    return s


def _init(cell_dir, g_ampa, seed, nmda_ratio=1.0):
    global _cell, NMDA_RATIO
    NMDA_RATIO = nmda_ratio
    _cell = EvokedCell(cell_dir, seed)
    _cell.g_ampa = dict(g_ampa)


def _work(args):
    i, seed, kind, dc, controls = args
    rng = np.random.default_rng(seed)
    pulses = protocol(rng, kind)
    ev = releases(rng, pulses)
    out = dict(i=i, stim=stim_array(pulses), dc=dc, kind=kind, V=_cell.run(ev, dc))
    if controls:
        out["V_nmda_block"] = _cell.run(ev, dc, nmda=False)
        ev2 = releases(np.random.default_rng(seed + 7919), pulses)     # same pulses, new release draws
        out["V_repeat"] = _cell.run(ev2, dc)
    return out


def main():
    global NMDA_RATIO, G_MAX, CAL_MV
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--chimera-soma", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--controls", type=int, default=20, help="first N traces also run with NMDA blocked and with a second release draw")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nmda-ratio", type=float, default=NMDA_RATIO, help="NMDA/AMPA peak conductance (0 = AMPA only)")
    ap.add_argument("--g-max", type=float, default=None, help="cap per-synapse AMPA conductance (uS) during calibration")
    ap.add_argument("--cal-mv", type=float, default=None, help="target EPSP of a full-intensity pulse (mV)")
    ap.add_argument("--g-json", default=None, help="json {L1: uS, L23: uS}: skip the calibration")
    args = ap.parse_args()
    NMDA_RATIO = args.nmda_ratio
    G_MAX = args.g_max
    CAL_MV = args.cal_mv or CAL_MV
    cell_dir = PERI[args.cell] if args.cell in PERI else os.path.join(MODELS, args.cell) if not args.chimera_soma else merged_dir(args.cell, args.chimera_soma)
    tag = args.cell if not args.chimera_soma else f"{args.cell}_soma{args.chimera_soma}"
    out = os.path.join(args.out, tag); os.makedirs(out, exist_ok=True)
    t0 = time.time()
    syn_seed = int(args.cell)     # same synapse sites for a cell and its chimeras
    c = EvokedCell(cell_dir, syn_seed)
    dmax, sites = c.dmax, c.sites
    if args.g_json:                       # stimulus strengths given (e.g. a chimera uses its parent's)
        g = {k: float(v) for k, v in json.load(open(args.g_json)).items()}
    elif args.chimera_soma:
        g = json.load(open(os.path.join(args.out, args.cell, "meta.json")))["g_ampa_uS"]
    else:
        g = {k: float(v) for k, v in c.calibrate().items()}
    print(tag, "g_AMPA (uS)", g, f"{time.time()-t0:.0f}s", flush=True)
    rheo = RHEO.get(args.chimera_soma or args.cell, 150.0)
    if args.chimera_soma:
        args.seed += 100          # new stimulus patterns, never seen when the parents were fitted
    rng = np.random.default_rng(args.seed)
    kinds = ["poisson", "trains", "single"]
    jobs = []
    for i in range(args.n):
        kind = kinds[i % 3]
        dc = float(rng.choice([0.0, rng.uniform(0.4, 1.1)], p=[0.3, 0.7]) * rheo)
        jobs.append((i, args.seed * 100000 + i, kind, dc, i < args.controls))
    del c      # workers are fresh processes (spawn): NEURON state must not be inherited from this one
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cell_dir, g, syn_seed, NMDA_RATIO)) as pool:
        res = []
        for r in pool.imap_unordered(_work, jobs):
            res.append(r)
            if len(res) % 10 == 0:
                print(tag, len(res), f"{time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["i"])
    ctrl = [r for r in res if "V_nmda_block" in r]
    np.savez_compressed(os.path.join(out, "evoked.npz"), stim=np.stack([r["stim"] for r in res]),
                        dc=np.array([r["dc"] for r in res], np.float32), V=np.stack([r["V"] for r in res]),
                        kind=np.array([r["kind"] for r in res]),
                        ctrl_idx=np.array([r["i"] for r in ctrl]),
                        V_nmda_block=np.stack([r["V_nmda_block"] for r in ctrl]) if ctrl else np.zeros((0, T_MS)),
                        V_repeat=np.stack([r["V_repeat"] for r in ctrl]) if ctrl else np.zeros((0, T_MS)))
    json.dump(dict(cell=args.cell, chimera_soma=args.chimera_soma, g_ampa_uS=g, n=args.n, T_ms=T_MS, dt=DT,
                   n_syn=N_SYN, tuft_frac=TUFT_FRAC, nmda_ratio=NMDA_RATIO, p0=P0, U=U, tau_rec=TAU_REC,
                   latency=LATENCY, cal_mV=CAL_MV, g_max_uS=G_MAX, rheobase_pA=rheo, apical_max_um=dmax,
                   site_dist_um=sites, wall_s=time.time() - t0), open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(tag, "done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
