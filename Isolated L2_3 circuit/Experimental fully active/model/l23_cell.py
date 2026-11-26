"""Experimental fully active mouse L2/3 pyramidal cell (NEURON).

A morphologically detailed L2/3 PC whose dendritic channels are set from L2/3 data where they exist
(the knowledge base is Viktor's review "What we really know about layer 2/3 pyramidal cells?", v3),
and from the least L5-specific choice where they do not. Every density lives in params.json so that
each assumption is written down once. The main departures from the inherited (L5/CA1) templates:

* HCN is weighted towards the soma, with the kinetics of Olah, Wu, Kaczmarek & Rowan (2025, eLife
  96002): Ih_Olah.mod, tau(v) = 25.46 exp(-0.5((v+100.28)/16.2)^2) ms, one activation gate. Density is
  a Gaussian centred on the soma (the paper's Fig 5 best fit: proximal centre, ~50 um half-width).
  Targets: Rin 75 -> 105 MOhm and -3.5 mV with Cs+, sag ~0.7 mV, ZD broadening of EPSPs falling with
  a ~105 um length constant.
* NMDA receptors at every excitatory synapse (AMPA + NMDA_Ca, Jahr-Stevens Mg block), with a
  proximally biased NMDA:AMPA ratio (Gaussian centred on the soma, the paper's Fig 4 best fit).
  NMDA carries a Ca2+ fraction into cai, so NMDA spikes show up in the dendritic Ca2+ readout.
* Dendritic Na+ (NaTs2_t) along the whole arbor: supports a bAP with a ~300 um length constant
  (Waters 2003) and fast dendritic Na+ spikes in distal apical branches (Gooch 2022, Larkum 2007).
* Uniform dendritic A-type K+ (Kv4, KA_L23.mod: low-threshold, m*h) as in Biro 2018 / Landau 2022; this is the
  "K+ brake": reducing it (4-AP condition) unmasks regenerative dendritic Ca2+ electrogenesis.
* VGCCs (HVA + LVA) in all dendrites with a proximal peak at ~80 um (Park 2019; bAP Ca2+ peaks
  near the soma, Waters 2003, Krieger 2009), plus dendritic SK recruited by bAP Ca2+ (Jones 2013/2017).
* Persistent Na+ on distal basal dendrites > 70 um (Ferrarese 2018 model of weak-input amplification).
* Spines are folded into cm and g_pas of spiny dendrite (> 30 um) by a factor spine_factor.

Usage
-----
    from model.l23_cell import L23Cell
    cell = L23Cell()                      # params.json defaults
    cell.block("HCN")                     # ZD7288 / Cs+ ; also "NMDA", "TTX", "4AP", "Ca", "SK", "AMPA"
    syn = cell.add_synapse(sec, x)        # AMPA + NMDA at one site
    cell.init(); h.continuerun(500)

Units: NEURON's (mV, ms, um, S/cm2, nA, uS). Voltages are raw (no junction-potential correction).
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from neuron import h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # Experimental fully active/
DEFAULT_PARAMS = os.path.join(HERE, "params.json")
MORPH_DIR = os.path.join(ROOT, "morphology")

_loaded = False


def load_mechanisms():
    global _loaded
    if _loaded or hasattr(h, "Ih_L23"):
        _loaded = True
        return
    for p in ("x86_64/.libs/libnrnmech.so", "x86_64/libnrnmech.so", "arm64/.libs/libnrnmech.so"):
        f = os.path.join(ROOT, p)
        if os.path.exists(f):
            h.nrn_load_dll(f)
            _loaded = True
            return
    raise FileNotFoundError(f'compile first: cd "{ROOT}" && nrnivmodl mechanisms')


def load_params(path=DEFAULT_PARAMS, **overrides):
    p = json.load(open(path))
    for k, v in overrides.items():
        cur = p
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur[kk]
        cur[keys[-1]] = v
    return p


class L23Cell:
    """One cell instance. Only one instance per process is supported (sections live in h)."""

    def __init__(self, params=None, morphology=None):
        load_mechanisms()
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
        self.p = copy.deepcopy(params) if params is not None else load_params()
        self.morph = morphology or os.path.join(MORPH_DIR, self.p["morphology"])
        self._build_morphology()
        self._discretise()
        self._insert_biophysics()
        self.blocked = set()
        self.synapses = []
        self.iclamps = []

    # ------------------------------------------------------------------ morphology
    def _build_morphology(self):
        for sec in list(h.allsec()):
            h.delete_section(sec=sec)
        swc = h.Import3d_SWC_read()
        swc.input(self.morph)
        imp = h.Import3d_GUI(swc, 0)
        imp.instantiate(None)
        self.soma = list(h.soma)
        self.basal = list(h.dend) if hasattr(h, "dend") else []
        self.apical = list(h.apic) if hasattr(h, "apic") else []
        for sec in list(h.allsec()):
            if sec.name().startswith("axon"):
                h.delete_section(sec=sec)
        a = self.p["axon"]
        self.hillock = h.Section(name="hillock")
        self.ais = h.Section(name="ais")
        self.myelin = h.Section(name="myelin")
        for sec, (L, d) in zip((self.hillock, self.ais, self.myelin),
                               (a["hillock_L_d"], a["ais_L_d"], a["myelin_L_d"])):
            sec.L, sec.diam = L, d
        self.hillock.connect(self.soma[0](0.5))
        self.ais.connect(self.hillock(1))
        self.myelin.connect(self.ais(1))
        self.axon = [self.hillock, self.ais, self.myelin]
        h.define_shape()
        self.dends = self.basal + self.apical
        self.trunk = self._find_trunk()
        tmin = self.p.get("trunk_min_diam_um", 0.0)
        if tmin > 0:
            for sec in self.trunk:
                for seg in sec:
                    seg.diam = max(seg.diam, tmin)

    def _find_trunk(self):
        """Main apical path: from the soma, always follow the child with the largest downstream length,
        until the section's end lies within tuft_depth_um of the top of the apical arbor."""
        def sub_len(sec):
            return sec.L + sum(sub_len(c) for c in sec.children())
        roots = [s for s in self.apical if h.SectionRef(sec=s).parent in self.soma]
        if not roots:
            return []
        sec = max(roots, key=sub_len)
        u = self._pia_axis()
        s0 = np.array([h.x3d(0, sec=self.soma[0]), h.y3d(0, sec=self.soma[0]), h.z3d(0, sec=self.soma[0])])
        hmax = max(float((np.array([s.x3d(s.n3d() - 1), s.y3d(s.n3d() - 1), s.z3d(s.n3d() - 1)]) - s0) @ u)
                   for s in self.apical)
        trunk = []
        while sec is not None:
            trunk.append(sec)
            end = np.array([sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)])
            if float((end - s0) @ u) >= hmax - self.p["tuft_depth_um"] or not sec.children():
                break
            sec = max(sec.children(), key=sub_len)
        return trunk

    def _pia_axis(self):
        s0 = np.array([h.x3d(0, sec=self.soma[0]), h.y3d(0, sec=self.soma[0]), h.z3d(0, sec=self.soma[0])])
        pts = np.array([[s.x3d(i), s.y3d(i), s.z3d(i)] for s in self.apical for i in range(s.n3d())])
        u = (pts - s0).mean(0)
        return u / np.linalg.norm(u)

    def _discretise(self):
        """d_lambda rule at 100 Hz, max 20 um per segment (odd nseg)."""
        Ra = self.p["passive"]["Ra"]
        for sec in h.allsec():
            sec.Ra = Ra
            sec.cm = self.p["passive"]["cm"]
        for sec in h.allsec():
            lam = 1e5 * np.sqrt(sec.diam / (4 * np.pi * 100.0 * Ra * sec.cm))
            n = int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1
            n = max(n, int(sec.L / 20.0) // 2 * 2 + 1)
            sec.nseg = max(1, n)
        if self.myelin.nseg > 21:
            self.myelin.nseg = 21
        h.distance(0, 0.5, sec=self.soma[0])
        # height towards the pia (for "layer" assignment); pia direction = mean apical direction
        s0 = np.array([h.x3d(0, sec=self.soma[0]), h.y3d(0, sec=self.soma[0]), h.z3d(0, sec=self.soma[0])])
        pts = []
        for sec in self.apical:
            for i in range(sec.n3d()):
                pts.append([sec.x3d(i), sec.y3d(i), sec.z3d(i)])
        pts = np.array(pts)
        u = (pts - s0).mean(0)
        self.pia_dir = u / np.linalg.norm(u)
        self.soma_xyz = s0

    def seg_xyz(self, sec, x):
        n = sec.n3d()
        arc = np.array([sec.arc3d(i) for i in range(n)])
        xyz = np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(n)])
        t = x * sec.L
        return np.array([np.interp(t, arc, xyz[:, k]) for k in range(3)])

    def height(self, sec, x):
        """Distance above the soma along the pia axis (um)."""
        return float((self.seg_xyz(sec, x) - self.soma_xyz) @ self.pia_dir)

    def segments(self, which="all"):
        """[(sec, x, path_distance_um)] for 'basal', 'apical', 'dend' (both), 'soma' or 'all'."""
        pools = {"basal": self.basal, "apical": self.apical, "dend": self.dends,
                 "soma": self.soma, "all": list(h.allsec())}[which]
        out = []
        for sec in pools:
            for seg in sec:
                out.append((sec, seg.x, h.distance(seg.x, sec=sec)))
        return out

    # ------------------------------------------------------------------ biophysics
    def _insert_biophysics(self):
        p = self.p
        pas, F = p["passive"], p["passive"]["spine_factor"]
        h.celsius = p["celsius"]
        for sec in h.allsec():
            sec.insert("pas")
            sec.e_pas = pas["e_pas"]
            sec.g_pas = pas["g_pas"]
        # spines folded into spiny dendrite beyond spine_start
        for sec in self.dends:
            for seg in sec:
                if h.distance(seg.x, sec=sec) > pas["spine_start_um"]:
                    seg.cm = pas["cm"] * F
                    seg.g_pas = pas["g_pas"] * F
        # myelin
        self.myelin.cm = p["axon"]["myelin_cm"]
        self.myelin.g_pas = p["axon"]["myelin_g_pas"]

        # mechanisms per compartment
        comp = {"soma": self.soma, "hillock": [self.hillock], "ais": [self.ais],
                "basal": self.basal, "apical": self.apical}
        for name, secs in comp.items():
            for mech in p["channels"][name]:
                for sec in secs:
                    sec.insert(mech)
        for sec in h.allsec():
            if h.ismembrane("na_ion", sec=sec):
                sec.ena = p["ena"]
            if h.ismembrane("k_ion", sec=sec):
                sec.ek = p["ek"]
        h.cao0_ca_ion = p["cao"]
        h.cai0_ca_ion = 1e-4
        self.set_densities()

    def _dist_value(self, spec, d, name, sec, seg):
        """spec: number, or dict {"type": ..., ...} describing a function of path distance d."""
        if not isinstance(spec, dict):
            return float(spec)
        t = spec["type"]
        if t == "exp":            # g0 * exp(-d / lam) (+ floor)
            return spec["g0"] * np.exp(-d / spec["lam"]) + spec.get("floor", 0.0)
        if t == "gauss":          # base + peak * exp(-(d - mu)^2 / (2 sd^2))
            return spec["base"] + spec["peak"] * np.exp(-0.5 * ((d - spec["mu"]) / spec["sd"]) ** 2)
        if t == "step":           # value beyond d0, else inner
            return spec["outer"] if d >= spec["d0"] else spec.get("inner", 0.0)
        if t == "gauss_tuft":     # proximal Gaussian bump (bAP Ca2+ peak) + a tuft value (Ca2+ zone)
            if self._is_tuft(sec, seg.x):
                return spec["tuft"]
            return spec["base"] + spec["peak"] * np.exp(-0.5 * ((d - spec["mu"]) / spec["sd"]) ** 2)
        if t == "tuft":           # value where height > tuft threshold, else inner
            return spec["outer"] if self._is_tuft(sec, seg.x) else spec.get("inner", 0.0)
        raise ValueError(t)

    def _is_tuft(self, sec, x):
        if not hasattr(self, "_hmax"):
            self._hmax = max(self.height(s, 1.0) for s in self.apical)
        return sec in self.apical and self.height(sec, x) >= self._hmax - self.p["tuft_depth_um"]

    def set_densities(self):
        p = self.p
        comp = {"soma": self.soma, "hillock": [self.hillock], "ais": [self.ais],
                "basal": self.basal, "apical": self.apical}
        for name, secs in comp.items():
            for key, spec in p["densities"][name].items():
                mech_param = key  # e.g. gbar_NaTs2_t, decay_CaDynamics, tauscale_Ih_L23
                for sec in secs:
                    for seg in sec:
                        d = h.distance(seg.x, sec=sec)
                        setattr(seg, mech_param, self._dist_value(spec, d, key, sec, seg))
        # global HCN kinetics (Ih_Olah: Olah et al. 2025 eLife)
        hm = p["hcn"]["mech"]
        for sec in h.allsec():
            if h.ismembrane(hm, sec=sec):
                for seg in sec:
                    for k, v in p["hcn"].items():
                        if k != "mech":
                            setattr(seg, f"{k}_{hm}", v)
        self._base = {}
        for sec in h.allsec():
            for seg in sec:
                for mech in seg:
                    if mech.name() in ("pas",):
                        continue
                    for attr in ("gbar",):
                        if hasattr(mech, attr):
                            self._base[(sec.name(), seg.x, mech.name())] = getattr(mech, attr)

    # ------------------------------------------------------------------ manipulations
    BLOCKS = {
        "HCN": [("Ih_Olah", 0.0), ("Ih_L23", 0.0)],
        "TTX": [("NaTs2_t", 0.0), ("NaTa_t", 0.0), ("Nap_Et2", 0.0)],
        "4AP": [("KA_L23", "fourap"), ("K_Tst", "fourap"), ("K_Pst", "fourap"), ("Kv3_1", "fourap_kv3")],
        "Ca": [("Ca_HVA", 0.0), ("Ca_LVA", 0.0)],
        "SK": [("SK", 0.0)],
        "Im": [("Im", 0.0)],
    }

    def block(self, *names, dend_only=False, **kw):
        """Pharmacology. HCN (ZD7288/Cs+), TTX, 4AP (A-type/K_P reduced to params['fourap'] of control;
        4APd = dendrites only),
        Ca (Cd2+/Ni2+), SK (apamin), Im (XE991), NMDA (APV), AMPA (NBQX). dend_only restricts channel
        blocks to dendrites (local application)."""
        for n in names:
            if n == "4APd":                  # local (dendritic) 4-AP
                self.block("4AP", dend_only=True)
                continue
            self.blocked.add(n)
            if n in ("NMDA", "AMPA"):
                self._apply_syn_weights()
                continue
            for mech, val in self.BLOCKS[n]:
                frac = self.p["fourap"] if val == "fourap" else self.p["fourap_kv3"] if val == "fourap_kv3" else val
                secs = self.dends if dend_only else list(h.allsec())
                for sec in secs:
                    if not h.ismembrane(mech, sec=sec):
                        continue
                    for seg in sec:
                        g0 = self._base[(sec.name(), seg.x, mech)]
                        setattr(seg, f"gbar_{mech}", g0 * frac)

    def unblock(self):
        for (sname, x, mech), g in self._base.items():
            pass
        for sec in h.allsec():
            for seg in sec:
                for mech in seg:
                    k = (sec.name(), seg.x, mech.name())
                    if k in self._base:
                        mech.gbar = self._base[k]
        self.blocked = set()
        self._apply_syn_weights()

    # ------------------------------------------------------------------ synapses
    def add_synapse(self, sec, x, g_ampa=None, nmda_ratio=None, kind="exc"):
        """Excitatory: AMPA (Exp2Syn) + NMDA_Ca at sec(x); inhibitory: GABA_A Exp2Syn.
        g_ampa is the peak AMPA conductance (uS); NMDA peak (unblocked) = nmda_ratio * g_ampa.
        Returns a dict; drive it with syn['netcons'] (NetCon(None, ...)) or connect(VecStim)."""
        s = self.p["synapse"]
        if kind == "exc":
            g = s["g_ampa_uS"] if g_ampa is None else g_ampa
            d = h.distance(x, sec=sec)
            r = self._dist_value(s["nmda_ratio"], d, "nmda", sec, sec(x)) if nmda_ratio is None else nmda_ratio
            am = h.Exp2Syn(sec(x)); am.tau1, am.tau2, am.e = s["ampa_tau1"], s["ampa_tau2"], 0.0
            nm = h.NMDA_Ca(sec(x)); nm.tau1, nm.tau2 = s["nmda_tau1"], s["nmda_tau2"]
            nm.cafrac = s["nmda_cafrac"]
            nm.vshift = s.get("nmda_vshift", 0.0)
            syn = dict(kind=kind, sec=sec, x=x, ampa=am, nmda=nm, g=g, ratio=r,
                       d=h.distance(x, sec=sec), pps=[am, nm])
        else:
            gb = h.Exp2Syn(sec(x)); gb.tau1, gb.tau2, gb.e = s["gaba_tau1"], s["gaba_tau2"], s["e_gaba"]
            g = s["g_gaba_uS"] if g_ampa is None else g_ampa
            syn = dict(kind=kind, sec=sec, x=x, gaba=gb, g=g, d=h.distance(x, sec=sec), pps=[gb])
        syn["netcons"] = []
        self.synapses.append(syn)
        return syn

    def drive(self, syn, source=None, delay=0.0):
        """Connect a NetCon from source (VecStim or None for manual .event()) to syn's receptors."""
        ncs = [h.NetCon(source, pp) for pp in syn["pps"]]
        for nc in ncs:
            nc.delay = delay
        syn["netcons"].extend(ncs)
        self._apply_syn_weights([syn])
        return ncs

    def _apply_syn_weights(self, syns=None):
        for syn in (syns or self.synapses):
            nc = syn["netcons"]
            if syn["kind"] == "exc":
                for k in range(0, len(nc), 2):
                    nc[k].weight[0] = 0.0 if "AMPA" in self.blocked else syn["g"]
                    nc[k + 1].weight[0] = 0.0 if "NMDA" in self.blocked else syn["g"] * syn["ratio"]
            else:
                for c in nc:
                    c.weight[0] = syn["g"]

    def clear_synapses(self):
        self.synapses = []

    # ------------------------------------------------------------------ running
    def iclamp(self, sec, x, delay, dur, amp):
        ic = h.IClamp(sec(x)); ic.delay, ic.dur, ic.amp = delay, dur, amp
        self.iclamps.append(ic)
        return ic

    def init(self, v_init=None, settle_ms=None, dt=None):
        """Initialise and settle to steady state (large-dt pre-run), then t = 0."""
        h.dt = dt or self.p["dt"]
        h.finitialize(v_init if v_init is not None else self.p["v_init"])
        settle = self.p["settle_ms"] if settle_ms is None else settle_ms
        if settle > 0:
            dt0 = h.dt
            h.t = -settle
            h.dt = 1.0
            while h.t < -h.dt:
                h.fadvance()
            h.dt = dt0
            h.t = 0
            h.fcurrent()
            h.frecord_init()
