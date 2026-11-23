"""Build an Allen Cell Types *all-active* model in NEURON, reproducing
allensdk.model.biophysical.utils.AllActiveUtils.

axon_type="stub" (default here): reconstructed axon replaced by 2 x 30 um, 1 um diameter
sections at soma(0.5), as allensdk does for stub-axon models. The Nandi et al. all-active
workflow (github.com/AllenInstitute/All-active-Workflow, ateamopt) fits with
axon_type "stub_axon" by default, which is why it is used here.
axon_type="truncated": allensdk's legacy option, 2 x 30 um with diameters read from the SWC,
connected at soma(1.0).

Then Ra and pas everywhere; every genome entry (including
cm, g_pas and e_pas per section list) applied per section list; ena/ek set where the ion
exists; celsius 34. Voltage convention matches allensdk: v_reported = v_sim - junction_potential
(junction_potential = -14 mV).

Model dir layout: <cell>/reconstruction.swc, <cell>/fit_parameters.json; compiled mechanisms
in <mech_dir>/x86_64.
"""
import json
import os

import numpy as np
from neuron import h


def _load_mechs(mech_dir):
    if hasattr(h, "NaTs2_t"):
        return
    for p in ("x86_64/.libs/libnrnmech.so", "x86_64/libnrnmech.so"):
        f = os.path.join(mech_dir, p)
        if os.path.exists(f):
            h.nrn_load_dll(f)
            return
    raise FileNotFoundError("compile modfiles first: cd %s && nrnivmodl modfiles" % mech_dir)


def _stub_axon():
    # allensdk Utils.generate_morphology (used by AllActiveUtils for axon_type="stub")
    h("soma[0] area(0.5)")
    for sec in list(h.allsec()):
        sec.nseg = 1 + 2 * int(sec.L / 40.0)
        if sec.name()[:4] == "axon":
            h.delete_section(sec=sec)
    h("create axon[2]")
    for sec in h.axon:
        sec.L = 30
        sec.diam = 1
        sec.nseg = 1 + 2 * int(sec.L / 40.0)
    h.axon[0].connect(h.soma[0], 0.5, 0.0)
    h.axon[1].connect(h.axon[0], 1.0, 0.0)
    h.define_shape()
    return [1.0, 1.0]


def _truncated_axon():
    # allensdk AllActiveUtils.generate_morphology, axon_type="truncated" (legacy models)
    for sec in h.allsec():
        sec.nseg = 1 + 2 * int(sec.L / 40.0)
    h("soma[0] area(0.5)")
    axon_diams = [h.axon[0].diam, h.axon[0].diam]
    h.distance(sec=h.soma[0])
    for sec in h.allsec():
        if sec.name()[:4] == "axon":
            if h.distance(0.5, sec=sec) > 60:
                axon_diams[1] = sec.diam
                break
    for sec in list(h.allsec()):
        if sec.name()[:4] == "axon":
            h.delete_section(sec=sec)
    h("create axon[2]")
    for i, sec in enumerate(h.axon):
        sec.L = 30
        sec.diam = axon_diams[i]
    for sec in h.allsec():
        sec.nseg = 1 + 2 * int(sec.L / 40.0)
    h.axon[0].connect(h.soma[0], 1.0, 0.0)
    h.axon[1].connect(h.axon[0], 1.0, 0.0)
    h.area(0.5, sec=h.soma[0])
    return [float(d) for d in axon_diams]


def build(cell_dir, mech_dir, dt=0.025, axon_type="stub"):
    _load_mechs(mech_dir)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    desc = json.load(open(os.path.join(cell_dir, "fit_parameters.json")))

    swc = h.Import3d_SWC_read()
    swc.input(os.path.join(cell_dir, "reconstruction.swc"))
    imprt = h.Import3d_GUI(swc, 0)
    h("objref this")
    imprt.instantiate(h.this)
    axon_diams = _stub_axon() if axon_type == "stub" else _truncated_axon()

    passive = desc["passive"][0]
    cond = desc["conditions"][0]
    h("access soma")
    for sec in h.allsec():
        sec.Ra = passive["ra"]
        sec.insert("pas")
    for p in desc["genome"]:
        arr, mech, name, val = p["section"], p["mechanism"], p["name"], float(p["value"])
        if arr == "glob":
            h("%s = %.17g" % (name, val))
            continue
        if not hasattr(h, arr):
            continue
        for sec in getattr(h, arr):
            if mech != "" and h.ismembrane(str(mech), sec=sec) != 1:
                sec.insert(mech)
            setattr(sec, name, val)
    for e in cond["erev"]:
        if not hasattr(h, e["section"]):
            continue
        for sec in getattr(h, e["section"]):
            if h.ismembrane("k_ion", sec=sec) == 1:
                sec.ek = float(e["ek"])
            if h.ismembrane("na_ion", sec=sec) == 1:
                sec.ena = float(e["ena"])
    h.v_init = cond["v_init"]
    h.celsius = cond["celsius"]
    h.dt = dt
    h.steps_per_ms = 1.0 / dt
    jp = desc["fitting"][0]["junction_potential"]
    return dict(junction_potential=jp, n_sections=sum(1 for _ in h.allsec()),
                n_segments=sum(s.nseg for s in h.allsec()), v_init=cond["v_init"],
                axon_type=axon_type, axon_diams_um=axon_diams)


class Runner:
    """IClamp at soma(0.5) and steady-state initialisation; runs a current waveform.
    Same interface and conventions as ../../code/allen_cell.py (first cell)."""

    def __init__(self, cell_dir, mech_dir, dt=0.025, settle_ms=3000.0, axon_type="stub"):
        self.info = build(cell_dir, mech_dir, dt, axon_type)
        self.dt = dt
        self.stim = h.IClamp(h.soma[0](0.5))
        self.stim.delay = 0
        self.stim.dur = 1e12
        self.stim.amp = 0
        self.vrec = h.Vector()
        self.vrec.record(h.soma[0](0.5)._ref_v, 1.0)  # sample every 1 ms
        self.ivec = h.Vector([0.0, 0.0])
        self.tvec = h.Vector([0.0, 1e9])
        self.ivec.play(self.stim._ref_amp, self.tvec, 1)
        self.settle_ms = settle_ms
        self._init()
        self.rest_raw = h.soma[0](0.5).v

    def _init(self):
        # steady state at zero current: large-dt implicit steps from v_init, then t = 0
        h.finitialize(h.v_init)
        h.t = -self.settle_ms
        h.dt = 10.0
        while h.t < -5.0:
            h.fadvance()
        h.dt = self.dt
        h.t = 0.0
        h.fcurrent()
        self.vrec.resize(0)
        h.frecord_init()

    def run(self, i_pA_fine, fine_dt):
        """Current (pA) sampled at fine_dt ms, linearly interpolated onto IClamp.amp.
        Returns somatic Vm (recording convention, mV) sampled every 1 ms."""
        self.ivec.from_python(np.asarray(i_pA_fine, float) * 1e-3)  # nA
        self.tvec.from_python(np.arange(len(i_pA_fine)) * fine_dt)
        self._init()
        h.continuerun(len(i_pA_fine) * fine_dt)
        return np.array(self.vrec) - self.info["junction_potential"]
