"""Build Allen Cell Types perisomatic model 488462783 (specimen 487661754, Cux2 L2/3 PC)
in NEURON, reproducing allensdk.model.biophysical.utils.Utils (perisomatic branch) exactly:
SWC import, axon replaced by 2 x 30 um stub (diam 1), passive + genome per section,
ena/ek on soma, celsius 34, v_init = e_pas. Output voltage convention matches allensdk:
v_reported = v_sim - junction_potential (junction_potential = -14 mV).
"""
import json
import os

from neuron import h


def build(model_dir, dt=0.025):
    mech = os.path.join(model_dir, "x86_64", ".libs", "libnrnmech.so")
    if not os.path.exists(mech):
        mech = os.path.join(model_dir, "x86_64", "libnrnmech.so")
    if not hasattr(h, "Ih"):
        h.nrn_load_dll(mech)
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    desc = json.load(open(os.path.join(model_dir, "fit_parameters.json")))

    swc = h.Import3d_SWC_read()
    swc.input(os.path.join(model_dir, "reconstruction.swc"))
    imprt = h.Import3d_GUI(swc, 0)
    h("objref this")
    imprt.instantiate(h.this)
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

    passive = desc["passive"][0]
    cond = desc["conditions"][0]
    h("access soma")
    for sec in h.allsec():
        sec.Ra = passive["ra"]
        sec.insert("pas")
        for seg in sec:
            seg.pas.e = passive["e_pas"]
    for c in passive["cm"]:
        h('forsec "%s" { cm = %g }' % (c["section"], c["cm"]))
    for p in desc["genome"]:
        if p["mechanism"] != "":
            h('forsec "%s" { insert %s }' % (p["section"], p["mechanism"]))
        h('forsec "%s" { %s = %.17g }' % (p["section"], p["name"], p["value"]))
    for e in cond["erev"]:
        h('forsec "%s" { ek = %g }' % (e["section"], e["ek"]))
        h('forsec "%s" { ena = %g }' % (e["section"], e["ena"]))
    h.v_init = cond["v_init"]
    h.celsius = cond["celsius"]
    h.dt = dt
    h.steps_per_ms = 1.0 / dt
    jp = desc["fitting"][0]["junction_potential"]
    nsec = sum(1 for _ in h.allsec())
    nseg = sum(s.nseg for s in h.allsec())
    return dict(junction_potential=jp, n_sections=nsec, n_segments=nseg, v_init=cond["v_init"])


class Runner:
    """Holds an IClamp at soma(0.5) and a steady-state SaveState; runs a current waveform."""

    def __init__(self, model_dir, dt=0.025, settle_ms=3000.0):
        self.info = build(model_dir, dt)
        self.dt = dt
        self.stim = h.IClamp(h.soma[0](0.5))
        self.stim.delay = 0
        self.stim.dur = 1e12
        self.stim.amp = 0
        self.vrec = h.Vector()
        self.vrec.record(h.soma[0](0.5)._ref_v, 1.0)  # sample every 1 ms
        # persistent play vectors (must exist before the SaveState snapshot)
        self.ivec = h.Vector([0.0, 0.0])
        self.tvec = h.Vector([0.0, 1e9])
        self.ivec.play(self.stim._ref_amp, self.tvec, 1)
        self.settle_ms = settle_ms
        self._init()
        self.rest_raw = h.soma[0](0.5).v

    def _init(self):
        # steady-state initialisation: large-dt backward-Euler steps at zero current
        # from v_init (= e_pas, as in allensdk) for settle_ms, then reset t = 0.
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
        """i_pA_fine: current waveform in pA sampled at fine_dt (ms), played with linear
        interpolation onto IClamp.amp. Returns Vm (recording convention, mV) sampled at 1 ms,
        aligned so sample k is t = k ms after stimulus onset of the waveform."""
        import numpy as np
        self.ivec.from_python(np.asarray(i_pA_fine, float) * 1e-3)  # nA
        self.tvec.from_python(np.arange(len(i_pA_fine)) * fine_dt)
        tstop = len(i_pA_fine) * fine_dt  # one extra step so the last 1 ms sample is recorded
        self._init()
        h.continuerun(tstop)
        v = np.array(self.vrec) - self.info["junction_potential"]
        return v
