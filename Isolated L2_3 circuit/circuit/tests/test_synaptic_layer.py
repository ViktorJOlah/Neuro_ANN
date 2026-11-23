"""The fixed synaptic layer reproduces NEST glif_psc's synaptic current."""
import numpy as np
import pytest

from circuit import synaptic_layer as sl


def test_alpha_peaks_at_weight_after_tau():
    t = np.linspace(0, 40, 40001)
    I = sl.alpha_current([0.0], [12.0], [0.0], [5.5], t)
    assert I.max() == pytest.approx(12.0, rel=1e-6)
    assert t[np.argmax(I)] == pytest.approx(5.5, abs=1e-3)


def test_stateful_filter_matches_closed_form():
    dt, T = 0.05, 400.0
    spikes = np.array([10.0, 13.5, 80.0, 81.0, 200.0])
    psc = sl.AlphaPSC(1, dt=dt)
    n = int(T / dt)
    drive = np.zeros(n)
    np.add.at(drive, np.round(spikes / dt).astype(int) - 1, 7.0)
    out = np.array([psc.step({1: drive[k:k + 1]})[0] for k in range(n)])
    ref = sl.alpha_current(spikes, 7.0, 0.0, sl.TAU_SYN[1], (np.arange(n) + 1) * dt)
    assert np.max(np.abs(out - ref)) < 1e-6


def test_matches_nest_glif_psc():
    nest = pytest.importorskip("nest")
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = 0.25
    pre = nest.Create("spike_generator", params={"spike_times": [10.0, 22.5, 24.0, 60.25]})
    parrot = nest.Create("parrot_neuron")
    post = nest.Create("glif_psc", params={"tau_syn": [5.5, 8.5, 2.8, 5.8], "V_th": 1e3})
    nest.Connect(pre, parrot)
    for port, w in ((1, 30.0), (2, -20.0)):
        nest.Connect(parrot, post, syn_spec={"weight": w, "delay": 1.6, "receptor_type": port})
    mm = nest.Create("multimeter", params={"record_from": ["I_syn"], "interval": 0.25})
    nest.Connect(mm, post)
    sr = nest.Create("spike_recorder")
    nest.Connect(parrot, sr)
    nest.Simulate(150.0)
    t, I = mm.events["times"], mm.events["I_syn"]
    s = sr.events["times"]
    d = sl.nest_effective_delay(1.6)
    ref = (sl.alpha_current(s, 30.0, d, sl.TAU_SYN[1], t) + sl.alpha_current(s, -20.0, d, sl.TAU_SYN[2], t))
    assert np.max(np.abs(ref - I)) < 0.05
