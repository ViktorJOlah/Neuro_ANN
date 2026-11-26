"""Orientation-tuned, grating-locked excitatory drive (Smith et al. 2013-style visual stimulation).

The Billeh V1 inputs onto one L2/3 cell (build_inputs.py) are dominated by the 1 kHz background and a few
LGN edges: almost all L2/3 E and L4 E edges onto it have syn_weight ~0 in the released network, so they add
no stimulus structure. Smith et al. 2013 saw orientation-tuned, grating-locked NMDA bursts in every distal
dendrite, which needs co-active tuned input. This module adds it as a separate, documented stream.

  * n_inputs presynaptic axons, each with n_contacts synapses, preferred orientation theta_i drawn from a
    von Mises around the cell's preferred orientation (concentration kappa_pop; Jia et al. 2010: inputs
    of all orientations, biased to the cell's preference).
  * rate_i(t) = r0 + r1 * exp(kappa_tune * (cos(2 (ori - theta_i)) - 1)) * m(t) during the grating,
    with m(t) = (1 + cos(2 pi f t - phi_i)) / 2 (drift at f Hz; phi_i = common phase + N(0, phase_sd)),
    so inputs with similar preference fire together on each grating cycle. r0 during grey screen.
  * Feedforward inhibition (Pvalb-like, untuned): inh_n perisomatic GABA-A inputs (soma and dendrite
    < 50 um) with rate inh_r0 + inh_r1 * m(t) during the grating (same drift modulation, common phase).
  * Functional clustering (Iacaruso et al. 2017; Smith model 'signal synapses'): a fraction cluster_frac
    of axons put all their contacts within span_um on one branch; axons are sorted by theta_i before
    being assigned to branches, so co-tuned inputs share branches.
"""
from __future__ import annotations

import numpy as np
from neuron import h

DEFAULT = dict(n_inputs=150, n_contacts=2, g_uS=0.4e-3, r0=0.5, r1=20.0, kappa_tune=2.0, kappa_pop=1.0,
               pref_deg=0.0, f_hz=2.0, phase_sd=0.6, cluster_frac=0.6, n_branches=30, span_um=25.0,
               min_d=40.0, t_on=500.0, t_off=2500.0,
               inh_n=0, inh_g_uS=1.0e-3, inh_r0=2.0, inh_r1=20.0, nmda_ratio=None)


def _branch_sites(cell, min_d):
    segs = []
    for sec in cell.dends:
        if sec in cell.trunk or cell._is_tuft(sec, 0.5):
            continue
        for seg in sec:
            if h.distance(seg.x, sec=sec) >= min_d:
                segs.append((sec, seg.x, sec.L / sec.nseg))
    return segs


def add(cell, ori_deg, cfg=None, seed=0, T=3000.0):
    """Create the synapses and return [(netcons, event_times)] for run_invivo-style queuing."""
    c = dict(DEFAULT); c.update(cfg or {})
    rng = np.random.default_rng(seed + 7)
    n = c["n_inputs"]
    theta = (c["pref_deg"] + np.degrees(rng.vonmises(0.0, c["kappa_pop"], n)) / 2.0) % 180.0
    phi = rng.normal(0.0, c["phase_sd"], n)
    segs = _branch_sites(cell, c["min_d"])
    w = np.array([s[2] for s in segs]); w /= w.sum()
    # branch assignment for clustered axons: sort by preference, deal contiguous groups to branches
    n_cl = int(round(c["cluster_frac"] * n))
    cl_idx = rng.choice(n, n_cl, replace=False)
    cl_sorted = cl_idx[np.argsort(theta[cl_idx])]
    centres = [segs[i][:2] for i in rng.choice(len(segs), c["n_branches"], replace=False, p=w)]
    branch_of = {int(a): (k * c["n_branches"]) // max(n_cl, 1) for k, a in enumerate(cl_sorted)}
    dt = 1.0
    tt = np.arange(0, T, dt)
    stim = (tt >= c["t_on"]) & (tt < c["t_off"])
    keep = []
    for i in range(n):
        tune = np.exp(c["kappa_tune"] * (np.cos(np.radians(2 * (ori_deg - theta[i]))) - 1))
        m = (1 + np.cos(2 * np.pi * c["f_hz"] * (tt - c["t_on"]) / 1000.0 - phi[i])) / 2
        rate = c["r0"] + stim * c["r1"] * tune * m
        spikes = tt[rng.random(len(tt)) < rate * dt / 1000.0]
        for _ in range(c["n_contacts"]):
            if i in branch_of:
                sec, x0 = centres[branch_of[i]]
                x = float(np.clip(x0 + rng.uniform(-0.5, 0.5) * c["span_um"] / sec.L, 0.01, 0.99))
            else:
                sec, x, _ = segs[rng.choice(len(segs), p=w)]
            syn = cell.add_synapse(sec, x, g_ampa=c["g_uS"], nmda_ratio=c["nmda_ratio"])
            syn["src"] = "visual"
            keep.append((cell.drive(syn, None), spikes + rng.uniform(0, 1.0, len(spikes))))
    peri = [(cell.soma[0], 0.5)] * 5 + [(sec, seg.x) for sec in cell.dends for seg in sec
                                        if h.distance(seg.x, sec=sec) <= 50]
    for j in range(c["inh_n"]):
        m = (1 + np.cos(2 * np.pi * c["f_hz"] * (tt - c["t_on"]) / 1000.0 - rng.normal(0, c["phase_sd"]))) / 2
        rate = c["inh_r0"] + stim * c["inh_r1"] * m
        spikes = tt[rng.random(len(tt)) < rate * dt / 1000.0] + rng.uniform(0, 1.0)
        sec, x = peri[rng.integers(len(peri))]
        syn = cell.add_synapse(sec, x, g_ampa=c["inh_g_uS"], kind="inh")
        syn["src"] = "visual inh"
        keep.append((cell.drive(syn, None), spikes))
    return keep, dict(theta=theta, centres=centres)
