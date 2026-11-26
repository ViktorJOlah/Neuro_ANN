"""Heterogeneous PC populations: several fitted Standard models in one batched PC layer.

A population file (JSON) names the fitted models and gives each PC its model:

    {"models": [{"name": "487661754", "dir": ".../Standard"},
                {"name": "318808427", "dir": ".../real_cells/results/318808427/standard"}, ...],
     "assignment": "assignment.npy",          # (n_pc,) int, index into models, PC order of the prep
     "input_scale": "input_scale.npy"}        # optional (n_pc,) float, multiplies each PC's synaptic input

A model dir is either the Standard folder (params.json + weights/front_end_cnnlstm.npz) or a per-cell
fit folder (front_end.npz + params.json or metrics.json, as written by data_budget/fit_budget.py and
real_cells/code/fit_standard.py). Fitted models already have their input rescaling folded into the
weights, so every model reads the same synaptic current in pA.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def _standard_module(standard_dir):
    sys.path.insert(0, os.path.abspath(standard_dir))
    import standard_model as sm
    return sm


def load_model(model_dir, standard_dir):
    """A StandardCell for a Standard folder or a per-cell fit folder."""
    sm = _standard_module(standard_dir)
    if os.path.exists(os.path.join(model_dir, "weights", "front_end_cnnlstm.npz")):
        return sm.load(model_dir)
    pj = os.path.join(model_dir, "params.json")
    if not os.path.exists(pj):
        pj = os.path.join(model_dir, "metrics.json")
    p = json.load(open(pj))
    net = sm.build_front_end()
    w = np.load(os.path.join(model_dir, "front_end.npz"))
    net.set_weights([w[f"w{i}"] for i in range(len(w.files))])
    return sm.StandardCell(p["izhikevich"], p["branches"], net)


IDENTITY_CORR = dict(k0=1.0, c1=0.0, tau1=2.0, c2=0.0, tau2=20.0)


def load_corr(c):
    """Membrane correction for the kernel layer: a calibration.json path, a dict, or None (identity)."""
    if c is None:
        return dict(IDENTITY_CORR)
    if isinstance(c, str):
        return json.load(open(c))["correction"]
    return c


def read_population(path):
    pop = json.load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    res = lambda f: f if os.path.isabs(f) else os.path.join(base, f)  # noqa: E731
    assign = np.load(res(pop["assignment"])).astype(np.int64)
    scale = np.load(res(pop["input_scale"])).astype(np.float32) if pop.get("input_scale") else None
    models = [dict(m, dir=res(m["dir"]), **({"corr": res(m["corr"])} if isinstance(m.get("corr"), str) else {}))
              for m in pop["models"]]
    return models, assign, scale


class MixedStandard:
    """K BatchedStandard layers, one per model, each over the PCs assigned to it."""

    def __init__(self, models, assign, standard_dir, I_first=None, jit=True, kernel=None):
        """kernel: None, or the shared kernel settings dict(E_e, E_i, V_ref); each model's membrane correction is
        read from m["corr"] (a calibration.json path or a dict), identity if missing."""
        from .pc_layer import BatchedStandard
        self.n = len(assign)
        self.groups = []
        I_first = np.zeros(self.n, np.float32) if I_first is None else np.asarray(I_first, np.float32)
        for k, m in enumerate(models):
            idx = np.flatnonzero(assign == k)
            if len(idx) == 0:
                continue
            cell = load_model(m["dir"], standard_dir)
            kern = None
            if kernel is not None:
                kern = dict(kernel, corr=load_corr(m.get("corr")))
            self.groups.append((idx, BatchedStandard(cell, len(idx), I_first=I_first[idx], jit=jit, kernel=kern)))
        assert sum(len(i) for i, _ in self.groups) == self.n, "every PC needs a model"

    def step_kernel(self, I_e, I_i):
        out = np.zeros(self.n, bool)
        for idx, layer in self.groups:
            out[idx] = layer.step_kernel(I_e[idx], I_i[idx])
        return out

    def step(self, I_t):
        I_t = np.asarray(I_t, np.float32)
        out = np.zeros(self.n, bool)
        for idx, layer in self.groups:
            out[idx] = layer.step(I_t[idx])
        return out
