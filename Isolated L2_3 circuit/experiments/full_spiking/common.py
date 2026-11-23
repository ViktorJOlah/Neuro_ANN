"""Shared pieces for the full-dataset (spiking) experiments: data split, models, simulation, metrics.

Spike-fidelity metrics (continuous and one-step predictions, after the evaluation start):
  * Coincidence factor Gamma (Kistler et al. 1997; Jolivet et al. 2008), window +-4 ms:
    1 = every spike matched, 0 = chance level. Averaged over traces where either the ground
    truth or the model spikes.
  * Explained variance of spike trains smoothed with a 10 ms Gaussian. This is the measure
    Teeter et al. 2018 (Allen Institute GLIF models, Nat. Commun. 9:709) used for spike-time
    fidelity; they reported it as a ratio to the cell's own repeat-to-repeat variance. Here the
    ground truth is a deterministic simulation, so the raw EV is reported, pooled over traces.
  * Spike-count error: mean |N_model - N_true| per trace, and its signed mean (bias).
  * Vm RMSE with both traces clipped at -20 mV, so the action-potential peak is excluded;
    that is not resolvable at 1 ms sampling.
Ground-truth spikes are upward crossings of -20 mV. Model spikes are the steps at which v
reaches the Izhikevich peak (v >= 30), which is where the cell resets.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)

V_CLIP = -20.0        # mV, Vm clipped here for the Vm loss and RMSE; also the GT spike threshold
GAMMA_WIN = 4         # ms
EV_SIGMA = 10.0       # ms
PROTOS = ("square", "ramp", "noise")


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("izh_base", os.path.join(EXP, "izhikevich_subthreshold", "run.py"))
sys.modules["run"] = base
dense3 = load_module("dense3", os.path.join(EXP, "dense3_izhikevich_subthreshold", "run.py"))
seq = load_module("seq", os.path.join(EXP, "cnnlstm_izhikevich_subthreshold", "run.py"))
SUB = os.path.join(EXP, "expanded_subthreshold", "results")


def gt_spikes(V):
    """Boolean (n, T): first sample of each upward crossing of V_CLIP."""
    s = np.zeros(V.shape, bool)
    s[:, 1:] = (V[:, 1:] > V_CLIP) & (V[:, :-1] <= V_CLIP)
    return s


def load_split(seed=0):
    """All 3000 traces. Spike-free square/noise traces keep the earlier split (same test set);
    spiking traces and ramps are split 70/15/15 by trace with their own permutation."""
    sub = base.load(seed)
    rng = np.random.default_rng(seed + 100)
    sets = {}
    for proto in PROTOS:
        x = np.load(os.path.join(base.DATA, f"{proto}.npy"), mmap_mode="r")
        par = pd.read_csv(os.path.join(base.DATA, f"{proto}_params.csv"))
        spk = rng.permutation(np.flatnonzero(par.n_spikes.values > 0))
        n_tr, n_va = int(0.7 * len(spk)), int(0.15 * len(spk))
        parts = {"train": spk[:n_tr], "val": spk[n_tr:n_tr + n_va], "test": spk[n_tr + n_va:]}
        for split in parts:
            ids = parts[split]
            if (proto, split) in sub:
                ids = np.concatenate([ids, sub[(proto, split)]["ids"]])
            ids = np.sort(ids)
            sets[(proto, split)] = dict(ids=ids, I=np.asarray(x[ids, :, 0], np.float32),
                                        V=np.asarray(x[ids, :, 1], np.float32),
                                        n_spikes=par.n_spikes.values[ids])
    return sets


# ---------------------------------------------------------------- models
MODEL_NAMES = ("izh", "dense3", "cnnlstm", "dense_cnnlstm")
LABELS = {"izh": "Izhikevich alone", "dense3": "3×32 dense → Izh", "cnnlstm": "CNN-LSTM → Izh",
          "dense_cnnlstm": "dense + CNN-LSTM → Izh"}
COLORS = {"izh": "#e67e22", "dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}


class Wrapped:
    """Uniform interface: forward(I, V_start) -> (Vm, spikes) in TF; current(I) in NumPy; params()."""

    def __init__(self, name, weights_dir=None):
        self.name = name
        if name == "izh":
            src = os.path.join(weights_dir or SUB, "izh", "metrics.json")
            p = json.load(open(src))["params"]
            self.m = base.Readout(p, ("V0", "Vs", "a", "time_scale"))
            self.m(tf.zeros((1, 10)))
        elif name == "dense3":
            self.m = dense3.DenseIzhikevich(json.load(open(dense3.BASE_FIT))["params"])
            self.m(tf.zeros((1, 10)))
        else:
            self.m = seq.SeqIzhikevich(json.load(open(seq.BASE_FIT))["params"], name)
            self.m(tf.zeros((1, 10)))
        if name != "izh":
            self.m.load_weights(os.path.join(weights_dir or SUB, name, "model.weights.h5"))
        self.izh = self.m.izh
        self.izh.output_mode = "both"
        cell = self.izh.cell
        for w in (cell.c, cell.d):  # resets matter once the cell spikes
            w.trainable = True

    @property
    def variables(self):
        return self.m.trainable_variables

    def _x(self, I):
        if self.name == "izh":
            return I[..., None] + self.m.I0  # the layer's input_gain multiplies by g
        return self.m.current(I)

    def forward(self, I, V_start):
        cell = self.izh.cell
        v0 = ((V_start - self.m.V0) / self.m.Vs)[:, None]
        state = [v0, tf.cast(cell.b, v0.dtype)[None, :] * v0]
        v, s = self.izh(self._x(I), initial_state=state)
        return v[..., 0] * self.m.Vs + self.m.V0, s[..., 0]

    def current(self, I):
        if self.name == "izh":
            return None
        return self.m.current(tf.constant(I)).numpy()[..., 0].astype(np.float64)

    def params(self):
        if self.name == "izh":
            return base.params_of(self.m)
        mod = dense3 if self.name == "dense3" else seq
        return mod.params_of(self.m)

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        if self.name != "izh":
            self.m.save_weights(os.path.join(out, "model.weights.h5"))
        with open(os.path.join(out, "params.json"), "w") as f:
            json.dump(self.params(), f, indent=2)


def load_trained(name, weights_dir):
    w = Wrapped(name)
    if name == "izh":
        p = json.load(open(os.path.join(weights_dir, "izh", "params.json")))
        w.m.izh.input_gain.assign([p["g"]])
        for k in ("I0", "V0", "Vs"):
            getattr(w.m, k).assign(p[k])
    else:
        w.m.load_weights(os.path.join(weights_dir, name, "model.weights.h5"))
    c = w.izh.cell
    if name == "izh":
        for k in ("a", "b", "c", "d", "time_scale"):
            getattr(c, k).assign([p[k]])
    return w


# ---------------------------------------------------------------- simulation + metrics
def predict(w, s):
    """One-step and continuous predictions (NumPy, float64) and model spike rasters."""
    p = w.params()
    x = w.current(s["I"])
    out = {}
    for kind, kw in (("one", dict(one_step=True)), ("cont", dict(clamp_until=base.SEED_MS))):
        vm = base.simulate_np(s["I"], p, s["V"], x=x, **kw)
        spk = vm >= 30.0 * p["Vs"] + p["V0"]     # v >= 30 in model units
        out[kind] = dict(vm=vm.astype(np.float32), spikes=spk)
    return out


def gamma(t_true, t_model, T, win=GAMMA_WIN):
    n_d, n_m = len(t_true), len(t_model)
    if n_d == 0 and n_m == 0:
        return np.nan
    if n_d == 0 or n_m == 0:
        return 0.0
    coinc, j = 0, 0
    used = np.zeros(n_m, bool)
    for t in t_true:
        k = np.flatnonzero(~used & (np.abs(t_model - t) <= win))
        if len(k):
            used[k[np.argmin(np.abs(t_model[k] - t))]] = True
            coinc += 1
    nu = n_m / T
    expected = 2 * nu * win * n_d
    norm = 1 - 2 * nu * win
    return float((coinc - expected) / (0.5 * (n_d + n_m)) / norm) if norm > 0 else np.nan


def gauss_smooth(s, sigma=EV_SIGMA):
    k = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    g = np.exp(-0.5 * (k / sigma) ** 2)
    return np.array([np.convolve(row.astype(float), g, mode="same") for row in s])


def spike_metrics(true_spk, model_spk, start):
    ts, ms = true_spk[:, start:], model_spk[:, start:]
    T = ts.shape[1]
    gam = [gamma(np.flatnonzero(a), np.flatnonzero(b), T) for a, b in zip(ts, ms)]
    gam = np.array([g for g in gam if not np.isnan(g)])
    st, sm = gauss_smooth(ts), gauss_smooth(ms)
    ev = 1 - np.var(sm - st) / np.var(st) if np.var(st) > 0 else np.nan
    dn = ms.sum(1).astype(int) - ts.sum(1).astype(int)
    return dict(gamma=float(gam.mean()) if len(gam) else np.nan, ev=float(ev),
                count_abs_err=float(np.abs(dn).mean()), count_bias=float(dn.mean()),
                n_true=int(ts.sum()), n_model=int(ms.sum()))


def vm_rmse(vm, V, start):
    e = np.minimum(vm, V_CLIP)[:, start:] - np.minimum(V, V_CLIP)[:, start:]
    return float(np.sqrt(np.mean(e ** 2)))


def evaluate(w, sets, split="test"):
    res, preds = {}, {}
    for proto in PROTOS:
        s = sets[(proto, split)]
        pr = predict(w, s)
        preds[proto] = pr
        ts = gt_spikes(s["V"])
        res[proto] = {}
        for kind, start in (("one", base.SKIP_MS), ("cont", base.SEED_MS)):
            res[proto][kind] = dict(vm_rmse_mV=vm_rmse(pr[kind]["vm"], s["V"], start),
                                    **spike_metrics(ts, pr[kind]["spikes"], start))
    return res, preds
