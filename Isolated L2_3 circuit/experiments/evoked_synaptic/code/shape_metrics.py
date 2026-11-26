"""EPSP smoothness and shape metrics on held-out evoked traces (model vs NEURON).

Per isolated event (same selection as peak_metrics.py: no other pulse from -100 to +150 ms, no recorded spike
from -20 to +150 ms, model spikes clipped at -20 mV), window 0-60 ms (L2/3) or 0-100 ms (L1), baseline = mean of
the 10 ms before the pulse:
  shape_rmse       RMSE of the baseline-subtracted waveform (mV)
  shape_rmse_norm  same after scaling each waveform to peak 1 (pure shape, amplitude removed)
  extra_peaks      local maxima with prominence >= 0.1 mV in the model minus in NEURON (NEURON has 1 per EPSP)
  pre_dip          most negative deflection 0-4 ms after the pulse (mV; NEURON ~0)
  ratio, tpk       mean peak ratio and median peak time (ms)
Whole free-running trace, away from spikes (same mask as fit_input_transform.score):
  rough_ratio      RMS second difference of Vm (mV/ms^2), model / NEURON  (1 = as smooth as NEURON)
  extrema_per_s    local maxima + minima (prominence >= 0.05 mV) per second, model and NEURON
  spurious_extrema_per_s  model extrema with no NEURON extremum of the same kind within 3 ms (wiggles, extra peaks)

    python shape_metrics.py <results_dir> [<results_dir> ...]   -> shape.json in each
"""
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = {0: 100, 1: 60}
PROM = 0.1
SPUR_PROM = 0.05
T0 = 250


def iso_events(V, stim, test):
    for k, i in enumerate(test):
        for j in (0, 1):
            for t in np.nonzero(stim[i, :, j] > 0)[0]:
                if t < 400 or t + 150 > V.shape[1]:
                    continue
                w = stim[i, t - 100:t + 150].copy(); w[100, j] = 0
                if w.sum() > 0 or (V[i, t - 20:t + 150] > -20).any():
                    continue
                yield k, i, j, t


def npeaks(x):
    return len(find_peaks(x, prominence=PROM)[0])


def compute(V, stim, vm, test):
    rows = []
    for k, i, j, t in iso_events(V, stim, test):
        n = WIN[j]
        g = V[i, t:t + n] - V[i, t - 10:t].mean()
        m = np.minimum(vm[k, t:t + n], -20.0) - vm[k, t - 10:t].mean()
        gn, mn = g / max(g.max(), 1e-3), m / max(m.max(), 1e-3)
        rows.append(dict(path=j, shape_rmse=float(np.sqrt(np.mean((m - g) ** 2))),
                         shape_rmse_norm=float(np.sqrt(np.mean((mn - gn) ** 2))),
                         peaks_gt=npeaks(g), peaks_model=npeaks(m),
                         pre_dip=float(min(m[:4].min(), 0.0)), pre_dip_gt=float(min(g[:4].min(), 0.0)),
                         peak=float(g.max()), peak_model=float(m.max()), tpk=int(g.argmax()), tpk_model=int(m.argmax())))
    out = {}
    for j, p in ((0, "L1"), (1, "L23")):
        r = [x for x in rows if x["path"] == j]
        if not r:
            continue
        f = lambda key: np.array([x[key] for x in r], float)
        out[p] = dict(n=len(r), shape_rmse=float(f("shape_rmse").mean()), shape_rmse_norm=float(f("shape_rmse_norm").mean()),
                      extra_peaks=float((f("peaks_model") - f("peaks_gt")).mean()),
                      frac_multi_peak_model=float((f("peaks_model") > 1).mean()), frac_multi_peak_gt=float((f("peaks_gt") > 1).mean()),
                      pre_dip=float(f("pre_dip").mean()), pre_dip_gt=float(f("pre_dip_gt").mean()),
                      ratio=float(f("peak_model").mean() / f("peak").mean()),
                      tpk=float(np.median(f("tpk"))), tpk_model=float(np.median(f("tpk_model"))))
    # whole-trace smoothness away from spikes
    Vt, vt = V[test][:, T0:], np.minimum(vm[:, T0:], -20.0)
    away = np.ones_like(Vt, bool)
    for X in (Vt, vm[:, T0:]):
        for a, b in zip(*np.nonzero((X[:, 1:] > -20) & (X[:, :-1] <= -20))):
            away[a, max(b - 5, 0):b + 30] = False
    a2 = away[:, 2:] & away[:, 1:-1] & away[:, :-2]
    d2 = lambda X: X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
    rg, rm = np.sqrt(np.mean(d2(Vt)[a2] ** 2)), np.sqrt(np.mean(d2(vt)[a2] ** 2))
    ext_g = ext_m = spur = 0
    for X, Y, A in zip(Vt, vt, away):
        for s in np.split(np.arange(len(A)), np.nonzero(np.diff(A.astype(int)))[0] + 1):
            if A[s[0]] and len(s) > 20:
                for sign in (1, -1):
                    pg = find_peaks(sign * X[s], prominence=SPUR_PROM)[0]
                    pm = find_peaks(sign * Y[s], prominence=SPUR_PROM)[0]
                    ext_g += len(pg); ext_m += len(pm)
                    spur += sum(1 for q in pm if not len(pg) or np.min(np.abs(pg - q)) > 3)
    sec = away.sum() / 1000.0
    out["trace"] = dict(rough_gt=float(rg), rough_model=float(rm), rough_ratio=float(rm / rg),
                        extrema_per_s_gt=float(ext_g / sec), extrema_per_s_model=float(ext_m / sec),
                        spurious_extrema_per_s=float(spur / sec))
    return out, rows


def run_dir(d, data_set="data_ampa"):
    m = json.load(open(os.path.join(d, "metrics.json")))
    z = np.load(os.path.join(HERE, "..", data_set, m["cell"], "evoked.npz"))
    p = np.load(os.path.join(d, "test_predictions.npz"))
    s, rows = compute(z["V"], z["stim"], p["vm"], p["test"])
    json.dump(dict(summary=s, events=rows), open(os.path.join(d, "shape.json"), "w"))
    return s


if __name__ == "__main__":
    for d in sys.argv[1:]:
        s = run_dir(d)
        print(d.split("results_ampa/")[-1], json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in s.items()}))
