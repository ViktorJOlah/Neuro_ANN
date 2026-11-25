"""Intrinsic excitability of each fitted model, measured on the model itself (not on Allen's features).

For every model in models.json:
  * f-I on 1 s current steps (0-600 pA, 10 pA): rheobase and the slope over the first 100 pA above it
  * rate under fluctuating input like the network's (OU current, tau 5 ms, mean mu, SD sigma), on a grid
    of mu: the "in-network" excitability used by the rank test

    python characterize_models.py --standard ../../Standard --out results/models
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from circuit.population import load_model  # noqa: E402


def ou(n, T, mu, sigma, tau=5.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T), np.float32)
    a = np.exp(-1.0 / tau)
    s = sigma * np.sqrt(1 - a * a)
    for t in range(1, T):
        x[:, t] = a * x[:, t - 1] + s * rng.standard_normal(n)
    return mu + x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--standard", required=True)
    p.add_argument("--models", default=os.path.join(HERE, "models.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--sigma", type=float, nargs="+", default=[30.0, 60.0])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = json.load(open(a.models))["models"]
    amps = np.arange(0, 601, 10, dtype=np.float32)
    T_pre, T_on, T_post = 200, 1000, 100
    I = np.zeros((len(amps), T_pre + T_on + T_post), np.float32)
    I[:, T_pre:T_pre + T_on] = amps[:, None]
    mus = np.arange(-50, 301, 25, dtype=np.float32)
    res = {}
    for m in spec:
        cell = load_model(os.path.join(HERE, m["dir"]), a.standard)
        _, spk = cell.simulate(I)
        f = spk[:, T_pre:T_pre + T_on].sum(1).astype(float)
        base = spk[:, :T_pre].sum(1)
        on = np.flatnonzero(f > 0)
        rheo = float(amps[on[0]]) if len(on) else np.nan
        sel = (amps >= rheo) & (amps <= rheo + 100)
        slope = float(np.polyfit(amps[sel], f[sel], 1)[0]) if sel.sum() > 2 else np.nan
        ou_rates = {}
        for sg in a.sigma:
            X = ou(len(mus) * 4, 2200, 0.0, sg, seed=1)
            X = X.reshape(len(mus), 4, -1) + mus[:, None, None]
            _, s = cell.simulate(X.reshape(len(mus) * 4, -1))
            ou_rates[str(sg)] = (s[:, 200:].reshape(len(mus), 4, -1).mean((1, 2)) * 1000).tolist()
        res[m["name"]] = dict(rheobase_pa=rheo, fi_slope_hz_per_pa=slope, fi_hz=f.tolist(),
                              spikes_before_step=int(base.sum()), ou_rates_hz=ou_rates)
        print(m["name"], f"rheobase {rheo:.0f} pA, slope {slope:.3f} Hz/pA, rate at 300 pA {f[30]:.0f} Hz, "
              f"OU(mu=100, sigma=60) {ou_rates[str(a.sigma[-1])][6]:.2f} Hz, pre-step spikes {int(base.sum())}", flush=True)
    json.dump(dict(amps_pa=amps.tolist(), ou_mu_pa=mus.tolist(), ou_tau_ms=5.0, models=res),
              open(os.path.join(a.out, "excitability.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
