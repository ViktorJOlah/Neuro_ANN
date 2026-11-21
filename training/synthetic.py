"""Synthetic NEURON-format traces for smoke tests (no NEURON needed).

An Izhikevich regular-spiking cell driven by excitatory / inhibitory synaptic
events. Output mimics calibrate.hoc: one ``Matrix.fprint`` header per trace,
column 0 = somatic Vm (mV), remaining columns = synaptic event weights.
With ``currents=True`` it mimics the single-compartment final.txt layout
instead: vm, ik, ina, inp0, inp1 (ik/ina are crude spike-locked currents).
"""
from __future__ import annotations

import argparse

import numpy as np


def simulate(n_traces=20, trace_length=2001, n_inputs=8, rate_scale=1.0, dt=0.5, seed=0,
             currents=False):
    rng = np.random.default_rng(seed)
    a, b, c, d = 0.02, 0.2, -65.0, 8.0
    n_exc = n_inputs // 2
    weights = np.where(np.arange(n_inputs) < n_exc, 1.0, -1.0)
    out = []
    for _ in range(n_traces):
        scale = rate_scale * rng.uniform(0.5, 1.5)
        p = 0.02 * scale * dt
        ev = (rng.random((trace_length, n_inputs)) < p).astype(np.float32)
        amp = rng.uniform(0.3, 1.0, n_inputs).astype(np.float32)
        inp = ev * amp
        drive = inp @ (weights * 6.0)
        v, u, g = -65.0, b * -65.0, 0.0
        vm = np.empty(trace_length, np.float32)
        spk = np.zeros(trace_length, np.float32)
        for t in range(trace_length):
            g = g * np.exp(-dt / 5.0) + drive[t]
            for _ in range(2):  # two half steps for stability
                v += 0.5 * dt * (0.04 * v * v + 5 * v + 140 - u + g)
            u += dt * a * (b * v - u)
            if v >= 30.0:
                vm[t] = 30.0
                spk[t] = 1.0
                v, u = c, u + d
            else:
                vm[t] = v
        if currents:
            k = np.exp(-np.arange(10) / 2.0)
            ina = -np.convolve(spk, k)[:trace_length] * 50 - 0.001 * (vm + 65)
            ik = np.convolve(spk, k[::-1] * 0.5 + 0.1)[:trace_length] * 20 + 0.0005 * (vm + 65)
            exc = (inp[:, :n_exc].sum(1) > 0) * 0.002
            inh = (inp[:, n_exc:].sum(1) > 0) * 0.003
            out.append(np.column_stack([vm, ik, ina, exc, inh]).astype(np.float32))
        else:
            out.append(np.column_stack([vm, inp]).astype(np.float32))
    return out


def write_neuron_txt(path, traces, headers=True):
    with open(path, "w") as f:
        for tr in traces:
            if headers:
                f.write(f"{tr.shape[0]} {tr.shape[1]}\n")
            for row in tr:
                f.write(" ".join(f"{x:g}" for x in row) + " \n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--traces", type=int, default=20)
    ap.add_argument("--length", type=int, default=2001)
    ap.add_argument("--inputs", type=int, default=8)
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--currents", action="store_true")
    a = ap.parse_args()
    tr = simulate(a.traces, a.length, a.inputs, a.rate_scale, seed=a.seed, currents=a.currents)
    write_neuron_txt(a.out, tr, headers=not a.currents)


if __name__ == "__main__":
    main()
