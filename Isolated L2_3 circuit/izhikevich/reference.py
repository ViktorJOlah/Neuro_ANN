"""NumPy reference: the published Izhikevich (2003) model, transcribed line by line.

Source: E. M. Izhikevich, "Simple Model of Spiking Neurons", IEEE Trans. Neural
Networks 14(6):1569-1572, 2003. The MATLAB listing in the paper (and spnet.m)
integrates, for every 1 ms step:

    fired = find(v >= 30);                     % spikes are detected first
    v(fired) = c(fired);                       % reset
    u(fired) = u(fired) + d(fired);
    v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I);   % two 0.5 ms steps for v
    v = v + 0.5*(0.04*v.^2 + 5*v + 140 - u + I);   % "for numerical stability"
    u = u + a.*(b.*v - u);                     % u uses the updated v

with v = -65 and u = b.*v at t = 0. ``simulate`` does exactly this, in the same
floating-point operation order, so it is the ground truth the Keras layer is
tested against. ``dt`` and ``n_substeps`` generalize the step; the defaults
(dt = 1 ms, 2 substeps) are the published scheme.
"""
from __future__ import annotations

import numpy as np

V_PEAK = 30.0

# Fig. 2 of Izhikevich (2003): (a, b, c, d) of the canonical cortical/thalamic classes.
PRESETS = {
    "RS": (0.02, 0.2, -65.0, 8.0),    # regular spiking
    "IB": (0.02, 0.2, -55.0, 4.0),    # intrinsically bursting
    "CH": (0.02, 0.2, -50.0, 2.0),    # chattering
    "FS": (0.1, 0.2, -65.0, 2.0),     # fast spiking
    "LTS": (0.02, 0.25, -65.0, 2.0),  # low-threshold spiking
    "TC": (0.02, 0.25, -65.0, 0.05),  # thalamo-cortical
    "RZ": (0.1, 0.26, -65.0, 2.0),    # resonator
}


def simulate(I, a, b, c, d, v0=-65.0, u0=None, dt=1.0, n_substeps=2, v_peak=V_PEAK,
             dtype=np.float64):
    """Run the published model.

    I: injected current, shape (T,) or (T, n_cells), in the paper's units.
    a, b, c, d: scalars or arrays of shape (n_cells,).
    Returns (v, spikes), each (T, n_cells): v after the step's integration (not
    clipped, as in the paper) and spikes[t] = v[t] >= v_peak. The paper's loop
    records the same spike one iteration later, when it resets it.
    """
    I = np.asarray(I, dtype=dtype)
    if I.ndim == 1:
        I = I[:, None]
    T, n = I.shape
    a, b, c, d = (np.broadcast_to(np.asarray(p, dtype=dtype), (n,)).copy() for p in (a, b, c, d))
    v = np.full(n, v0, dtype=dtype)
    u = b * v if u0 is None else np.broadcast_to(np.asarray(u0, dtype=dtype), (n,)).copy()
    h = dtype(dt) / dtype(n_substeps)
    dt = dtype(dt)
    v_out = np.empty((T, n), dtype=dtype)
    for t in range(T):
        fired = v >= v_peak
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        for _ in range(n_substeps):
            v = v + h * (dtype(0.04) * v ** 2 + dtype(5) * v + dtype(140) - u + I[t])
        u = u + dt * (a * (b * v - u))
        v_out[t] = v
    return v_out, v_out >= v_peak
