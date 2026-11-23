"""Fixed synaptic layer: presynaptic spikes -> somatic current, exactly as the Allen GLIF cells do it.

In the Allen V1 point network (NEST ``glif_psc``) synapses are current based. A spike that leaves
the presynaptic cell at t_s arrives after the edge delay d and adds an alpha-shaped current

    I(t) = w * (t'/tau) * exp(1 - t'/tau),   t' = t - t_s - d >= 0,

which peaks at the edge weight w (pA) at t' = tau. The time constant depends on the receptor
port of the edge: 1 = E->E 5.5 ms, 2 = I->E 8.5 ms, 3 = E->I 2.8 ms, 4 = I->I 5.8 ms. A cell
sums the currents of all its synapses linearly (no reversal potential, no shunting).

So the whole synaptic input of a cell is a fixed, parameter-free linear filter of the
presynaptic spike trains, and its output is an injected current, the input the Standard cell
model was trained on. Put this layer in front of the ANN and its trained weights are untouched.

Two forms, which agree to float precision:
  * ``alpha_current``: the closed-form sum over spikes on any time grid, then averaged into bins
    (use it for a fixed or replayed input stream),
  * ``AlphaPSC``: a stateful per-step filter (two state variables per port and cell, exact
    discretization), for circuits where the presynaptic spikes are produced as the run goes.
"""
from __future__ import annotations

import numpy as np

TAU_SYN = {1: 5.5, 2: 8.5, 3: 2.8, 4: 5.8}  # ms, Allen glif_psc receptor ports


def nest_effective_delay(delay, resolution=0.25):
    """Delay at which a NEST synapse's current actually starts, in a run at ``resolution`` ms.

    NEST rounds the delay to the time grid and the current starts one step after the spike's
    recorded time, so in the Allen network (0.25 ms steps) a 1.6 ms edge acts as 1.75 ms. Use this
    to make the layer match a NEST run exactly; checked to 0.01 pA against glif_psc's I_syn.
    """
    return np.round(np.asarray(delay, float) / resolution) * resolution + resolution


def alpha_kernel(t, tau):
    t = np.asarray(t, float)
    return np.where(t >= 0, (t / tau) * np.exp(1.0 - t / tau), 0.0)


def alpha_current(spike_times, weights, delays, taus, t_grid):
    """Summed alpha PSC current (pA) at the times ``t_grid`` (ms).

    spike_times, weights, delays, taus: one entry per (spike, synapse) pair. A presynaptic spike
    that reaches the cell through k synapses appears k times, once per synapse.
    """
    t_grid = np.asarray(t_grid, float)
    out = np.zeros_like(t_grid)
    arrival = np.asarray(spike_times, float) + np.asarray(delays, float)
    w = np.broadcast_to(np.asarray(weights, float), arrival.shape)
    tau = np.broadcast_to(np.asarray(taus, float), arrival.shape)
    for a, wk, tk in zip(arrival, w, tau):
        lo = np.searchsorted(t_grid, a)
        hi = np.searchsorted(t_grid, a + 25 * tk)  # kernel < 1e-8 of peak after 25 tau
        out[lo:hi] += wk * alpha_kernel(t_grid[lo:hi] - a, tk)
    return out


def binned_current(spike_times, weights, delays, taus, t_stop, bin_ms=1.0, sub=20):
    """Mean current in each ``bin_ms`` bin from 0 to t_stop: what the ANN reads at 1 ms steps."""
    n = int(round(t_stop / bin_ms))
    fine = (np.arange(n * sub) + 0.5) * (bin_ms / sub)
    return alpha_current(spike_times, weights, delays, taus, fine).reshape(n, sub).mean(1)


class AlphaPSC:
    """Stateful alpha-PSC filter for ``n_cells`` targets and the four Allen receptor ports.

    Call ``step(drive)`` once per time step of length ``dt``: ``drive[port]`` is the summed weight
    (pA) of the spikes arriving at each target during this step (delays already applied). It
    returns the summed current (n_cells,) at the end of the step. Each port is the linear system
    y1' = -y1/tau, I' = y1 - I/tau with y1 += w*e/tau per arriving spike, which NEST integrates
    the same way.
    """

    def __init__(self, n_cells: int, dt: float = 1.0, taus=TAU_SYN):
        self.dt = dt
        self.ports = sorted(taus)
        self.taus = np.array([taus[p] for p in self.ports], float)[:, None]
        self.y1 = np.zeros((len(self.ports), n_cells))
        self.I = np.zeros((len(self.ports), n_cells))
        e1 = np.exp(-dt / self.taus)
        self.P11, self.P21, self.P22 = e1, dt * e1, e1

    def reset(self):
        self.y1[:] = 0.0
        self.I[:] = 0.0

    def step(self, drive: dict) -> np.ndarray:
        """Advance one step, then add the spikes that arrive at the end of it (NEST order)."""
        self.I = self.P21 * self.y1 + self.P22 * self.I
        self.y1 = self.P11 * self.y1
        for k, p in enumerate(self.ports):
            if p in drive:
                self.y1[k] += np.asarray(drive[p], float) * np.e / self.taus[k, 0]
        return self.I.sum(0)
