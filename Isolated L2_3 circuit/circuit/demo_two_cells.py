"""Two-cell demo of the fixed synaptic layer: a GLIF pair next to a pair of Standard ANN cells.

Cell A gets a noisy current and fires irregularly. Cell B gets a weaker noisy current of its own
plus one excitatory synapse from A (Allen L2/3 E -> L2/3 E: port 1, tau 5.5 ms, delay 1.6 ms).

  GLIF pair  both cells are the Allen GLIF model of the L2/3 PC used in the V1 network (node type
             487661754, the same Allen cell the Standard model was fitted to), simulated in NEST
             with a real NEST synapse.
  ANN pair   both cells are the Standard model. A's spikes go through ``synaptic_layer`` and the
             resulting current is added to B's injected current. The ANN's weights are untouched.

Checks: the layer's current matches NEST's own synaptic current (I_syn), and a GLIF B that gets
the layer's current through a current generator, with no synapse, spikes like the synapse-driven B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import synaptic_layer as sl

GLIF_JSON = "487661754_glif_psc.json"


def ou_current(T, mean, sd, tau, rng):
    x = np.zeros(T)
    a = np.exp(-1.0 / tau)
    for t in range(1, T):
        x[t] = a * x[t - 1] + np.sqrt(1 - a * a) * rng.standard_normal()
    return mean + sd * x


def run_glif(params, I_A, I_B, w, delay, I_B_extra=None, res=0.25):
    """NEST run. I_* are 1 ms piecewise-constant currents (pA); I_B_extra is at ``res`` steps."""
    import nest
    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.resolution = res
    T = len(I_A)
    a, b = nest.Create("glif_psc", 2, params=params)
    times = np.arange(T) * 1.0 + res  # NEST step generators need times > 0
    gA = nest.Create("step_current_generator", params={"amplitude_times": times, "amplitude_values": I_A})
    I_B_fine = np.repeat(I_B, int(round(1.0 / res)))
    if I_B_extra is not None:
        I_B_fine = I_B_fine + I_B_extra
    tf = np.arange(len(I_B_fine)) * res + res
    gB = nest.Create("step_current_generator", params={"amplitude_times": tf, "amplitude_values": I_B_fine})
    nest.Connect(gA, a)
    nest.Connect(gB, b)
    if w:
        nest.Connect(a, b, syn_spec={"weight": w, "delay": delay, "receptor_type": 1})
    mm = nest.Create("multimeter", params={"record_from": ["V_m", "I_syn"], "interval": res})
    nest.Connect(mm, a + b)
    sr = nest.Create("spike_recorder")
    nest.Connect(a + b, sr)
    nest.Simulate(T + 2 * res)
    ev = mm.events
    out = {}
    for name, gid in (("A", a.global_id), ("B", b.global_id)):
        m = ev["senders"] == gid
        out[name] = {"t": ev["times"][m], "V": ev["V_m"][m], "I_syn": ev["I_syn"][m]}
    s = sr.events
    for name, gid in (("A", a.global_id), ("B", b.global_id)):
        out[name]["spikes"] = s["times"][s["senders"] == gid]
    return out


def load_standard(standard_dir):
    sys.path.insert(0, standard_dir)
    import standard_model as sm
    return sm.load()


def cross_corr(pre, post, win=40.0, bin_ms=1.0):
    """Rate of post spikes (Hz) around each pre spike."""
    edges = np.arange(-win, win + bin_ms, bin_ms)
    d = (np.asarray(post)[None, :] - np.asarray(pre)[:, None]).ravel()
    h, _ = np.histogram(d, edges)
    return edges[:-1] + bin_ms / 2, h / max(len(pre), 1) / (bin_ms / 1000.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glif-dir", required=True, help="Allen v1_point components/cell_models/nest_models")
    p.add_argument("--standard-dir", required=True, help="the Standard folder (standard_model.py, weights/)")
    p.add_argument("--out", required=True)
    p.add_argument("--T", type=int, default=60000, help="ms; long enough for the cross-correlograms")
    p.add_argument("--w", type=float, default=60.0, help="synaptic weight, pA (Allen L2/3 E->E max is 17)")
    p.add_argument("--delay", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    T = a.T
    I_A = ou_current(T, 125.0, 60.0, 20.0, rng)
    I_B = ou_current(T, 80.0, 25.0, 20.0, rng)
    I_A[:300] = I_B[:300] = 0.0  # both start from rest
    tau = sl.TAU_SYN[1]
    d_eff = float(sl.nest_effective_delay(a.delay))  # the delay the NEST synapse really has (1.75 ms)
    params = json.load(open(os.path.join(a.glif_dir, GLIF_JSON)))

    # ---- GLIF pair (NEST synapse), its unconnected control, and B driven by the layer instead -----
    g = run_glif(params, I_A, I_B, a.w, a.delay)
    g0 = run_glif(params, I_A, I_B, 0.0, a.delay)
    res = 0.25
    t_fine = g["B"]["t"]
    layer_glif = sl.alpha_current(g["A"]["spikes"], a.w, d_eff, tau, t_fine)
    # current in each NEST step, as a piecewise-constant input to the unconnected B
    sub = 20
    tt = (np.arange(int(T / res) * sub) + 0.5) * (res / sub)
    extra = sl.alpha_current(g["A"]["spikes"], a.w, d_eff, tau, tt).reshape(-1, sub).mean(1)
    # a NEST current generator acts one step after its scheduled time, so schedule it one step early
    extra = np.r_[extra[1:], 0.0]
    gL = run_glif(params, I_A, I_B, 0.0, a.delay, I_B_extra=extra)

    # ---- ANN pair -----------------------------------------------------------------------------
    cell = load_standard(a.standard_dir)
    vA, sA = cell.simulate(I_A[None].astype(np.float32))
    tA = np.flatnonzero(sA[0]).astype(float)
    I_syn_ann = sl.binned_current(tA, a.w, d_eff, tau, T)
    vB, sB = cell.simulate((I_B + I_syn_ann)[None].astype(np.float32))
    vB0, sB0 = cell.simulate(I_B[None].astype(np.float32))
    tB, tB0 = np.flatnonzero(sB[0]).astype(float), np.flatnonzero(sB0[0]).astype(float)

    # stateful form of the layer gives the same current
    psc = sl.AlphaPSC(1, dt=1.0 / sub)
    arr = np.zeros(T * sub)
    idx = np.round((tA + d_eff) * sub).astype(int) - 1  # a spike arriving at the end of step idx
    np.add.at(arr, idx[idx < len(arr)], a.w)
    step_I = np.array([psc.step({1: arr[k:k + 1]})[0] for k in range(T * sub)])
    ref = sl.alpha_current(tA, a.w, d_eff, tau, (np.arange(T * sub) + 1) / sub)

    refr = np.zeros(len(t_fine), bool)
    for s_ in g["B"]["spikes"]:
        refr |= (t_fine > s_) & (t_fine <= s_ + params["t_ref"])

    def count_in(spk, t0=300.0):
        return int((np.asarray(spk) >= t0).sum())

    def matched(x, y, tol=1.0):
        x, y = np.sort(x), np.sort(y)
        if len(x) == 0 or len(y) == 0:
            return 0.0
        j = np.clip(np.searchsorted(y, x), 1, len(y) - 1) if len(y) > 1 else np.zeros(len(x), int)
        d = np.minimum(np.abs(y[j] - x), np.abs(y[np.maximum(j - 1, 0)] - x))
        return float((d <= tol).sum() / max(len(x), len(y)))

    secs = (T - 300) / 1000.0
    isi = lambda s: np.diff(s[s >= 300]) if len(s) > 2 else np.array([np.nan])
    stats = {
        "weight_pA": a.w, "delay_ms": a.delay, "effective_delay_ms": d_eff, "tau_ms": tau,
        "layer_vs_nest_I_syn": {
            # glif_psc holds its recorded I_syn during B's own refractory period (t_ref) while the
            # synaptic state keeps evolving, so those samples are left out of the comparison.
            "max_abs_err_pA": float(np.max(np.abs(layer_glif - g["B"]["I_syn"])[~refr])),
            "samples_compared": int((~refr).sum()), "samples_in_B_refractory": int(refr.sum()),
            "peak_pA": float(np.max(g["B"]["I_syn"])),
        },
        # B driven by the layer's current through a step-current generator instead of a synapse. The
        # generator holds the current constant within each 0.25 ms step, while NEST integrates a
        # synapse's alpha current exactly, so Vm differs slightly and spikes shift by up to ~1.5 ms.
        "glif_B_layer_current_vs_synapse": {
            "spikes_matched_within_2ms": matched(gL["B"]["spikes"], g["B"]["spikes"], 2.0),
            "n_spikes": [int(len(gL["B"]["spikes"])), int(len(g["B"]["spikes"]))],
            "median_abs_dVm_mV": float(np.median(np.abs(gL["B"]["V"] - g["B"]["V"]))),
        },
        "stateful_vs_closed_form_max_err_pA": float(np.max(np.abs(step_I - ref))),
        "rates_hz": {
            "GLIF A": count_in(g["A"]["spikes"]) / secs,
            "GLIF B unconnected": count_in(g0["B"]["spikes"]) / secs,
            "GLIF B connected": count_in(g["B"]["spikes"]) / secs,
            "ANN A": count_in(tA) / secs,
            "ANN B unconnected": count_in(tB0) / secs,
            "ANN B connected": count_in(tB) / secs,
        },
        "cv_isi_A": {"GLIF": float(np.nanstd(isi(g["A"]["spikes"])) / np.nanmean(isi(g["A"]["spikes"]))),
                     "ANN": float(np.nanstd(isi(tA)) / np.nanmean(isi(tA)))},
        "ann_input_range_pA": {"I_A": [float(I_A.min()), float(I_A.max())],
                               "I_B + synaptic": [float((I_B + I_syn_ann).min()), float((I_B + I_syn_ann).max())]},
    }
    with open(os.path.join(a.out, "two_cell_demo.json"), "w") as f:
        json.dump(stats, f, indent=2)
    np.savez_compressed(os.path.join(a.out, "two_cell_demo.npz"), I_A=I_A, I_B=I_B, I_syn_ann=I_syn_ann,
                        vA=vA[0], vB=vB[0], vB0=vB0[0], tA=tA, tB=tB, tB0=tB0,
                        glif_tA=g["A"]["spikes"], glif_tB=g["B"]["spikes"], glif_tB0=g0["B"]["spikes"])
    I_nest = np.where(refr, np.nan, g["B"]["I_syn"])  # glif_psc does not update I_syn while B is refractory
    plot(a.out, T, I_A, I_B, g, g0, layer_glif, I_nest, vA[0], vB[0], vB0[0], tA, tB, tB0, I_syn_ann, stats)
    print(json.dumps(stats, indent=2))


def plot(out, T, I_A, I_B, g, g0, layer_glif, I_nest, vA, vB, vB0, tA, tB, tB0, I_syn_ann, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A_COL, B_COL, CTRL, LAYER = "#2a78d6", "#eb6834", "#9a9a9a", "#1baf7a"
    w0, w1 = 1500, 2500  # window shown, ms
    fig = plt.figure(figsize=(13, 11.5))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.6, 1.2, 0.8, 1.2, 1.3], hspace=0.18, wspace=0.12)
    axes = np.empty((5, 2), object)
    for c in range(2):
        axes[0, c] = fig.add_subplot(gs[0, c])
        for r in (1, 2, 3):
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes[0, c])
        axes[4, c] = fig.add_subplot(gs[4, c])
    fig.subplots_adjust(hspace=0.18)
    t1 = np.arange(T)

    def spikes_up(ax, t, V, spk, col, label, lw=1.0, peak=30.0):
        V = V.copy()
        ax.plot(t, V, color=col, lw=lw, label=label)
        s = np.asarray(spk)
        s = s[(s >= w0) & (s <= w1)]
        ax.vlines(s, np.interp(s, t, V), peak, color=col, lw=lw)

    for col, name in enumerate(("GLIF pair (NEST synapse)", "Standard ANN pair (fixed synaptic layer)")):
        ax = axes[0, col]
        ax.plot(t1, I_A, color=A_COL, lw=0.8, label="drive to A")
        ax.plot(t1, I_B, color=B_COL, lw=0.8, label="drive to B")
        ax.set_title(name, fontsize=11, loc="left")
        if col == 0:
            ax.set_ylabel("injected\ncurrent (pA)")
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    # GLIF
    gA, gB, gB0 = g["A"], g["B"], g0["B"]
    spikes_up(axes[1, 0], gA["t"], gA["V"], gA["spikes"], A_COL, "cell A")
    axes[2, 0].plot(gB["t"], I_nest, color="0.15", lw=2.2, label="NEST synaptic current")
    axes[2, 0].plot(gB["t"], layer_glif, color=LAYER, lw=1.0, ls="--", label="fixed layer on A's spikes")
    spikes_up(axes[3, 0], gB0["t"], gB0["V"], gB0["spikes"], CTRL, "B without synapse", lw=0.8)
    spikes_up(axes[3, 0], gB["t"], gB["V"], gB["spikes"], B_COL, "B with synapse from A")
    # ANN
    spikes_up(axes[1, 1], t1, np.minimum(vA, 30), tA, A_COL, "cell A")
    axes[2, 1].plot(t1, I_syn_ann, color=LAYER, lw=1.2, label="fixed layer on A's spikes (1 ms bins)")
    spikes_up(axes[3, 1], t1, np.minimum(vB0, 30), tB0, CTRL, "B without synapse", lw=0.8)
    spikes_up(axes[3, 1], t1, np.minimum(vB, 30), tB, B_COL, "B with synapse from A")

    for col in range(2):
        axes[1, col].set_ylabel("A: Vm (mV)" if col == 0 else "")
        axes[2, col].set_ylabel("synaptic current\ninto B (pA)" if col == 0 else "")
        axes[3, col].set_ylabel("B: Vm (mV)" if col == 0 else "")
        for r in (1, 2, 3):
            axes[r, col].legend(frameon=False, fontsize=8, loc="upper right")
        for r in range(4):
            axes[r, col].spines[["top", "right"]].set_visible(False)
            axes[r, col].set_xlim(w0, w1)
        for r in (0, 1, 2):
            axes[r, col].tick_params(labelbottom=False)
        axes[3, col].set_xlabel("time (ms)")
    for r in range(4):
        lo = min(axes[r, 0].get_ylim()[0], axes[r, 1].get_ylim()[0])
        hi = max(axes[r, 0].get_ylim()[1], axes[r, 1].get_ylim()[1])
        axes[r, 0].set_ylim(lo, hi)
        axes[r, 1].set_ylim(lo, hi)

    # bottom row: B's spike rate around A's spikes, connected vs unconnected
    for col, (pre, post, post0) in enumerate(((gA["spikes"], gB["spikes"], gB0["spikes"]), (tA, tB, tB0))):
        ax = axes[4, col]
        x, y = cross_corr(pre, post)
        x0, y0 = cross_corr(pre, post0)
        ax.step(x0, y0, where="mid", color=CTRL, lw=1.2, label="B without synapse")
        ax.step(x, y, where="mid", color=B_COL, lw=1.8, label="B with synapse")
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlim(-40, 40)
        ax.set_xlabel("time from A's spike (ms)")
        if col == 0:
            ax.set_ylabel("B rate around\nA's spikes (Hz)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    hi = max(axes[4, 0].get_ylim()[1], axes[4, 1].get_ylim()[1])
    axes[4, 0].set_ylim(0, hi)
    axes[4, 1].set_ylim(0, hi)

    r = stats["rates_hz"]
    fig.text(0.01, 0.005,
             f"Synapse: E->E port 1, w = {stats['weight_pA']:.0f} pA, tau = 5.5 ms, delay = {stats['delay_ms']} ms ({stats['effective_delay_ms']} ms on NEST's grid). "
             f"Rates (Hz): GLIF A {r['GLIF A']:.1f}, B {r['GLIF B unconnected']:.1f} -> {r['GLIF B connected']:.1f};  "
             f"ANN A {r['ANN A']:.1f}, B {r['ANN B unconnected']:.1f} -> {r['ANN B connected']:.1f}.  "
             f"Layer vs NEST I_syn max error {stats['layer_vs_nest_I_syn']['max_abs_err_pA']:.1e} pA.",
             fontsize=8.5, color="0.25")
    fig.savefig(os.path.join(out, "two_cell_demo.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
