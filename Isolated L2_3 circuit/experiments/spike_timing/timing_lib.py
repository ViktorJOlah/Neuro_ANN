"""Shared helpers for the spike-timing work on the real Allen L2/3 recordings.

* load_fitted(cell)      the Standard as fitted in experiments/real_cells (front_end.npz + params.json)
* run_forced(...)        the Standard's Izhikevich + branch loop with a per-sample clamp mask: wherever
                         clamp[:, t] = 1 the recorded Vm (and recorded spikes) drive the state, exactly as
                         the seed period does in StandardCell.run. Clamping a few ms around every recorded
                         spike gives "spike-forced" runs: the model always has the true spike history and
                         only has to predict the voltage (and the next spike) in between.
* gamma(...)             coincidence factor, identical to experiments/real_cells/code/compare.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))               # Isolated L2_3 circuit/
RC = os.path.join(ROOT, "experiments", "real_cells")
STD = os.path.join(ROOT, "Standard")
sys.path.insert(0, STD)
sys.path.insert(0, os.path.join(RC, "code"))
import standard_model as sm  # noqa: E402
import channels as ch  # noqa: E402

CELLS = ("318808427", "382982932", "502269786", "513531471", "521409057")   # 485468180 left out (see real_cells README)
SEED_MS = 450
WIN = 4


def load_sweeps(cell):
    meta = json.load(open(os.path.join(RC, "data", cell, "dataset.json")))
    z = np.load(os.path.join(RC, "data", cell, "sweeps_1ms.npz"))
    return [dict(r, I=z[f"s{r['sweep']}_I"], V=z[f"s{r['sweep']}_V"], spk_ms=z[f"s{r['sweep']}_spk_ms"])
            for r in meta["sweeps"]]


def load_fitted(cell, folder=None):
    folder = folder or os.path.join(RC, "results", cell, "standard")
    p = json.load(open(os.path.join(folder, "params.json")))
    net = sm.build_front_end()
    w = np.load(os.path.join(folder, "front_end.npz"))
    net.set_weights([w[f"w{i}"] for i in range(len(w.files))])
    return sm.StandardCell(p["izhikevich"], p["branches"], net), p


def gt_spike_ms(V):
    V = np.asarray(V)
    return np.flatnonzero((V[1:] > sm.GT_SPIKE_MV) & (V[:-1] <= sm.GT_SPIKE_MV)) + 1


def spike_clamp(V, seed_ms, pre=2, post=4):
    """(B, T) clamp mask: the seed period plus [t0-pre, t0+post] around every recorded spike onset t0."""
    V = np.atleast_2d(V)
    C = np.zeros(V.shape, np.float32)
    C[:, :seed_ms] = 1
    for i in range(len(V)):
        for t0 in gt_spike_ms(V[i]):
            C[i, max(0, t0 - pre):t0 + post + 1] = 1
    return C


@tf.function(reduce_retracing=True)
def run_forced(cell, x, V_gt, gt_spk, gt_in, clamp, coinc=0):
    """As StandardCell.run, but clamp is a (B, T) 0/1 mask instead of 't < clamp_until'.
    coinc > 0: a clamp step is skipped when the model itself spiked within the last `coinc` ms (a coincident
    model spike stands in for the recorded one, so the cell is not kicked twice)."""
    P, Q = cell.p, cell.q
    a, b, ts, Vs, V0, g = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"], P["g"]
    c = (Q["c_mV"] - V0) / Vs
    s_na, s_sk, s_ca = tf.exp(Q["ln_s_na"]) * cell.na_on, tf.exp(Q["ln_s_sk"]) * cell.sk_on, tf.exp(Q["ln_s_ca"])
    s_kv, s_bk = tf.exp(Q["ln_s_kv"]) * cell.kv_on, tf.exp(Q["ln_s_bk"]) * cell.bk_on
    s_kp, s_im, s_ih = tf.exp(Q["ln_s_kp"]) * cell.kp_on, tf.exp(Q["ln_s_im"]) * cell.im_on, tf.exp(Q["ln_s_ih"]) * cell.ih_on
    n, T = tf.shape(x)[0], tf.shape(x)[1]
    v = (V_gt[:, 0] - V0) / Vs
    u = b * v
    st = ch.init_state(V_gt[:, 0] - ch.JP)
    s = tf.zeros([n], tf.float32)
    ib = tf.zeros([n], tf.float32)
    theta = tf.zeros([n], tf.float32)
    th_dec = tf.exp(-1.0 / tf.maximum(Q["th_tau"], 1.0))
    vm_ta, s_ta = tf.TensorArray(tf.float32, size=T), tf.TensorArray(tf.float32, size=T)
    h = 0.5 * ts
    since = tf.fill([n], 1e6)
    cm_prev = clamp[:, 0]
    for t in tf.range(T):
        cm = clamp[:, t]
        if coinc > 0:
            cm = cm * tf.cast(tf.logical_or(t < 2, since > coinc), tf.float32)
        prev = tf.maximum(t - 1, 0)
        first = tf.cast(t > 0, tf.float32)
        cprev = cm_prev * first + (1.0 - first) if coinc > 0 else clamp[:, prev] * first + (1.0 - first)
        # clamped: recorded Vm of the previous step (reset if it was inside a recorded spike)
        vc = (V_gt[:, prev] - V0) / Vs
        vc = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * vc
        uc = u + first * gt_spk[:, prev] * Q["d"]
        vf = s * c + (1.0 - s) * v
        uf = u + s * Q["d"]
        v = cprev * vc + (1.0 - cprev) * vf
        u = cprev * uc + (1.0 - cprev) * uf
        xt = x[:, t] + g * ib
        for _ in range(2):
            v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
        u = u + ts * a * (b * v - u)
        vm = Vs * v + V0
        v_th = Q["V_th"] + Q["th_k"] * (sm.H0 - st["h_t"]) + theta
        s_free = tf.cast(vm >= v_th, tf.float32)
        vb_c = gt_spk[:, t] * sm.V_SPIKE + (1 - gt_spk[:, t]) * (
            gt_in[:, t] * (Q["c_mV"] - ch.JP) + (1 - gt_in[:, t]) * (V_gt[:, t] - ch.JP))
        vb_f = s_free * sm.V_SPIKE + (1 - s_free) * (tf.minimum(vm, v_th) - ch.JP)
        vb = cm * vb_c + (1 - cm) * vb_f
        spike_now = cm * gt_spk[:, t] + (1 - cm) * s_free
        s = (1 - cm) * s_free                       # model spikes only count (and reset) outside the clamp
        since = tf.where(s > 0.5, tf.zeros_like(since), since + 1.0)
        cm_prev = cm
        theta = theta * th_dec + Q["th_d"] * spike_now
        st, cur = ch.step(st, vb, s_ca)
        i_na = -(cur["NaTs"] + cur["Nap"] - ch.na_steady(tf.minimum(vb, sm.V_REF - ch.JP))) * ch.PA_PER_DENSITY
        i_sk = -cur["SK"] * ch.PA_PER_DENSITY
        i_kv = -(cur["Kv3_1"] - ch.kv31_steady(tf.minimum(vb, sm.V_REF - ch.JP))) * ch.PA_PER_DENSITY * (1 - spike_now)
        i_bk = -cur["BK"] * ch.PA_PER_DENSITY * (1 - spike_now)
        ss = ch.slow_steady(tf.minimum(vb, sm.V_REF - ch.JP))
        i_kp = -(cur["K_P"] - ss["K_P"]) * ch.PA_PER_DENSITY * (1 - spike_now)
        i_im = -(cur["Im"] - ss["Im"]) * ch.PA_PER_DENSITY * (1 - spike_now)
        i_ih = -(cur["Ih"] - ss["Ih"]) * ch.PA_PER_DENSITY * (1 - spike_now)
        ib = (s_na * i_na + Q["b_na"] + s_sk * i_sk + Q["b_sk"]
              + s_kv * i_kv + Q["b_kv"] + s_bk * i_bk + Q["b_bk"]
              + s_kp * i_kp + Q["b_kp"] + s_im * i_im + Q["b_im"] + s_ih * i_ih + Q["b_ih"])
        vm_ta = vm_ta.write(t, s * sm.SPIKE_DISPLAY_MV + (1.0 - s) * vm)
        s_ta = s_ta.write(t, s)
    return tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack())


def simulate(cell, I, V, clamp, x=None, chunk=32):
    """Vm and model spikes for (B, T) current/recorded Vm with a (B, T) clamp mask (rows may differ in length
    only if pre-padded by the caller)."""
    I, V, clamp = (np.atleast_2d(np.asarray(a, np.float32)) for a in (I, V, clamp))
    if x is None:
        x = cell.current(I)
    S, M = sm.gt_masks(V)
    vms, sps = [], []
    for k in range(0, len(I), chunk):
        vm, s = run_forced(cell, *(tf.constant(a[k:k + chunk]) for a in (x, V, S, M, clamp)))
        vms.append(vm.numpy()); sps.append(s.numpy() > 0.5)
    return np.concatenate(vms), np.concatenate(sps)


def gamma(td, tm, T, win=WIN):
    td, tm = np.asarray(td, float), np.asarray(tm, float)
    nd, nm = len(td), len(tm)
    if nd == 0 and nm == 0:
        return np.nan
    if nd == 0 or nm == 0:
        return 0.0
    used, co = np.zeros(nm, bool), 0
    for t in td:
        k = np.flatnonzero(~used & (np.abs(tm - t) <= win))
        if len(k):
            used[k[np.argmin(np.abs(tm[k] - t))]] = True
            co += 1
    nu = nm / T
    return float((co - 2 * nu * win * nd) / (0.5 * (nd + nm)) / (1 - 2 * nu * win))


def away_mask(T, spikes, pre=3, post=10):
    m = np.ones(T, bool)
    for t in spikes:
        m[max(0, int(t) - pre):int(t) + post] = False
    return m
