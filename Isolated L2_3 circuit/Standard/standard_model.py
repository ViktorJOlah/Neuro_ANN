"""The Standard cell model ("The Chosen One"), self-contained: numpy + tensorflow/keras only.

    I (pA, 1 ms) --64 ms window--> CNN-LSTM front end --> x(t) --> Izhikevich layer (holds Vm, spikes)
                                                                   ^
         stateful conductance branches (NaTs+Nap, Ca->SK, Kv3.1, BK, K_P, Im, Ih) read the
         Izhikevich Vm and add g * I_branch(t-1) to x(t)

Front end:  x(t) = g*(I(t) + I0) + Net(I(t-63..t) / 100),  Net = Conv1D(32,5,causal,relu) x2 -> LSTM(32) -> Dense(1).
Izhikevich: v = (Vm - V0)/Vs, two half-steps of h = time_scale/2 per 1 ms; spike when Vm >= V_th, then
            v <- c, u <- u + d. Branch gates are carried step to step (their memory of past spikes).
Each branch k has a scale exp(ln_s_k) and a bias b_k, and an on/off switch (na_on, sk_on, kv_on, bk_on,
kp_on, im_on, ih_on) for blocker experiments.

Usage:
    import standard_model as sm
    cell = sm.load()                        # this folder's weights and params.json
    vm, spikes = cell.simulate(I)           # I: (n, T) pA at 1 ms; free-running from rest
    vm, spikes = cell.simulate(I, V_seed=V, seed_ms=450)   # recorded Vm drives it for 450 ms first
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf
import keras

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import channels as ch  # noqa: E402

W, I_SCALE = 64, 100.0
GT_SPIKE_MV = -20.0          # recorded spike = upward crossing of -20 mV (for seeding)
SPIKE_DISPLAY_MV = 30.0      # model spikes are drawn at +30 mV
V_SPIKE = SPIKE_DISPLAY_MV - ch.JP
V_REF = -46.0                # below this the front end already learned the steady-state Na/Kv3.1/slow-K current
NAMES = ("V_th", "c_mV", "d", "ln_s_na", "b_na", "ln_s_sk", "b_sk", "ln_s_ca", "ln_s_kv", "b_kv", "ln_s_bk", "b_bk",
         "ln_s_kp", "b_kp", "ln_s_im", "b_im", "ln_s_ih", "b_ih", "th_k", "th_d", "th_tau")
DEFAULTS = dict(th_k=0.0, th_d=0.0, th_tau=30.0)
H0 = float(ch.nats_rates(np.float32(V_REF - ch.JP))[1])


def build_front_end(units=32):
    net = keras.Sequential([keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
                            keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
                            keras.layers.LSTM(units), keras.layers.Dense(1)], name="net")
    net.build((None, W, 1))
    return net


def current_windows(I):
    """(B, T) -> (B, T, 64): row t holds I(t-63..t), left-padded with I[:, 0]."""
    pad = tf.repeat(I[:, :1], W - 1, axis=1)
    return tf.signal.frame(tf.concat([pad, I], 1), W, 1, axis=1)


def gt_masks(V):
    """Recorded spike onsets (upward -20 mV crossings) and the 3 ms after each while Vm >= -20 mV."""
    S = np.zeros(V.shape, bool)
    S[:, 1:] = (V[:, 1:] > GT_SPIKE_MV) & (V[:, :-1] <= GT_SPIKE_MV)
    M = S.copy()
    for k in range(1, 4):
        M[:, k:] |= S[:, :-k] & (V[:, k:] >= GT_SPIKE_MV)
    return S.astype(np.float32), M.astype(np.float32)


class StandardCell(tf.Module):
    def __init__(self, izh, vals, net):
        super().__init__()
        self.net = net
        self.p = {k: tf.constant(float(izh[k]), tf.float32) for k in ("a", "b", "time_scale", "Vs", "V0", "g", "I0")}
        vals = {**DEFAULTS, **vals}
        self.q = {k: tf.Variable(float(vals[k]), dtype=tf.float32, name=k) for k in NAMES}
        for k in ("na_on", "sk_on", "kv_on", "bk_on", "kp_on", "im_on", "ih_on"):
            setattr(self, k, tf.Variable(1.0, dtype=tf.float32, name=k))

    def params(self):
        return {k: float(v.numpy()) for k, v in self.q.items()}

    def set(self, vals):
        for k, v in vals.items():
            self.q[k].assign(v)

    # ---- front end: the Izhikevich cell's input current
    def current(self, I, chunk=4):
        I = np.asarray(I, np.float32)
        out = []
        for k in range(0, len(I), chunk):
            Ik = tf.constant(I[k:k + chunk])
            B, T = tf.shape(Ik)[0], tf.shape(Ik)[1]
            y = self.net(tf.reshape(current_windows(Ik), (-1, W, 1)) / I_SCALE)
            out.append((self.p["g"] * (Ik + self.p["I0"]) + tf.reshape(y, (B, T))).numpy())
        return np.concatenate(out).astype(np.float32)

    # ---- Izhikevich layer + branches. x: front-end current; V_gt/gt_spk/gt_in: recorded Vm and its
    # spike masks, used for t < clamp_until (or every step if one_step); V_gt[:, 0] sets the initial state.
    @tf.function
    def run(self, x, V_gt, gt_spk, gt_in, clamp_until, one_step=False):
        P, Q = self.p, self.q
        a, b, ts, Vs, V0, g = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"], P["g"]
        c = (Q["c_mV"] - V0) / Vs
        s_na, s_sk, s_ca = tf.exp(Q["ln_s_na"]) * self.na_on, tf.exp(Q["ln_s_sk"]) * self.sk_on, tf.exp(Q["ln_s_ca"])
        s_kv, s_bk = tf.exp(Q["ln_s_kv"]) * self.kv_on, tf.exp(Q["ln_s_bk"]) * self.bk_on
        s_kp, s_im, s_ih = tf.exp(Q["ln_s_kp"]) * self.kp_on, tf.exp(Q["ln_s_im"]) * self.im_on, tf.exp(Q["ln_s_ih"]) * self.ih_on
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
        for t in tf.range(T):
            clamp = tf.logical_or(t < clamp_until, one_step)
            if clamp:
                prev = tf.maximum(t - 1, 0)
                first = tf.cast(t > 0, tf.float32)
                v = (V_gt[:, prev] - V0) / Vs
                v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
                u = u + first * gt_spk[:, prev] * Q["d"]
            else:
                v = s * c + (1.0 - s) * v
                u = u + s * Q["d"]
            xt = x[:, t] + g * ib
            for _ in range(2):
                v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
            u = u + ts * a * (b * v - u)
            vm = Vs * v + V0
            v_th = Q["V_th"] + Q["th_k"] * (H0 - st["h_t"]) + theta
            s = tf.cast(vm >= v_th, tf.float32)
            if clamp:
                vb = gt_spk[:, t] * V_SPIKE + (1 - gt_spk[:, t]) * (
                    gt_in[:, t] * (Q["c_mV"] - ch.JP) + (1 - gt_in[:, t]) * (V_gt[:, t] - ch.JP))
                spike_now = gt_spk[:, t]
            else:
                vb = s * V_SPIKE + (1 - s) * (tf.minimum(vm, v_th) - ch.JP)
                spike_now = s
            theta = theta * th_dec + Q["th_d"] * spike_now
            st, cur = ch.step(st, vb, s_ca)
            i_na = -(cur["NaTs"] + cur["Nap"] - ch.na_steady(tf.minimum(vb, V_REF - ch.JP))) * ch.PA_PER_DENSITY
            i_sk = -cur["SK"] * ch.PA_PER_DENSITY
            i_kv = -(cur["Kv3_1"] - ch.kv31_steady(tf.minimum(vb, V_REF - ch.JP))) * ch.PA_PER_DENSITY * (1 - spike_now)
            i_bk = -cur["BK"] * ch.PA_PER_DENSITY * (1 - spike_now)
            ss = ch.slow_steady(tf.minimum(vb, V_REF - ch.JP))
            i_kp = -(cur["K_P"] - ss["K_P"]) * ch.PA_PER_DENSITY * (1 - spike_now)
            i_im = -(cur["Im"] - ss["Im"]) * ch.PA_PER_DENSITY * (1 - spike_now)
            i_ih = -(cur["Ih"] - ss["Ih"]) * ch.PA_PER_DENSITY * (1 - spike_now)
            ib = (s_na * i_na + Q["b_na"] + s_sk * i_sk + Q["b_sk"]
                  + s_kv * i_kv + Q["b_kv"] + s_bk * i_bk + Q["b_bk"]
                  + s_kp * i_kp + Q["b_kp"] + s_im * i_im + Q["b_im"] + s_ih * i_ih + Q["b_ih"])
            vm_ta = vm_ta.write(t, s * SPIKE_DISPLAY_MV + (1.0 - s) * vm)
            s_ta = s_ta.write(t, s)
        return tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack())

    def simulate(self, I, V_seed=None, seed_ms=0, one_step=False, V_rest=-82.0, chunk=64):
        """Vm (mV, spikes drawn at +30) and spikes (bool), both (n, T).
        V_seed: recorded Vm (n, T) that drives the model for seed_ms (or every step if one_step);
        without it the cell starts at V_rest and runs freely."""
        I = np.atleast_2d(np.asarray(I, np.float32))
        x = self.current(I)
        V = np.full(I.shape, V_rest, np.float32) if V_seed is None else np.atleast_2d(np.asarray(V_seed, np.float32))
        S, M = gt_masks(V)
        outs = [self.run(tf.constant(x[k:k + chunk]), tf.constant(V[k:k + chunk]), tf.constant(S[k:k + chunk]),
                         tf.constant(M[k:k + chunk]), tf.constant(int(seed_ms)), tf.constant(bool(one_step)))
                for k in range(0, len(I), chunk)]
        return np.concatenate([o[0].numpy() for o in outs]), np.concatenate([o[1].numpy() for o in outs]) > 0.5


def load(folder=HERE):
    p = json.load(open(os.path.join(folder, "params.json")))
    net = build_front_end()
    w = np.load(os.path.join(folder, "weights", "front_end_cnnlstm.npz"))
    net.set_weights([w[f"w{i}"] for i in range(len(w.files))])
    return StandardCell(p["izhikevich"], p["branches"], net)
