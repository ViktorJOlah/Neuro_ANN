"""Hardcoded synaptic kernel layer for the Standard model: evoked events -> conductance -> current at the Izhikevich layer.

    somatic DC current --64 ms window--> CNN-LSTM front end --> x(t) --+--> Izhikevich (Vm, spikes) + branches
                                                                       |          |
    stim pulses (L1, L2/3) --> [depression R] --> rise/decay states --> g(t) --> I_syn = g (E_rev - Vm(t-1))

Per pathway p and kernel component k, two recursive states are updated every 1 ms like the branch gates:
    a      = w_pk * s_p(t - delay) * R_p * G_p(t)           (event amplitude, nS)
    xr    <- xr * exp(-1/tau_r) + a,  xd <- xd * exp(-1/tau_d) + a
    g_pk   = norm(tau_r, tau_d) * (xd - xr)                   (difference of exponentials, peak = w)
    R_p   <- 1 - (1 - R_p) exp(-1/tau_rec);  R_p <- R_p (1 - U s_p) at each event   (Tsodyks-Markram style)
With tau_r -> tau_d this is the AlphaSynapse. The layer "knows" how long ago each event arrived through its state,
with no window. The current uses the Izhikevich Vm of the previous step, so the driving force comes for free.
The EPSP shape is smooth by construction: the only fast thing is the kernel itself.

G_p(t) is the per-event gain:
  clean  G = 1. Events never touch the ANN. Learnable: w, tau_r, tau_d, E_rev, U, tau_rec per pathway.
  soft   G = exp(TCN(stim history, 64 ms)), read only at event times: the ANN scales each event's conductance
         increment (summation / intensity nonlinearity) but cannot change the kernel shape.
Optional NMDA-like term (--nmda): a second slow kernel with a Mg block B(V) = 1 / (1 + [Mg]/3.57 exp(-0.062 V)).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras

import standard_model as sm
import channels as ch

PATHS = ("L1", "L23")


def dexp_norm(tr, td):
    """1 / peak of exp(-t/td) - exp(-t/tr) (continuous time)."""
    tp = tr * td / (td - tr) * tf.math.log(td / tr)
    return 1.0 / (tf.exp(-tp / td) - tf.exp(-tp / tr))


class KernelLayer(tf.Module):
    """Parameters live in unconstrained form; `values()` gives the physical ones."""

    def __init__(self, ncomp=1, nmda=False, delay=(2, 2), init=None, signed=False):
        super().__init__()
        self.ncomp, self.nmda, self.delay = ncomp, nmda, tuple(int(d) for d in delay)
        # signed: components 2.. are subtracted (a slow undershoot). This is not a synaptic property: it corrects
        # the bare Izhikevich membrane, which decays more slowly than the cell when current bypasses the front end.
        self.sign = tf.constant([1.0] + [-1.0 if signed else 1.0] * (ncomp - 1), tf.float32)
        i = init or {}
        # one row per pathway (L1, L2/3), one column per kernel component
        tr0 = np.array(i.get("tau_r", [[2.0] + [6.0] * (ncomp - 1), [0.7] + [3.0] * (ncomp - 1)]), np.float32)
        td0 = np.array(i.get("tau_d", [[12.0] + [40.0] * (ncomp - 1), [4.0] + [20.0] * (ncomp - 1)]), np.float32)
        w0 = np.abs(np.array(i.get("w", [[1.0] + [0.2] * (ncomp - 1), [2.0] + [0.2] * (ncomp - 1)]), np.float32))
        self.ln_tr = tf.Variable(np.log(tr0), name="ln_tau_r")
        self.ln_dt = tf.Variable(np.log(np.maximum(td0 - tr0, 0.2)), name="ln_tau_d_minus_r")   # keeps tau_d > tau_r
        self.ln_w = tf.Variable(np.log(w0), name="ln_w")                                         # peak conductance, nS per unit intensity
        self.E = tf.Variable(np.array(i.get("E_rev", [0.0, 0.0]), np.float32), name="E_rev")     # mV (recording convention)
        self.lg_U = tf.Variable(np.array(i.get("lg_U", [-1.0, -1.0]), np.float32), name="logit_U")
        self.ln_rec = tf.Variable(np.log(np.array(i.get("tau_rec", [300.0, 300.0]), np.float32)), name="ln_tau_rec")
        if nmda:
            self.ln_wn = tf.Variable(np.log(np.array(i.get("w_nmda", [0.05, 0.05]), np.float32)), name="ln_w_nmda")
            self.ln_trn = tf.Variable(np.log(np.array([2.0, 2.0], np.float32)), name="ln_tau_r_nmda")
            self.ln_tdn = tf.Variable(np.log(np.array([50.0, 50.0], np.float32)), name="ln_tau_d_nmda")
        self.on = tf.Variable(1.0, trainable=False, name="syn_on")

    def values(self):
        tr = tf.exp(self.ln_tr)
        v = dict(tau_r=tr, tau_d=tr + tf.exp(self.ln_dt), w=tf.exp(self.ln_w) * self.sign, E_rev=self.E,
                 U=tf.sigmoid(self.lg_U), tau_rec=tf.exp(self.ln_rec))
        if self.nmda:
            v.update(w_nmda=tf.exp(self.ln_wn), tau_r_nmda=tf.exp(self.ln_trn),
                     tau_d_nmda=tf.exp(self.ln_trn) + tf.exp(self.ln_tdn))
        return v

    def as_json(self):
        return {k: np.asarray(v.numpy()).tolist() for k, v in self.values().items()}


def build_gain_net(width=16):
    """Soft variant: 64 ms dilated causal CNN over the stimulus -> log-gain per pathway (zero-initialised => G = 1)."""
    inp = keras.Input((None, 2))
    x = keras.layers.Conv1D(width, 1)(inp)
    for d in (1, 2, 4, 8, 16, 32):
        r = keras.layers.Conv1D(width, 2, dilation_rate=d, padding="causal", activation="relu")(x)
        x = keras.layers.Add()([x, keras.layers.Conv1D(width, 1)(r)])
    y = keras.layers.Conv1D(2, 1, kernel_initializer="zeros")(keras.layers.Activation("relu")(x))
    return keras.Model(inp, y)


def run_syn(S, K, x, ev, V_gt, gt_spk, gt_in, clamp_until, one_step=False, return_isyn=False):
    """S.run (Standard Izhikevich layer + branches) with the kernel layer's current added to the Izhikevich input.
    x: front-end current (B, T) from the somatic DC only; ev: event amplitudes (B, T, 2) = intensity x gain,
    already aligned so ev[:, t] arrives at step t (delay applied by the caller)."""
    P, Q = S.p, S.q
    a, b, ts, Vs, V0, g = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"], P["g"]
    c = (Q["c_mV"] - V0) / Vs
    s_na, s_sk, s_ca = tf.exp(Q["ln_s_na"]) * S.na_on, tf.exp(Q["ln_s_sk"]) * S.sk_on, tf.exp(Q["ln_s_ca"])
    s_kv, s_bk = tf.exp(Q["ln_s_kv"]) * S.kv_on, tf.exp(Q["ln_s_bk"]) * S.bk_on
    s_kp, s_im, s_ih = tf.exp(Q["ln_s_kp"]) * S.kp_on, tf.exp(Q["ln_s_im"]) * S.im_on, tf.exp(Q["ln_s_ih"]) * S.ih_on
    kv_ = K.values()
    tr, td, w = kv_["tau_r"], kv_["tau_d"], kv_["w"]                     # (2, ncomp)
    dr, dd, nrm = tf.exp(-1.0 / tr), tf.exp(-1.0 / td), dexp_norm(tr, td) * w
    U, drec, E = kv_["U"], tf.exp(-1.0 / kv_["tau_rec"]), kv_["E_rev"]  # (2,)
    if K.nmda:
        trn, tdn = kv_["tau_r_nmda"], kv_["tau_d_nmda"]
        drn, ddn, nrmn = tf.exp(-1.0 / trn), tf.exp(-1.0 / tdn), dexp_norm(trn, tdn) * kv_["w_nmda"]
    n, T = tf.shape(x)[0], tf.shape(x)[1]
    v = (V_gt[:, 0] - V0) / Vs
    u = b * v
    vm_prev = V_gt[:, 0]
    st = ch.init_state(V_gt[:, 0] - ch.JP)
    s = tf.zeros([n], tf.float32)
    ib = tf.zeros([n], tf.float32)
    theta = tf.zeros([n], tf.float32)
    xr = tf.zeros([n, 2, K.ncomp]); xd = tf.zeros([n, 2, K.ncomp])
    xrn = tf.zeros([n, 2]); xdn = tf.zeros([n, 2])
    R = tf.ones([n, 2])
    th_dec = tf.exp(-1.0 / tf.maximum(Q["th_tau"], 1.0))
    vm_ta, s_ta, i_ta = (tf.TensorArray(tf.float32, size=T) for _ in range(3))
    h = 0.5 * ts
    for t in tf.range(T):
        clamp = tf.logical_or(t < clamp_until, one_step)
        if clamp:
            prev = tf.maximum(t - 1, 0)
            first = tf.cast(t > 0, tf.float32)
            v = (V_gt[:, prev] - V0) / Vs
            v = first * gt_in[:, prev] * c + (1.0 - first * gt_in[:, prev]) * v
            u = u + first * gt_spk[:, prev] * Q["d"]
            vdrive = V_gt[:, prev]
        else:
            v = s * c + (1.0 - s) * v
            u = u + s * Q["d"]
            vdrive = vm_prev
        # ---- synaptic kernel layer
        e = ev[:, t]                                                   # (n, 2)
        amp = e * R                                                    # depression scales the release
        R = 1.0 - (1.0 - R) * drec
        R = R * (1.0 - U * tf.minimum(e, 1.0) * tf.cast(e > 0, tf.float32))
        xr = xr * dr + amp[:, :, None]
        xd = xd * dd + amp[:, :, None]
        gsyn = tf.reduce_sum(nrm * (xd - xr), -1)                     # (n, 2) nS
        vclip = tf.minimum(vdrive, -20.0)[:, None]
        isyn = tf.reduce_sum(gsyn * (E - vclip), -1)                  # pA (nS x mV)
        if K.nmda:
            xrn = xrn * drn + amp; xdn = xdn * ddn + amp
            B = 1.0 / (1.0 + tf.exp(-0.062 * vclip) * (1.0 / 3.57))
            isyn = isyn + tf.reduce_sum(nrmn * (xdn - xrn) * B * (0.0 - vclip), -1)
        isyn = isyn * K.on
        xt = x[:, t] + g * (ib + isyn)
        for _ in range(2):
            v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
        u = u + ts * a * (b * v - u)
        vm = Vs * v + V0
        v_th = Q["V_th"] + Q["th_k"] * (sm.H0 - st["h_t"]) + theta
        s = tf.cast(vm >= v_th, tf.float32)
        vm_prev = tf.minimum(vm, v_th)
        if clamp:
            vb = gt_spk[:, t] * sm.V_SPIKE + (1 - gt_spk[:, t]) * (
                gt_in[:, t] * (Q["c_mV"] - ch.JP) + (1 - gt_in[:, t]) * (V_gt[:, t] - ch.JP))
            spike_now = gt_spk[:, t]
        else:
            vb = s * sm.V_SPIKE + (1 - s) * (tf.minimum(vm, v_th) - ch.JP)
            spike_now = s
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
        i_ta = i_ta.write(t, isyn)
    out = (tf.transpose(vm_ta.stack()), tf.transpose(s_ta.stack()))
    return out + (tf.transpose(i_ta.stack()),) if return_isyn else out


def events(K, stim, gain_net=None):
    """Event amplitudes at the Izhikevich step, with each pathway's delay applied. stim: (B, T, 2)."""
    e = stim
    if gain_net is not None:
        e = e * tf.exp(tf.clip_by_value(gain_net(stim), -3.0, 3.0))
    cols = []
    for j, d in enumerate(K.delay):
        c = e[:, :, j]
        cols.append(tf.concat([tf.zeros_like(c[:, :d]), c[:, :tf.shape(c)[1] - d]], 1) if d else c)
    return tf.stack(cols, -1)
