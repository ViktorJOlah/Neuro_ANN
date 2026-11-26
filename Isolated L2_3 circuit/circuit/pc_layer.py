"""All L2/3 PCs as one batch of Standard cells, advanced one 1 ms step at a time.

``StandardCell.run`` integrates a whole current trace at once. In a circuit the input of the next
millisecond depends on the spikes of this one, so here the same model is split into a per-step
update over a batch of n cells:

    I(t) (pA) -> 64 ms window buffer -> CNN-LSTM front end -> x(t) -> Izhikevich + conductance branches -> spike(t)

The state (window buffer, v, u, branch gates, last branch current, threshold) lives in tf.Variables,
and ``step`` is one compiled function, so a step costs one batched front-end call plus a few
elementwise ops. With the same input it reproduces ``StandardCell.simulate`` (free running from
rest); ``tests/test_pc_layer.py`` checks that.

Kernel mode (the default synaptic input since 2026-11-26, experiments/evoked_synaptic PR 22): ``step_kernel(I_e, I_i)``
takes the Allen alpha-PSC currents per receptor port (excitatory, inhibitory; pA as Allen defines them) and does not
pass them through the front end. They become conductances normalised at V_ref,

    I_syn = I_e (E_e - Vm) / (E_e - V_ref) + I_i (E_i - Vm) / (E_i - V_ref)        (Vm of the previous step)

then the cell's membrane correction I -> k0 I + c1 LP_tau1(I) + c2 LP_tau2(I) (experiments/kernel_calibration),
and enter the Izhikevich input directly. The front end sees zero somatic current, so its output is a constant.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


class BatchedStandard:
    def __init__(self, cell, n: int, V_rest: float = -82.0, I_first=None, jit: bool = True, kernel=None):
        """kernel: None (current mode) or dict(corr={k0, c1, tau1, c2, tau2}, E_e, E_i, V_ref) for ``step_kernel``."""
        import standard_model as sm
        import channels as ch
        self.sm, self.ch, self.cell, self.n = sm, ch, cell, n
        P, Q = cell.p, cell.q
        self.V0, self.Vs = P["V0"], P["Vs"]
        v0 = tf.fill([n], (V_rest - float(P["V0"])) / float(P["Vs"]))
        st = ch.init_state(tf.fill([n], V_rest - ch.JP))
        self.keys = sorted(st)
        self.v = tf.Variable(v0)
        self.u = tf.Variable(P["b"] * v0)
        self.st = {k: tf.Variable(tf.cast(st[k], tf.float32)) for k in self.keys}
        self.s = tf.Variable(tf.zeros([n]))
        self.ib = tf.Variable(tf.zeros([n]))
        self.theta = tf.Variable(tf.zeros([n]))
        first = np.zeros(n, np.float32) if I_first is None else np.asarray(I_first, np.float32)
        # the trained model pads the first window with I(0); the first step overwrites the last column
        self.win = tf.Variable(np.repeat(first[:, None], sm.W, 1))
        self.vm = tf.Variable(tf.fill([n], V_rest))
        self._step = tf.function(self._step_impl, jit_compile=False)
        self._front = tf.function(self._front_impl, jit_compile=jit)
        self.kernel = kernel
        if kernel is not None:
            c = kernel["corr"]
            self.kc = {k: tf.constant(float(c[k]), tf.float32) for k in ("k0", "c1", "tau1", "c2", "tau2")}
            self.kE = {k: tf.constant(float(kernel[k]), tf.float32) for k in ("E_e", "E_i", "V_ref")}
            self.z1 = tf.Variable(tf.zeros([n]))
            self.z2 = tf.Variable(tf.zeros([n]))
            self.x0 = tf.constant(self._front_impl(tf.zeros([1, sm.W]))[0])     # front end on zero current: constant
            self._kstep = tf.function(self._kstep_impl, jit_compile=False)

    def _front_impl(self, win):
        sm, P = self.sm, self.cell.p
        y = self.cell.net(win[:, :, None] / sm.I_SCALE)[:, 0]
        return P["g"] * (win[:, -1] + P["I0"]) + y

    def _step_impl(self, I_t):
        win = tf.concat([self.win[:, 1:], I_t[:, None]], 1)
        self.win.assign(win)
        return self._izh(self._front(win))

    def _kstep_impl(self, I_e, I_i):
        kc, kE = self.kc, self.kE
        V = tf.minimum(self.vm, -20.0)                                    # previous step's Vm (spikes clipped)
        I = I_e * (kE["E_e"] - V) / (kE["E_e"] - kE["V_ref"]) + I_i * (kE["E_i"] - V) / (kE["E_i"] - kE["V_ref"])
        d1, d2 = tf.exp(-1.0 / kc["tau1"]), tf.exp(-1.0 / kc["tau2"])
        z1 = self.z1 * d1 + (1.0 - d1) * I
        z2 = self.z2 * d2 + (1.0 - d2) * I
        self.z1.assign(z1); self.z2.assign(z2)
        I_eff = kc["k0"] * I + kc["c1"] * z1 + kc["c2"] * z2
        return self._izh(self.x0 + self.cell.p["g"] * I_eff)

    def _izh(self, x):
        sm, ch, P, Q, c_ = self.sm, self.ch, self.cell.p, self.cell.q, self.cell
        a, b, ts, Vs, V0, g = P["a"], P["b"], P["time_scale"], P["Vs"], P["V0"], P["g"]
        c = (Q["c_mV"] - V0) / Vs
        s_na, s_sk, s_ca = tf.exp(Q["ln_s_na"]) * c_.na_on, tf.exp(Q["ln_s_sk"]) * c_.sk_on, tf.exp(Q["ln_s_ca"])
        s_kv, s_bk = tf.exp(Q["ln_s_kv"]) * c_.kv_on, tf.exp(Q["ln_s_bk"]) * c_.bk_on
        s_kp, s_im, s_ih = tf.exp(Q["ln_s_kp"]) * c_.kp_on, tf.exp(Q["ln_s_im"]) * c_.im_on, tf.exp(Q["ln_s_ih"]) * c_.ih_on
        th_dec = tf.exp(-1.0 / tf.maximum(Q["th_tau"], 1.0))
        h = 0.5 * ts
        s = self.s
        v = s * c + (1.0 - s) * self.v
        u = self.u + s * Q["d"]
        xt = x + g * self.ib
        for _ in range(2):
            v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + xt)
        u = u + ts * a * (b * v - u)
        vm = Vs * v + V0
        st = {k: self.st[k].read_value() for k in self.keys}
        v_th = Q["V_th"] + Q["th_k"] * (sm.H0 - st["h_t"]) + self.theta
        s = tf.cast(vm >= v_th, tf.float32)
        vb = s * sm.V_SPIKE + (1 - s) * (tf.minimum(vm, v_th) - ch.JP)
        theta = self.theta * th_dec + Q["th_d"] * s
        st, cur = ch.step(st, vb, s_ca)
        lo = tf.minimum(vb, sm.V_REF - ch.JP)
        i_na = -(cur["NaTs"] + cur["Nap"] - ch.na_steady(lo)) * ch.PA_PER_DENSITY
        i_sk = -cur["SK"] * ch.PA_PER_DENSITY
        i_kv = -(cur["Kv3_1"] - ch.kv31_steady(lo)) * ch.PA_PER_DENSITY * (1 - s)
        i_bk = -cur["BK"] * ch.PA_PER_DENSITY * (1 - s)
        ss = ch.slow_steady(lo)
        i_kp = -(cur["K_P"] - ss["K_P"]) * ch.PA_PER_DENSITY * (1 - s)
        i_im = -(cur["Im"] - ss["Im"]) * ch.PA_PER_DENSITY * (1 - s)
        i_ih = -(cur["Ih"] - ss["Ih"]) * ch.PA_PER_DENSITY * (1 - s)
        ib = (s_na * i_na + Q["b_na"] + s_sk * i_sk + Q["b_sk"]
              + s_kv * i_kv + Q["b_kv"] + s_bk * i_bk + Q["b_bk"]
              + s_kp * i_kp + Q["b_kp"] + s_im * i_im + Q["b_im"] + s_ih * i_ih + Q["b_ih"])
        self.v.assign(v)
        self.u.assign(u)
        for k in self.keys:
            self.st[k].assign(tf.cast(st[k], tf.float32))
        self.s.assign(s)
        self.ib.assign(ib)
        self.theta.assign(theta)
        out_vm = s * sm.SPIKE_DISPLAY_MV + (1.0 - s) * vm
        self.vm.assign(out_vm)
        return s

    def step_kernel(self, I_e, I_i) -> np.ndarray:
        """Kernel mode: advance 1 ms with alpha-PSC currents per port (n,) pA; returns spiked (n,) bool."""
        return self._kstep(tf.constant(np.asarray(I_e, np.float32)), tf.constant(np.asarray(I_i, np.float32))).numpy() > 0.5

    def step(self, I_t) -> np.ndarray:
        """Advance every cell by 1 ms with input current I_t (n,) pA; returns spiked (n,) bool."""
        return self._step(tf.constant(np.asarray(I_t, np.float32))).numpy() > 0.5
