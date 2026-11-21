"""Windowed (batch, 64, 2) setup for every model, following Viktor's ANN convention.

Input for every model, for target time t:
    window[:, k, 0] = Vm(t - 64 + k)          k = 0..63   (64 ms of past voltage, up to t-1)
    window[:, k, 1] = I(t - 63 + k)           k = 0..63   (current shifted forward by one: t-63..t)
Target: Vm(t). Every simulation gives T - 64 overlapping windows, each with a unique end. Windows
from all protocols are pooled and shuffled into one training set.

Izhikevich top layer (IzhikevichStep): every model ends with one step of the published 2003
Izhikevich model (two 0.5 ms half-steps for v, then u with the new v).
  * v(t-1) is the last voltage in the window, mapped to model units: v = (Vm - V0) / Vs.
  * u(t-1) is not observed, so it is rebuilt by running the u equation over the 64 ms window,
    driven by the window's voltages and starting from u = b*v. With the fitted a it relaxes in
    roughly 30 ms, so 64 ms is enough. A spike in the window (a sample >= -20 mV) adds d to u
    and, if it is the last sample, resets v to c, as in the published reset.
  * The ANN supplies the cell's input current x(t). The Izhikevich-only model uses
    x = g * (I(t) + I0).
  * Output: Vm(t) = Vs * min(v(t), 30) + V0. A spike step is clipped to the model's peak, and
    the next step sees it in its window and resets.
In continuous prediction the model's own Vm(t) goes into the next window's voltage column.
So one-step and continuous prediction are the same computation, differing only in where the
voltage column comes from.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf
import keras

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)

W = 64
V_SPIKE = -20.0                 # a window sample >= this counts as a spike (ground truth and model)
V_MEAN, V_SCALE = -70.0, 30.0   # fixed input normalization for the ANNs
I_SCALE = 100.0
X_SCALE = 20.0                  # ANN output scale: the cell's input current spans tens of units
MODEL_NAMES = ("izh", "dense3", "cnnlstm", "dense_cnnlstm")
LABELS = {"izh": "Izhikevich alone", "dense3": "3×32 dense → Izh", "cnnlstm": "CNN-LSTM → Izh",
          "dense_cnnlstm": "dense + CNN-LSTM → Izh"}
COLORS = {"izh": "#e67e22", "dense3": "#2e86c1", "cnnlstm": "#27ae60", "dense_cnnlstm": "#8e44ad"}


# ------------------------------------------------------------------ data
class Windows:
    """All 64-step windows of a set of traces, gathered by index (vectorized)."""

    def __init__(self, traces):
        """traces: list of (I, V) 1-D arrays of possibly different lengths."""
        self.Iall = np.concatenate([np.asarray(i, np.float32) for i, _ in traces])
        self.Vall = np.concatenate([np.asarray(v, np.float32) for _, v in traces])
        starts = np.cumsum([0] + [len(v) for _, v in traces[:-1]])
        # global index of each window's target sample t (t >= 64 within its trace)
        self.t = np.concatenate([s0 + np.arange(W, len(v)) for s0, (_, v) in zip(starts, traces)]).astype(np.int64)

    def __len__(self):
        return len(self.t)

    def gather(self, idx):
        t = self.t[idx][:, None]
        k = np.arange(W)[None, :]
        X = np.stack([self.Vall[t - W + k], self.Iall[t - W + 1 + k]], -1)
        return X, self.Vall[t[:, 0]]


def windows_of_trace(I, V):
    """All windows of one or more equal-length traces, (n_traces * (T-64), 64, 2), plus targets."""
    I, V = np.atleast_2d(I), np.atleast_2d(V)
    n, T = V.shape
    vw = np.lib.stride_tricks.sliding_window_view(V, W, axis=1)[:, :T - W]          # V[t-64 : t]
    iw = np.lib.stride_tricks.sliding_window_view(I, W, axis=1)[:, 1:T - W + 1]     # I[t-63 : t+1]
    X = np.stack([vw, iw], -1).reshape(-1, W, 2)
    return X.astype(np.float32), V[:, W:].reshape(-1)


# ------------------------------------------------------------------ Izhikevich step
class IzhikevichStep(keras.layers.Layer):
    def __init__(self, init, trainable_params=("a", "b", "time_scale"), map_trainable=False, **kw):
        super().__init__(**kw)
        self.init_p, self.train_names, self.map_trainable = dict(init), tuple(trainable_params), map_trainable

    def build(self, input_shape):
        for n in ("a", "b", "c", "d", "time_scale"):
            setattr(self, n, self.add_weight(name=n, shape=(), trainable=n in self.train_names,
                                             initializer=keras.initializers.Constant(self.init_p[n])))
        for n in ("Vs", "V0"):
            setattr(self, n, self.add_weight(name=n, shape=(), trainable=self.map_trainable,
                                             initializer=keras.initializers.Constant(self.init_p[n])))

    def call(self, window, x):
        V = window[:, :, 0]
        v_hist = (V - self.V0) / self.Vs
        spk = tf.cast(V >= V_SPIKE, V.dtype)
        rate = self.time_scale * self.a
        u = self.b * v_hist[:, 0]
        for k in range(1, W):                       # u(t-1), driven by the window's voltages
            u = u + self.d * spk[:, k - 1]
            u = u + rate * (self.b * v_hist[:, k] - u)
        v = v_hist[:, W - 1]
        fired = spk[:, W - 1]
        v = fired * self.c + (1.0 - fired) * v
        u = u + fired * self.d
        h = 0.5 * self.time_scale
        for _ in range(2):
            v = v + h * (0.04 * (v * v) + 5.0 * v + 140.0 - u + x)
        return self.Vs * tf.minimum(v, 30.0) + self.V0


def equilibrium_current(p, V_rest=-82.0):
    v = (V_rest - p["V0"]) / p["Vs"]
    return -(0.04 * v * v + 5 * v + 140 - p["b"] * v)


# ------------------------------------------------------------------ models
RS = dict(a=0.02, b=0.2, c=-65.0, d=8.0, time_scale=1.0, Vs=1.0, V0=0.0)


class WindowModel(keras.Model):
    def __init__(self, arch, izh_init=None, units=32):
        super().__init__()
        self.arch = arch
        if arch == "izh":
            p = dict(izh_init)
            self.step = IzhikevichStep(p, ("a", "b", "time_scale"), map_trainable=True)
            self.g0, self.I00 = p["g"], p["I0"]
            self.net = None
        else:
            p = dict(RS)
            # The published cell with its 1 ms step is kept fixed under the ANNs. If time_scale is
            # trainable, one-step training collapses it to 0: v(t) = v(t-1), a persistence
            # predictor with 0.1 mV one-step error that cannot run continuously.
            self.step = IzhikevichStep(p, ())
            # output: x = x_rest + X_SCALE * net(window); x_rest holds the cell at -82 mV
            self.x_rest = equilibrium_current(p)
            head = [keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01))]
            if arch == "dense3":
                body = [keras.layers.Flatten()] + [keras.layers.Dense(units, activation="tanh") for _ in range(3)]
            else:
                front = [keras.layers.Dense(units, activation="tanh") for _ in range(3)] if arch == "dense_cnnlstm" else []
                body = front + [keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
                                keras.layers.Conv1D(units, 5, padding="causal", activation="relu"),
                                keras.layers.LSTM(units)]
            self.net = keras.Sequential(body + head, name="net")

    def build(self, input_shape):
        self.step.build(None)
        if self.net is not None:
            self.net.build((None, W, 2))
        else:
            self.g = self.add_weight(name="g", shape=(), initializer=keras.initializers.Constant(self.g0))
            self.I0 = self.add_weight(name="I0", shape=(), initializer=keras.initializers.Constant(self.I00))

    def current(self, window):
        if self.net is None:
            return self.g * (window[:, -1, 1] + self.I0)
        z = tf.stack([(window[:, :, 0] - V_MEAN) / V_SCALE, window[:, :, 1] / I_SCALE], -1)
        return self.x_rest + X_SCALE * self.net(z)[:, 0]

    def call(self, window):
        return self.step(window, self.current(window))

    def izh_params(self):
        return {n: float(getattr(self.step, n).numpy()) for n in ("a", "b", "c", "d", "time_scale", "Vs", "V0")}


def make_model(arch):
    init = None
    if arch == "izh":
        init = json.load(open(os.path.join(EXP, "izhikevich_subthreshold", "results", "spiking", "metrics.json")))["params"]
    m = WindowModel(arch, init)
    m(tf.zeros((1, W, 2)))
    return m


# ------------------------------------------------------------------ prediction
def predict_one_step(m, I, V, batch=8192):
    """(n, T) traces -> (n, T) one-step predictions (first 64 steps copied from ground truth)."""
    X, _ = windows_of_trace(I, V)
    pred = np.concatenate([m(tf.constant(X[k:k + batch])).numpy() for k in range(0, len(X), batch)])
    out = np.array(V, np.float32, copy=True)
    out[:, W:] = pred.reshape(V.shape[0], -1)
    return out


def predict_continuous(m, I, V, seed):
    """Ground truth for t < seed, then the model's own predictions fill the voltage column."""
    n, T = V.shape
    buf = np.array(V, np.float32, copy=True)
    I = np.asarray(I, np.float32)

    @tf.function(reduce_retracing=True)
    def f(win):
        return m(win)

    for t in range(seed, T):
        win = np.stack([buf[:, t - W:t], I[:, t - W + 1:t + 1]], -1)
        buf[:, t] = f(tf.constant(win)).numpy()
    return buf
