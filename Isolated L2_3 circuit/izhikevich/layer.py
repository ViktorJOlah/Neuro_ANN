"""Batched, differentiable Izhikevich (2003) top layer for Keras 3 (TensorFlow backend).

Rewrite of ``CustomLayer_for_Vu`` (top layer encoders/Izhikevich_experimental.ipynb).
What changed and why:

* **Published dynamics.** The forward pass is the 2003 MATLAB scheme step for step
  (see ``reference.py``): spike when v >= 30 mV, then v <- c and u <- u + d, two
  half-steps for v and u updated with the new v. The old layer added an artificial
  threshold (-35 mV) that clamped V to +40 mV for one step before resetting, and
  updated u with the old V, so its traces were not the published model's.
* **Batches.** ``tf.cond(self.V < threshold)`` only works on a scalar, which is why
  the old layer could not take a batch. Spike/reset is now elementwise arithmetic,
  so any (batch, time, cells) input works and every cell can have its own a, b, c, d.
* **Backpropagation.** The spike is a Heaviside step whose derivative is zero almost
  everywhere, and ``tf.cond`` branches carry no gradient through the decision. The
  forward pass still uses the exact step, but the backward pass uses a surrogate
  gradient (fast-sigmoid / SuperSpike, Zenke & Ganguli 2018), and the whole time
  loop runs inside ``keras.layers.RNN`` so gradients flow through time (BPTT) to the
  input current and to a, b, c, d.

The reset is written as ``s*c + (1-s)*v`` and ``u + s*d`` with s in {0, 1}; both are
exact in IEEE arithmetic, so the forward pass is bit-identical to the reference in
float64 (see tests/test_izhikevich.py).

Input: current, shape (batch, time, n_cells), in the paper's units (one value per
step of ``dt`` ms). Output: ``(v, spikes)``, each (batch, time, n_cells), or only
one of them with ``output_mode="v"`` / ``"spikes"``.
"""
from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from .reference import PRESETS, V_PEAK


def exact_graph_mode():
    """Stop TensorFlow's graph optimizer from rewriting float arithmetic.

    Eager calls are bit-identical to the reference as written. Inside tf.function
    (model.fit, model.predict) Grappler's arithmetic optimizer and constant folding
    regroup the update's float operations; the change is ~1 ulp per step, but spike
    times in this model are chaotic, so traces drift apart after a few spikes.
    Turning both passes off restores bit-identity. This is process-wide and costs
    little here (the graph is a small elementwise loop).
    """
    tf.config.optimizer.set_experimental_options(
        {"arithmetic_optimization": False, "constant_folding": False})


def make_spike_fn(scale: float):
    """Heaviside(x >= 0) forward, fast-sigmoid derivative 1/(1+|x|/scale)^2 backward."""

    @tf.custom_gradient
    def spike(x):
        s = tf.cast(x >= 0, x.dtype)

        def grad(dy):
            return dy / tf.square(1.0 + tf.abs(x) / scale)

        return s, grad

    return spike


def make_stable_ops(v_peak: float):
    """Forward-exact ops whose backward pass ignores the spike overshoot.

    With the published 1 ms step, v overshoots v_peak (to ~10^3 mV) on the spike step.
    The true Jacobian there (1 + h*(0.08 v + 5) per substep) is huge, and multiplying
    those through time makes the gradients explode (~10^12 on a 500-step trace).
    ``quad`` returns 0.04*v*v but back-propagates the slope at min(v, v_peak);
    ``below_peak`` returns v unchanged but passes gradient only where v < v_peak.
    """

    @tf.custom_gradient
    def quad(v):
        def grad(dy):
            return dy * 0.08 * tf.minimum(v, v_peak)

        return 0.04 * (v * v), grad

    @tf.custom_gradient
    def below_peak(v):
        def grad(dy):
            return dy * tf.cast(v < v_peak, dy.dtype)

        return tf.identity(v), grad

    return quad, below_peak


class IzhikevichCell(keras.layers.Layer):
    """One integration step for ``n_cells`` independent Izhikevich neurons. State (v, u)."""

    def __init__(self, n_cells, a, b, c, d, v0=-65.0, dt=1.0, n_substeps=2, v_peak=V_PEAK,
                 trainable_params=("a", "b", "c", "d"), surrogate_scale=10.0, detach_reset=True,
                 stable_grad=True, **kwargs):
        super().__init__(**kwargs)
        self.n_cells = int(n_cells)
        self.init_values = dict(a=a, b=b, c=c, d=d)
        self.v0, self.dt, self.n_substeps, self.v_peak = float(v0), float(dt), int(n_substeps), float(v_peak)
        self.trainable_params = tuple(trainable_params)
        self.surrogate_scale = float(surrogate_scale)
        self.detach_reset = bool(detach_reset)
        self.stable_grad = bool(stable_grad)
        self.state_size = [self.n_cells, self.n_cells]
        self.output_size = 2 * self.n_cells

    def build(self, input_shape):
        for name, value in self.init_values.items():
            arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (self.n_cells,)).copy()
            init = (lambda arr: lambda shape, dtype=None: tf.constant(arr, dtype=dtype))(arr)
            setattr(self, name, self.add_weight(name=name, shape=(self.n_cells,), initializer=init,
                                                trainable=name in self.trainable_params))
        self._spike = make_spike_fn(self.surrogate_scale)
        if self.stable_grad:
            self._quad, self._below_peak = make_stable_ops(self.v_peak)
        else:
            self._quad, self._below_peak = (lambda v: 0.04 * (v * v)), (lambda v: v)

    def get_initial_state(self, batch_size=None):
        v = tf.fill([batch_size, self.n_cells], tf.constant(self.v0, self.compute_dtype))
        return [v, tf.cast(self.b, self.compute_dtype)[None, :] * v]

    def call(self, I, states, training=False):
        v, u = states
        a, b, c, d = (tf.cast(p, v.dtype) for p in (self.a, self.b, self.c, self.d))
        # 1. spike detection and reset (published order: before integration)
        s = self._spike(v - self.v_peak)
        s_reset = tf.stop_gradient(s) if self.detach_reset else s
        v = s_reset * c + (1.0 - s_reset) * v
        u = u + s_reset * d
        # 2. integrate v in substeps, then u with the new v (same op order as reference.py)
        h = tf.constant(self.dt / self.n_substeps, v.dtype)
        for _ in range(self.n_substeps):
            v = v + h * (self._quad(v) + 5.0 * v + 140.0 - u + I)
        u = u + tf.constant(self.dt, v.dtype) * (a * (b * self._below_peak(v) - u))
        # spike output for this step: v >= v_peak (reset happens at the start of the next step)
        s_out = self._spike(v - self.v_peak)
        return tf.concat([v, s_out], axis=-1), [v, u]


class IzhikevichLayer(keras.layers.Layer):
    """Izhikevich neurons as a Keras layer over whole sequences.

    neuron: a preset name from ``reference.PRESETS`` ("RS", "FS", ...) or a dict with
    a, b, c, d (scalars or length-n_cells lists, for heterogeneous populations).
    input_gain: trainable per-cell scale on the input current (starts at 1, exact).
    exact: call ``exact_graph_mode()`` so graph-mode runs stay bit-identical too.
    Other keyword arguments go to ``IzhikevichCell`` (dt, n_substeps, v0, v_peak,
    trainable_params, surrogate_scale, detach_reset, stable_grad).
    """

    def __init__(self, n_cells=1, neuron="RS", output_mode="both", input_gain_trainable=False,
                 unroll=False, exact=True, **cell_kwargs):
        super().__init__(dtype=cell_kwargs.pop("dtype", None), name=cell_kwargs.pop("name", None))
        if exact:
            exact_graph_mode()
        params = dict(zip("abcd", PRESETS[neuron])) if isinstance(neuron, str) else dict(neuron)
        self.n_cells, self.output_mode = int(n_cells), output_mode
        self.input_gain_trainable = input_gain_trainable
        self.cell = IzhikevichCell(n_cells, dtype=self.dtype_policy, **params, **cell_kwargs)
        self.rnn = keras.layers.RNN(self.cell, return_sequences=True, unroll=unroll,
                                    dtype=self.dtype_policy)

    def build(self, input_shape):
        self.input_gain = self.add_weight(name="input_gain", shape=(self.n_cells,), initializer="ones",
                                          trainable=self.input_gain_trainable)

    def call(self, I, training=False):
        I = tf.convert_to_tensor(I, self.compute_dtype)
        if I.shape.rank == 2:  # (batch, time) -> one cell
            I = I[..., None]
        out = self.rnn(I * tf.cast(self.input_gain, I.dtype), training=training)
        v, s = out[..., : self.n_cells], out[..., self.n_cells:]
        return {"v": v, "spikes": s}.get(self.output_mode, (v, s))
