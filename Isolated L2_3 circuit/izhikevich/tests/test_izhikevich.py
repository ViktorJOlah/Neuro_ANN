"""Checks of the Izhikevich layer against the published model.

Run from "Isolated L2_3 circuit/":  python -m pytest izhikevich/tests -q
"""
import numpy as np
import pytest
import keras
import tensorflow as tf

from izhikevich import PRESETS, IzhikevichLayer, simulate

NAMES = list(PRESETS)


def paper_loop(I, a, b, c, d):
    """The MATLAB listing of Izhikevich (2003), literally (1 ms steps, two 0.5 ms v steps)."""
    v, u = -65.0, b * -65.0
    out = []
    for t in range(len(I)):
        if v >= 30:
            v = c
            u = u + d
        v = v + 0.5 * (0.04 * (v * v) + 5 * v + 140 - u + I[t])  # MATLAB v.^2 == v.*v
        v = v + 0.5 * (0.04 * (v * v) + 5 * v + 140 - u + I[t])
        u = u + a * (b * v - u)
        out.append(v)
    return np.array(out)


def currents(T=1000, seed=0):
    """Steps of several amplitudes plus a noisy trace (random, non-representable values)."""
    rng = np.random.default_rng(seed)
    out = []
    for amp in (0.0, 3.0, 5.0, 10.0, 15.0):
        I = np.zeros(T)
        I[100:800] = amp
        out.append(I)
    out.append(rng.normal(6.0, 4.0, T))
    return np.stack(out)  # (6, T)


def heterogeneous_layer(dtype, **kw):
    a, b, c, d = (list(p) for p in zip(*PRESETS.values()))
    return IzhikevichLayer(len(NAMES), {"a": a, "b": b, "c": c, "d": d}, dtype=dtype, **kw)


def reference_batch(I, dtype=np.float64):
    """(batch, T, n_presets) reference traces: every current into every preset."""
    return np.stack([simulate(np.repeat(i[:, None], len(NAMES), 1),
                              *(np.array(p) for p in zip(*PRESETS.values())), dtype=dtype)[0]
                     for i in I])


def test_reference_is_the_published_loop():
    for name in NAMES:
        for I in currents():
            np.testing.assert_array_equal(simulate(I, *PRESETS[name])[0][:, 0], paper_loop(I, *PRESETS[name]))


@pytest.mark.parametrize("dtype,npdtype", [("float64", np.float64), ("float32", np.float32)])
def test_layer_bit_identical_to_reference(dtype, npdtype):
    """All 7 published cell classes in one batched, heterogeneous call: zero difference."""
    I = currents()
    layer = heterogeneous_layer(dtype)
    v, s = layer(np.repeat(I[..., None], len(NAMES), -1).astype(npdtype))
    v_ref = reference_batch(I, npdtype)
    np.testing.assert_array_equal(v.numpy(), v_ref)
    np.testing.assert_array_equal(s.numpy(), (v_ref >= 30).astype(npdtype))


def test_bit_identical_in_graph_mode_predict():
    """model.predict runs in a tf.function graph; exact_graph_mode keeps it bit-identical."""
    old = keras.config.floatx()
    keras.config.set_floatx("float64")  # otherwise Keras casts float64 inputs to float32
    try:
        I = currents(T=600)
        layer = heterogeneous_layer("float64", output_mode="v")
        inp = keras.Input((600, len(NAMES)))
        model = keras.Model(inp, layer(inp))
        model.compile(jit_compile=False)
        v = model.predict(np.repeat(I[..., None], len(NAMES), -1), batch_size=4, verbose=0)
        np.testing.assert_array_equal(v, reference_batch(I))
    finally:
        keras.config.set_floatx(old)


def test_batch_equals_one_by_one():
    I = currents()[..., None]
    layer = IzhikevichLayer(1, "RS", dtype="float64", output_mode="v")
    batched = layer(I).numpy()
    for k in range(len(I)):
        np.testing.assert_array_equal(batched[k], layer(I[k:k + 1]).numpy()[0])


def test_published_firing_patterns():
    """Qualitative sanity check of Fig. 2 behaviour under a 10-unit step."""
    I = np.zeros(1000)
    I[100:] = 10.0
    spikes = {n: np.flatnonzero(simulate(I, *PRESETS[n])[1][:, 0]) for n in NAMES}
    isi = {n: np.diff(t) for n, t in spikes.items()}
    assert isi["RS"][-5:].mean() > 1.2 * isi["RS"][0]        # RS: spike-frequency adaptation
    assert isi["IB"][0] <= 6 and isi["IB"][1:].min() >= 30   # IB: initial burst, then tonic
    assert (isi["CH"] <= 6).sum() >= 5 and (isi["CH"] >= 30).sum() >= 3  # CH: repeated bursts
    assert len(spikes["FS"]) > 3 * len(spikes["RS"])         # FS: high rate


def test_subthreshold_gradient_matches_finite_differences():
    """Without spikes the backward pass is the exact derivative of the published update."""
    rng = np.random.default_rng(1)
    I = tf.constant(rng.uniform(0.0, 2.0, (3, 300, 1)))
    layer = IzhikevichLayer(1, "RS", dtype="float64", output_mode="v", input_gain_trainable=True)
    layer(I)
    assert float(tf.reduce_max(layer(I))) < 0  # really subthreshold

    def loss():
        return tf.reduce_mean(tf.square(layer(I) + 60.0))

    with tf.GradientTape() as tape:
        L = loss()
    grads = dict(zip([w.name for w in layer.trainable_variables],
                     tape.gradient(L, layer.trainable_variables)))
    for w in layer.trainable_variables:
        if w.name in ("c", "d"):  # no spikes -> no effect
            assert float(tf.reduce_max(tf.abs(grads[w.name]))) == 0.0
            continue
        x0, eps = w.numpy(), 1e-6 * max(1.0, float(np.abs(w.numpy()).max()))
        w.assign(x0 + eps); lp = float(loss())
        w.assign(x0 - eps); lm = float(loss())
        w.assign(x0)
        fd = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(float(grads[w.name][0]), fd, rtol=1e-5)


def test_gradients_finite_and_nonzero_with_spikes():
    I = tf.constant(currents()[1:, :, None], tf.float32)
    layer = IzhikevichLayer(1, "RS", input_gain_trainable=True)
    with tf.GradientTape() as tape:
        tape.watch(I)
        v, s = layer(I)
        L = tf.reduce_sum(s) + 1e-3 * tf.reduce_sum(tf.minimum(v, 30.0))
    gs = tape.gradient(L, [I] + layer.trainable_variables)
    for g in gs:
        g = g.numpy()
        assert np.isfinite(g).all() and np.abs(g).max() > 0
        assert np.abs(g).max() < 1e6  # no explosion through the spike overshoot


def test_training_recovers_reset_parameter():
    """Backprop through spikes: fit d to spike counts of a d=4 cell, starting at d=8."""
    rng = np.random.default_rng(0)
    B, T = 16, 500
    I = np.zeros((B, T, 1), np.float32)
    I[:, 50:, 0] = rng.uniform(4, 15, B)[:, None]
    target = np.stack([simulate(I[k, :, 0], 0.02, 0.2, -65.0, 4.0)[1].sum(0) for k in range(B)])
    layer = IzhikevichLayer(1, {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}, output_mode="spikes",
                            trainable_params=("d",))
    inp = keras.Input((T, 1))
    model = keras.Model(inp, keras.ops.sum(layer(inp), axis=1))
    model.compile(keras.optimizers.Adam(0.1), "mse", jit_compile=False)
    hist = model.fit(I, target.astype(np.float32), epochs=60, batch_size=16, verbose=0).history["loss"]
    assert np.mean(hist[-5:]) < 0.2 * hist[0]
    assert abs(float(layer.cell.d.numpy()[0]) - 4.0) < 1.0
