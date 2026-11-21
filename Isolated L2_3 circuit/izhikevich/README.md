# Izhikevich top layer

A Keras 3 (TensorFlow backend) layer that runs the published Izhikevich (2003) neuron on
batches of input current and can be trained by backpropagation. It replaces
`CustomLayer_for_Vu` in `top layer encoders/Izhikevich_experimental.ipynb`.

```python
from izhikevich import IzhikevichLayer

# ANN -> current -> spiking cell. I: (batch, time, n_cells), one value per 1 ms step.
layer = IzhikevichLayer(n_cells=1, neuron="RS")          # or "IB", "CH", "FS", "LTS", "TC", "RZ"
v, spikes = layer(I)                                      # each (batch, time, n_cells)

# heterogeneous population, only d and the input gain trained
layer = IzhikevichLayer(3, {"a": [0.02, 0.02, 0.1], "b": 0.2, "c": [-65, -55, -65], "d": [8, 4, 2]},
                        trainable_params=("d",), input_gain_trainable=True)
```

Run the checks from `Isolated L2_3 circuit/`: `python -m pytest izhikevich/tests -q` (about 1 minute on CPU).

## What was wrong with the old layer, and what changed

| old `CustomLayer_for_Vu` | new `IzhikevichLayer` |
|---|---|
| Extra threshold at -35 mV clamps V to +40 mV for one step, then resets at the next | Published rule: spike when v >= 30 mV, then v <- c, u <- u + d |
| u updated with the old V | u updated with the new v, as in the paper |
| `(0.04V + 5)V`, one Euler step of 0.5 ms | Paper's operation order, two 0.5 ms half-steps for v per 1 ms (`dt`, `n_substeps` configurable) |
| `tf.cond` on V only works on a scalar, so no batches | Elementwise spike/reset, any batch size, per-cell a, b, c, d |
| No gradient through the spike decision | Surrogate gradient for the spike, BPTT through the whole trace via `keras.layers.RNN` |

## Is it numerically identical to the published model?

Yes, bit-identical: maximum absolute difference **0.0** (tolerance 0, `assert_array_equal`).

* `reference.py` is the paper's MATLAB loop in NumPy. A test checks it against a literal,
  line-by-line transcription of that loop, with zero difference.
* The layer matches the reference with zero difference for all 7 cell classes in Fig. 2 of
  the paper (RS, IB, CH, FS, LTS, TC, RZ), under step currents of 0 to 15 and a noisy current,
  in one batched call with different parameters per cell. This holds in float64 and in float32
  (each against the reference run in the same precision), eagerly and inside `model.predict`.
* A batched call gives exactly the same traces as calling each sample alone.
* The firing patterns behave as in Fig. 2 (RS adapts, IB bursts then fires tonically, CH
  bursts repeatedly, FS fires fast).

Three things are needed for bit-identity, and the layer handles the first itself:

1. **TensorFlow's graph optimizer.** Inside `tf.function` (so inside `fit` and `predict`) Grappler
   regroups float operations by about 1 ulp per step. `IzhikevichLayer(exact=True)` (the
   default) turns off its arithmetic optimization and constant folding for the process.
2. **XLA.** Compile with `jit_compile=False`. XLA fuses multiply-adds and the traces diverge.
   Keras's default `"auto"` was exact on CPU here, but it may turn on XLA on a GPU. GPU runs were
   not tested.
3. **Precision.** The paper's MATLAB code runs in double precision. Use `dtype="float64"`, and
   `keras.config.set_floatx("float64")` when you pass NumPy arrays to `fit` or `predict`,
   because Keras otherwise casts them to float32. In float32 the model itself, not the layer,
   differs from the float64 model after about 5 spikes under a step of 10. The 1 ms scheme makes
   v overshoot to around 10^3 mV on a spike step, so spike times are very sensitive to rounding.

## Backpropagation

* The forward pass uses the exact Heaviside spike. The backward pass uses the fast-sigmoid
  surrogate `1/(1 + |v - 30|/scale)^2` (`surrogate_scale=10` mV).
* `detach_reset=True` (the default) stops gradient through the reset decision. Without it, the
  surrogate leaks a false gradient into subthreshold stretches, the fits went the wrong way or
  diverged, and the subthreshold gradient no longer matched finite differences.
* `stable_grad=True` (the default) keeps the forward pass exact. On the backward pass the spike
  overshoot is ignored: the quadratic's slope is taken at min(v, 30), and u gets gradient only
  where v < 30. Without it, gradients over a 500-step spiking trace reached about 10^12.
* Tested: below threshold, the gradients with respect to a, b and the input gain match central
  finite differences to 1e-5. With spikes, the gradients with respect to the input and all
  parameters are finite and nonzero. Starting from d = 8, `model.fit` on spike counts recovers a
  d = 4 cell to within 1. It reaches 3.6, and the loss falls from 18 to 1. Integer spike counts
  only pin d down to a range.
