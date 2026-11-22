# Standard cell model ("The Chosen One")

The model Viktor chose on 2026-11-22 as the standard architecture for every L2/3 PC: a CNN-LSTM front end
reads a 64 ms current window and drives an Izhikevich layer that holds Vm and decides spikes. Stateful
conductance branches from the Allen mod files (NaTs+Nap, Ca→SK, Kv3.1, BK, K_P, Im, Ih) read the
Izhikevich Vm and add current. Everything runs at 1 ms.

Fitted to: Allen perisomatic model 488462783 (specimen 487661754, Cux2 L2/3 PC), NEURON square, ramp and noise sims.

## Files
- `standard_model.py`: self-contained model (numpy + TensorFlow/Keras only). `load()`, `cell.current(I)`, `cell.simulate(I, V_seed=None, seed_ms=0)`.
- `channels.py`: branch kinetics (Allen mod files, 34 °C), needed by `standard_model.py`.
- `params.json`: Izhikevich parameters (`izhikevich`), branch scales, biases, threshold and reset (`branches`), and test metrics.
- `weights/front_end_cnnlstm.npz`: CNN-LSTM weights (Conv1D 32×5 ×2 → LSTM 32 → Dense 1), loaded by `standard_model.py`.
- `weights/front_end_cnnlstm_full.weights.h5`: the original Keras checkpoint, for `experiments/current_window/run.py`.
- `export_standard.py`: rebuilds this folder from the experiment results and checks it reproduces them (max |ΔVm| 0.0000 mV, 0 spike mismatches on 24 test traces).

## Use
```python
import sys; sys.path.insert(0, "Standard")
import standard_model as sm
cell = sm.load()
vm, spikes = cell.simulate(I)                          # I: (n, T) pA at 1 ms, from rest
vm, spikes = cell.simulate(I, V_seed=V, seed_ms=450)   # recorded Vm drives it for 450 ms first
cell.sk_on.assign(0.0)                                 # SK block (also na_on, kv_on, bk_on, kp_on, im_on, ih_on)
```

## Fitted values
- Izhikevich: a 3.014, b 0.0223, time_scale 0.0152, Vs 3.616, V0 207.46, g 0.1856, I0 6.787
- Spike: V_th -26.82 mV, reset c -43.32 mV, d -0.129
- Branch scales (× Allen soma density): na 1.02, sk 0.763, ca 1.26, kv 0.106, bk 0.0163, kp 0.249, im 1e-12, ih 1e-12. Na bias 0.819.

## Test performance (continuous prediction, 450 ms seed, held-out traces)
| protocol | Γ (±4 ms) | spike-count bias |
|---|---|---|
| square | 0.10 | +0.28 |
| noise | 0.19 | -0.18 |
| ramp | 0.19 | -0.01 |

Spike-free continuous RMSE: square 0.50 mV, noise 0.70 mV.
Known flaw: on strong steps (≥180 pA) the second spike comes ~25 ms after the first (NEURON ~52 ms).

## How it was trained (the recipe for other cells)
1. Front end: `experiments/current_window/run.py --model cnnlstm` on spike-free square, noise and ramp traces,
   closed loop, Vm MSE after 200 ms. ANN outputs the Izhikevich input current; Vm lives only in the Izhikevich state.
2. Branches + spiking: ANN frozen; `experiments/conductance_branches/fit_currents.py --model cnnlstm --slow` (least squares over branch
   scales, bias and d on the inverted Izhikevich input, then Nelder-Mead on V_th, c, bias against the free-running spike loss),
   then `--keep-base --isi-loss 1.0` (adds the first-ISI penalty on strong steps).
These scripts currently read the first cell's dataset; pointing them at another cell's sims is the next step.
