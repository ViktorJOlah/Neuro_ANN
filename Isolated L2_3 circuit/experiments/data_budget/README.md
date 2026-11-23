# Data budget: how much recording does the Standard model need?

Roadmap v2, step 2. The **Standard** model (CNN-LSTM front end + Izhikevich layer + conductance
branches, `Isolated L2_3 circuit/Standard/`) is fitted to the 5 extra Allen L2/3 PCs
(`simulations/additional_cells/`, noisy traces) on 5, 10, 20 and 30 minutes of data and on the
full simulated pool. **Every fit starts from the same Standard weights and parameters**, with the
same optimiser settings and seeds; only the amount of training data changes.

> Dependency: `Standard/` lives on the PR #5 branch (`claude/project-thread-p7c2ak`). Until that
> is merged, pass its location with `--standard`.

## Method

**Split.** Per cell and protocol, 15 of 100 traces are held out as test traces (evenly spaced
indices), 45 per cell in all. The rest are put in a fixed random order (seeded by the specimen id).

**Budgets.** A budget of N minutes takes equal time from square, noise and ramp (water-filling, so
a protocol that runs out gives its share to the others), always the first traces of the fixed
order. The sets are nested: 5 min ⊂ 10 min ⊂ 20 min ⊂ 30 min ⊂ full.

| Cell | Training pool (min) | Budgets fitted |
|---|---|---|
| 318808427 | 19.8 | 0, 5, 10, full (20 and 30 min = full) |
| 486262299 | 47.3 | 0, 5, 10, 20, 30, full |
| 502359001 | 23.2 | 0, 5, 10, 20, full (30 = full) |
| 513531471 | 24.1 | 0, 5, 10, 20, full (30 = full) |
| 521409057 | 26.5 | 0, 5, 10, 20, full (30 = full) |

Ramps are 12 s long and squares 2 s, which is why the pools differ. "0 min" is the Standard unfitted.

**Input rescaling (applied before every fit).** The cells' rheobases differ a lot from the
Standard cell's, and the unfitted Standard fires hundreds of extra spikes on some cells. Before
training, the input is rescaled by k = 83.8 pA / (lowest spiking square amplitude in the training
set). It is folded exactly into the model (first Conv1D kernel × k, g × k, I0 / k), so the starting
model is still the Standard, just reading a rescaled current. k is stored in each `metrics.json`.

**Stage 1 (front end + Izhikevich a, b, g, I0, V0, Vs, time scale).** Closed-loop training
through the full model on 600 ms subthreshold segments (ending ≥10 ms before the first spike),
200 ms recorded-Vm seed, MSE of Vm clipped at −20 mV, Adam with cosine decay, 250 iterations of
batch 64.

**Stage 2 (spike parameters V_th, c, d, b_na, SK scale).** Front end frozen; Nelder-Mead
(120 evaluations, c ≤ V_th − 3) on 96 segments of 1.5 s (2/3 with spikes), loss = clipped-Vm MSE
+ smoothed spike-train MSE + 0.1 × spike-count MSE.

**Scoring.** Continuous free-running simulation of each held-out trace after a 450 ms recorded-Vm
seed, scored against the noisy and the clean NEURON trace: Γ coincidence factor (±4 ms),
|spike-count error| per trace, Vm RMSE clipped at −20 mV, and Vm RMSE away from spikes (5 ms before
to 30 ms after every model or NEURON spike removed).

## Results (medians over the 5 cells, scored on noisy test traces)

| Budget | Square Γ / \|Δn\| / RMSE away from spikes | Noise Γ / \|Δn\| / RMSE | Ramp Γ / \|Δn\| / RMSE |
|---|---|---|---|
| 0 min (Standard) | 0.08 / 5.2 / 10.0 mV | 0.14 / 19.5 / 5.4 mV | 0.03 / 29.9 / 8.3 mV |
| 5 min | 0.10 / 0.3 / 1.5 mV | 0.09 / 2.7 / 2.8 mV | 0.07 / 0.9 / 1.6 mV |
| 10 min | 0.07 / 0.5 / 1.8 mV | 0.05 / 3.9 / 2.9 mV | 0.13 / 2.5 / 1.4 mV |
| 20 min | 0.04 / 0.6 / 1.8 mV | 0.06 / 4.3 / 3.0 mV | 0.06 / 2.0 / 1.6 mV |
| full (20–47 min) | 0.07 / 0.5 / 2.0 mV | 0.08 / 4.9 / 2.6 mV | 0.03 / 3.3 / 1.6 mV |

- **5 minutes already gives nearly all of the gain.** Vm error away from spikes falls 2–7× and spike-count error
  7–30× from the unfitted Standard; more data does not improve the held-out scores further.
- **Spike timing (Γ) stays low at every budget (0.03–0.14),** like the Standard on its own cell
  (0.10–0.19). It is limited by the model and fitting procedure, not by the amount of data.
- **Noise cost is negligible at this noise level.** The added patch noise is 0.06–0.08 mV RMS, far
  below the 1.5–3 mV model error, so noisy and clean scores agree to < 0.01 mV. A clean-trained
  control was not run.
- 318808427 under-fires on noise traces at every budget (≈ −15 spikes per trace).

## Files

- `fit_budget.py --cell <id> --minutes 0|5|10|20|30|full --standard <Standard dir>` — one fit
  (≈ 35 min on 1 CPU thread, ≈ 5 GB RAM; more than 2 in parallel ran out of memory on 16 GB).
- `plot_budget.py` — `learning_curves.png`, `noise_cost.png`, `parameters.png`,
  `examples_<cell>.png`, `summary.csv` in `results/`.
- `results/<cell>/<budget>/` — `metrics.json` (train/test trace indices, k, fitted parameters,
  scores, training history), `test_predictions.npz` (model Vm on the test traces),
  `front_end.npz` (fitted front-end weights).
