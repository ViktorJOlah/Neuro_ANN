# Standard ANN vs Allen GLIF and all-active NEURON models on real recordings

Question (Viktor, 2026-11-23): fitted to real patch-clamp recordings, can the Standard ANN beat the Allen
Institute's own GLIF and full biophysical models of the same cells?

## Cells and sources

Six VISp L2/3 spiny (pyramidal) cells from the Allen Cell Types Database that have both published GLIF models
and an all-active NEURON model:

| cell | GLIF levels | all-active model |
|---|---|---|
| 513531471 | 1-5 | Nandi et al. 2022 hof #0 |
| 485468180 | 1-5 | Nandi et al. 2022 hof #0 |
| 502269786 | 1-5 | Nandi et al. 2022 hof #0 |
| 318808427 | 1, 3 | Nandi et al. 2022 hof #0 (Allen also released one) |
| 521409057 | 1, 3 | Nandi et al. 2022 hof #0 |
| 382982932 | 1, 3 | Nandi et al. 2022 hof #0 (Allen also released one) |

The Nandi et al. all-active models (github.com/AllenInstitute/All-active-Manuscript) are used for all six so that
every cell has the same kind of biophysical model. They run with `simulations/additional_cells/code/allactive_cell.py`
(stub axon, 34 °C, junction potential −14 mV) at dt 0.025 ms. GLIF runs from Allen's own code (vendored unchanged
apart from imports in `code/glif/`) at the model dt of 50 µs.

## Data and split

`code/build_dataset.py` turns each NWB (50 kHz) into 1 ms sweeps: Long Square + 2 s suprathreshold square
("square"), Noise 1 + Noise 2 ("noise"), Ramp ("ramp"), each cropped to stimulus ±0.5 s.

* **Test** (every model is scored here): all Noise 2 sweeps (identical repeats), every third Long Square by
  amplitude, the last Ramp.
* **Train** (only the Standard is trained): the rest, which is only **1.8–3 min per cell**.
* The NWB stimulus is used without the small bias/leak current, as in Allen's own fits.

## The Standard fit

`code/fit_standard.py` uses the data-budget recipe (`experiments/data_budget/fit_budget.py`): start from the
Standard weights, rescale input by rheobase, stage 1 = 250 Adam steps on subthreshold segments, stage 2 =
Nelder–Mead (120 evaluations) on the spiking parameters. Each fit takes about 31 min. `--unfitted` gives the
Standard with no refit as a control.

## Fairness caveats

* GLIF was fit on Noise 1 and tested by Allen on Noise 2, so Noise 2 is truly held out for GLIF too.
* The all-active models were fit on long-square features, so the test long squares were part of their fit.
* "GLIF best" is the GLIF level with the best Noise 2 Γ per cell, chosen on the test data (generous to GLIF).
* The Standard had 2–3 min of training data; the data-budget study suggested about 5 min is needed.
* Cell 485468180 is unreliable: in Noise 2 it fired 6 and 3 spikes in its two repeats (Γ between repeats ≈ 0),
  its rheobase is ~35 pA, and the Standard fit failed on it (it fires before the stimulus). The headline medians
  below leave it out; `results/metrics_summary.csv` has every cell.
* 513531471 runs down during the recording (Noise 2 repeats go from 21 to 8 spikes).

## Results (medians over the 5 reliable cells, test sweeps only)

| metric | Standard | all-active | GLIF best | GLIF3 |
|---|---|---|---|---|
| Noise: Γ (±4 ms coincidence) ↑ | 0.12 | 0.27 | **0.40** | 0.33 |
| Noise: spike-train EV / repeat ceiling ↑ | 0.23 | 0.42 | 0.57 | **0.69** |
| Noise: \|spike-count error\| per sweep ↓ | **7.0** | 42 | 13.7 | 8.5 |
| Noise: Vm EV away from spikes ↑ | 0.89 | 0.90 | **0.98** | **0.98** |
| Noise: Vm RMSE away from spikes (mV) ↓ | 3.6 | 4.6 | **2.0** | **2.0** |
| Square: Vm EV away from spikes ↑ | **0.95** | 0.85 | 0.93 | 0.90 |
| Square: Vm RMSE away from spikes (mV) ↓ | **2.7** | 3.3 | 4.0 | 4.5 |
| Square: subthreshold steady-state error (mV) ↓ | **1.5** | 3.8 | 3.0 | 3.1 |
| Square: \|spike-count error\| ↓ | 2.2 | **0.5** | 3.2 | 2.4 |
| Square: first-spike latency error (ms) ↓ | 50 | 15 | 20 | **7** |
| Square / ramp Γ | ≈0 for every model (spikes drift over 1 s steps) | | | |

The cell's own repeat-to-repeat Γ on Noise 2 is 0.60–0.71, so no model is near the ceiling.

**Verdict.** The Standard beats both Allen models on subthreshold voltage during steps (EV, RMSE,
steady state) and has the smallest spike-count error on noise, where the all-active models over-fire 1.5–8×.
It loses clearly on spike timing: on Noise 2 its Γ is less than half of the all-active model's and about a third
of GLIF's, and its first-spike latency on steps is the worst. With 2–3 min of training data and a
count/voltage-driven fit, the Standard does not yet beat the biophysical models where it matters most for network
work (spike timing). The unfitted Standard is far worse on everything (noise Vm EV 0.07), so the refit itself works.

## Files

* `code/fetch_allen.py`: downloads NWB, sweeps table, GLIF configs, SWC, model list (and `--nandi` parameters).
* `code/build_dataset.py`, `code/fit_standard.py`, `code/run_allen_models.py`, `code/compare.py`.
* `data/<cell>/`: 1 ms sweeps, split, metadata, model inputs (raw NWBs are not in git; they are in
  `/mnt/project-files/real_cells_nwb/`).
* `results/<cell>/<model>/predictions.npz`, `results/metrics_per_sweep.csv`, `results/metrics_summary.csv`,
  `results/summary.json`, `results/figures/`.

Reproduce from "Isolated L2_3 circuit/":

```
python experiments/real_cells/code/fetch_allen.py --nwb-dir <dir> --nandi
python experiments/real_cells/code/build_dataset.py --nwb-dir <dir>
python experiments/real_cells/code/run_allen_models.py --cell <cell> --nwb-dir <dir>
python experiments/real_cells/code/fit_standard.py --cell <cell>        # and --unfitted
python experiments/real_cells/code/compare.py
```
