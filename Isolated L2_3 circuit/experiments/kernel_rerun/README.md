# Rerun with the synaptic kernel layer as the default (steps 0-4)

From 2026-11-26 the default architecture is the Standard model plus the hardcoded synaptic kernel layer
(PR 22, `experiments/evoked_synaptic/code/synaptic_kernel.py`). Synaptic events no longer pass through the
CNN-LSTM front end. Each one drives a conductance, I = g·(E_rev − Vm), straight into the Izhikevich layer.
This folder reruns every earlier step that fed synaptic input through the front end.

## Step 0: membrane correction from the somatic model alone (`../kernel_calibration/`)

The bare Izhikevich membrane responds more slowly than the cell, and the front end normally corrects for that.
Synaptic current that bypasses the front end therefore goes through a per-cell correction:

    I_izh = k0·I_syn + c1·LP_τ1(I_syn) + c2·LP_τ2(I_syn)        (LP = unit-area exponential low-pass)

Library cells have no evoked recordings, so `calibrate.py` fits (k0, c1, τ1, c2, τ2) from each somatic model S
alone. It asks the bypass path to reproduce S's own response to subthreshold DC plus Poisson trains of alpha PSC
currents, with the current going through the front end. PSC sizes scale with the cell's rheobase. 30 ms after
each spike is masked, and held-out validation traces pick the iterate. If no iterate beats "no correction", the
identity is kept.

- 48 models: the 41 library cells, plus the 7 other fits of the 8-fit Aim 2 set.
- Median held-out RMSE falls from 2.59 to 1.26 mV. The fit is better in 43 models; in 5 no iterate beat no correction, so the identity was kept.
- Typical correction (10th / 50th / 90th percentile):
  - k0: 0.89 / 0.94 / 1.00
  - c1: −0.14 / −0.08 / 0.00, with τ1 4.7 / 6.6 / 9.9 ms
  - c2: −0.28 / −0.11 / 0.00, with τ2 35 / 47 / 62 ms
  - The DC gain, k0 + c1 + c2, ranges from 0.3 to 1.0 (Standard: 0.59).
- Check against the evoked recordings (`kernel_soft_corr`, 1 kernel component plus this correction):
  - 754: Vm RMSE 0.71 mV, vs 0.68 for the PR 22 two-component kernel and 0.67 for tcn64 D.
  - 057: 1.88 mV, vs 1.86 and 1.68.
  - L2/3 peak ratio 0.88 (754) and 0.86 (057), with no pre-EPSP dip.

Figure: `results/fig_step0_calibration.png`; numbers in `results/step0_summary.json`.

## Step 1: hybrid L2/3 circuit in kernel mode (`circuit/hybrid.py --synapses kernel`, the default)

The Allen alpha PSCs of each receptor port become conductances, normalised so that each one gives its Allen
current at V_ref = −55 mV (E_exc 0 mV, E_inh −80 mV). These conductances go through the kernel path, with the
correction applied; the front end sees zero current. Kernel mode is about 8× faster per ms than the current path.
`tests/test_kernel_mode.py` checks the batched step against `run_syn`.

The tuned settings are gain 1.18 onto PCs, and PC→IN Pvalb 1.06, Sst 1.04 and Htr3a 1.04. Rates compared with
the GLIF network:

| class | difference from GLIF |
|---|---|
| E | +0.1% |
| Pvalb | −0.8% |
| Sst | −1.5% |
| Htr3a | +1.8% |

The per-cell rate correlation with the GLIF network is r = 0.96, the top-10% share is 0.30, and 28% of cells are
silent. Results are in `results/step1_circuit/`.

## Step 2: in vivo comparison (595 Neuropixels V1 L2/3 RS units)

Rates at the matched mean:

| | in vivo | GLIF | ANN, current path (PR 14) | ANN, kernel layer |
|---|---|---|---|---|
| top-10% share | 0.50 | 0.33 | 0.30 | 0.30 |
| p99 (Hz) | 12.1 | 5.8 | 5.5 | 5.3 |
| max (Hz) | 23.3 | 11.7 | 9.5 | 8.6 |
| KS shape, active cells | | 0.19 | 0.19 | 0.19 |
| Fano, median (10 trials, 1-3 s) | 2.65 | 0.35 | 0.35 | 0.34 |

The homogeneous network still has a short upper tail, and its trial-to-trial variability is still far below in
vivo. `run_trials.sh` runs grating trials 1-9, and `fano.py` computes the Fano factors. Figures:
`results/fig_step2_invivo.png` and `results/fig_step2_fano.png`.

## Step 3: Aim 2 controls (`../aim2_controls/results_kernel`, `results_library_kernel`)

`make_kernel_populations.py` gives every model its own correction. Each condition was retuned to the homogeneous
kernel PC rate (1.378 Hz), with PC→IN at the step 1 values. The commands are in `run_aim2_kernel.sh`.

| condition | gain | top-10% share | p99 (Hz) | silent | identity R² | input-drive R² |
|---|---|---|---|---|---|---|
| homogeneous | 1.18 | 0.30 | 5.6 | 0.28 | | |
| 8 fits, shuffled | 1.62 | 0.43 (current path 0.45) | 7.2 | 0.59 | 0.68 | 0.12 |
| 8 fits, depth-placed | 1.62 | 0.43 | 7.2 | 0.59 | 0.69 | 0.12 |
| 8 fits, homogeneous interneurons | 1.62 | 0.44 | 7.2 | 0.59 | 0.68 | 0.12 |
| 8 fits, rheobase-matched | 1.05 | 0.44 (current path 0.35) | 8.8 | 0.44 | 0.31 | 0.48 |
| 8 fits, rheobase + DC gain matched | 1.37 | 0.36 | 7.2 | 0.36 | 0.16 | 0.67 |
| library, shuffled | 1.00 | 0.63 (current path 0.63) | 11.2 | 0.69 | 0.81 | 0.05 |
| library, depth-placed | 1.00 | 0.63 | 11.2 | 0.69 | 0.81 | 0.05 |
| library, rheobase-matched | 1.12 | 0.44 (current path 0.39) | 9.2 | 0.43 | 0.38 | 0.40 |
| library, rheobase + DC gain matched | 1.35 | 0.44 | 9.6 | 0.40 | 0.30 | 0.51 |

The heterogeneity conclusions are unchanged. Heterogeneity raises the top-10% share mainly through silent cells.
Cell identity, not input drive, explains the rates, and depth placement has no effect.

The rheobase-matched control changes. In kernel mode, synaptic drive reaches each model scaled by its correction's
DC gain, and that gain differs between models (0.3 to 1.0). Matching DC rheobase alone therefore leaves a
difference in excitability: a model's rate follows its DC gain, with Spearman 0.77. The `rheo_matched_dc` control
also divides each PC's input by its DC gain, relative to the Standard's.

- 8-fit set: this restores the earlier result (share 0.36, identity R² 0.16).
- Library: the share stays at 0.44. With 38 cells, the response to synaptic input differs beyond rheobase and DC
  gain, for example in f-I slope and adaptation, so rheobase is not the whole story there.

## Step 4: evoked data amount with the kernel layer

The soft kernel plus the S-derived correction was fitted from scratch on 2 min, 5 min or all of the evoked AMPA
data, for cells 754 and 057:

| cell, data | Vm RMSE (mV) | L2/3 peak ratio | L1 peak ratio |
|---|---|---|---|
| 754, 2 min | 0.74 | 0.78 | 0.68 |
| 754, 5 min | 0.72 | 0.89 | 0.69 |
| 754, full | 0.72 | 0.89 | 0.74 |
| 057, 2 min | 1.95 | 0.66 | 0.82 |
| 057, 5 min | 1.99 | 0.75 | 0.91 |
| 057, full | 2.00 | 0.86 | 1.01 |

For Vm error, 5 min matches the full data. EPSP amplitude needs more data: with 2 min the peaks come out too
small. Figure: `results/fig_step4_data_amount.png`; numbers in `results/step4_summary.json`.

## Reproduce

```bash
cd "Isolated L2_3 circuit/experiments/kernel_calibration"; bash run_library.sh; bash run_fits8.sh
cd ../aim2_controls; python make_kernel_populations.py
cd ../..; python -m circuit.hybrid tune --prep H --standard-dir Standard --reference GLIF_SPIKES \
    --corr experiments/kernel_calibration/results/487661754/calibration.json --pc-to-in-gain "Pvalb=1.06,Sst=1.04,Htr3a=1.04" ...
bash experiments/kernel_rerun/run_aim2_kernel.sh <job>    # shuf depth shin rheo rheodc lshuf ldepth lrheo lrheodc hin
bash experiments/kernel_rerun/run_trials.sh; python experiments/kernel_rerun/fano.py
python experiments/kernel_rerun/plot_rerun.py
```

A population hybrid run needs about 7 GB, and a full GLIF network trial up to 13 GB, so on a 15 GB machine run
them one at a time.
