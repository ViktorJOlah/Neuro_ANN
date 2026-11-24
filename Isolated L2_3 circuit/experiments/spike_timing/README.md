# Spike timing of the Standard on real L2/3 recordings

Goal (2026-11-24): on the real Allen recordings of PR 11 (`experiments/real_cells/`), the fitted Standard won on
subthreshold Vm and spike counts but lost on spike timing (Noise 2 Γ 0.12 vs 0.27 all-active, 0.40 GLIF best).
Raise Γ without giving up the subthreshold and count advantage, keeping the Standard architecture (CNN-LSTM
front end, Izhikevich top layer, conductance branches, 1 ms steps, 64 ms current-only input).

Cells: the 5 reliable cells of PR 11 (485468180 left out, as there). Everything is scored by
`experiments/real_cells/code/compare.py` on the same held-out test sweeps as PR 11.

## Result

The gap is **not** closed. The best balanced recipe ("timing refit") moves the median Noise 2 Γ from 0.12 to
0.16, keeps the spike counts (|count error| 6.5 vs 7.0) and the step subthreshold fit, and improves the noise
Vm and the step first-spike latency. It helps 3 cells a lot (318808427 0.11 → 0.16, 382982932 0.15 → 0.27,
521409057 0.08 → 0.13), leaves 502269786 about the same and hurts 513531471 (the cell that runs down during the
recording). The all-active model (0.27) and GLIF (0.40) are still better.

| metric (median of 5 cells) | Standard (PR 11) | branch refit | **timing refit** | branch refit v2 | all-active | GLIF best |
|---|---|---|---|---|---|---|
| Noise 2 Γ ↑ | 0.12 | 0.17 | **0.16** | 0.21 | 0.27 | 0.40 |
| Noise 2 Γ / repeat Γ ↑ | 0.25 | 0.26 | 0.23 | 0.30 | 0.41 | 0.70 |
| Noise 2 spike-train EV / ceiling ↑ | 0.23 | 0.32 | **0.38** | -0.42 | 0.42 | 0.57 |
| Noise 2 spike-count bias (model − data) | -1.7 | +18 | +5.5 | +47 | +42 | -13.7 |
| Noise 2 \|count error\| ↓ | 7.0 | 18 | **6.5** | 47 | 42 | 13.7 |
| Noise Vm EV away from spikes ↑ | 0.89 | 0.91 | 0.91 | 0.94 | 0.90 | 0.98 |
| Noise Vm RMSE away from spikes (mV) ↓ | 3.58 | 3.58 | **3.24** | 2.47 | 4.62 | 1.98 |
| Square Vm EV away from spikes ↑ | 0.95 | 0.92 | 0.93 | 0.94 | 0.85 | 0.93 |
| Square steady-state error (mV) ↓ | 1.45 | 1.42 | 1.50 | 1.33 | 3.82 | 3.04 |
| Square \|count error\| ↓ | 2.2 | 2.4 | 2.3 | 3.0 | 0.5 | 3.2 |
| Square first-spike latency error (ms) ↓ | 50 | 35 | **37** | 29 | 15 | 20 |

Noise 2 Γ per cell (before → after):

| cell | Standard (PR 11) | branch refit | **timing refit** | branch refit v2 | all-active | GLIF best |
|---|---|---|---|---|---|---|
| 318808427 | 0.11 | 0.18 | **0.16** | 0.21 | 0.27 | 0.40 |
| 382982932 | 0.15 | 0.17 | **0.27** | 0.27 | 0.39 | 0.42 |
| 502269786 | 0.20 | 0.19 | **0.18** | 0.25 | 0.43 | 0.50 |
| 513531471 | 0.12 | 0.07 | **0.04** | 0.07 | 0.14 | 0.40 |
| 521409057 | 0.08 | 0.16 | **0.13** | 0.14 | 0.25 | 0.31 |

Noise 2 spike-count bias per sweep: see `results_table.md`. "Branch refit v2" has the highest Γ (0.21) and the
best noise Vm, but fires 1.5–2× too much (+32 to +103 spikes per 20 s sweep), so it is not the recommended model.

## Diagnosis: why the PR 11 fits time spikes badly

`diagnose.py` and `residual_by_spike.py` run the fitted models *spike-forced*: the recorded Vm drives the
model from 2 ms before to 4 ms after every recorded spike (`timing_lib.run_forced`, the same clamp the seed
period uses), so the model always has the true spike history and only has to predict Vm, and the next spike, in
between.

* Given the true history, the PR 11 fits almost never fire (forced Γ ≈ 0): in noise their Vm 20–320 ms after a
  spike is 3–16 mV below the recording, and 2 ms before recorded spikes it sits at -42 to -63 mV.
* Branch ablation (spike-forced, 382982932): turning SK off flips that error to +4..+11 mV; K_P, Im, Kv3.1, BK,
  Na and d barely matter. **SK is 2.5–5× too strong.**
* In free runs the fit hides this behind a high reset (c ≈ -42 mV, well above the recorded post-spike Vm of
  -46 to -51 mV), which keeps the cell near threshold. Firing then follows the reset, not the input, which is
  why counts were fine but timing was not.

## Recipe ("timing refit")

1. `fit_branches.py` (front end frozen, about 3 min per cell): Nelder–Mead on spike-forced 1000 ms segments
   (half from the noise sweeps, spiking parts included) over the SK, Ca and K_P scales, d and the input offset
   I0, minimising Vm MSE with both traces clipped at -40 mV. SK comes out at 0.2–0.4× its PR 11 value.
   Output: `results/<cell>/standard_branches/`. `--set v2` also fits the Im, Kv3.1 and Na scales and an SK
   bias (`branches_v2/`).
2. `fit_spikes.py --mode smooth` (front end and branches frozen, about 4 min per cell): the PR 11 stage-2 loss
   (clipped Vm MSE + 10 ms-smoothed spike-train MSE + 0.1 × count MSE) over V_th, c, d, Na bias and Na scale on
   free-running 2 s segments (3/4 noise). This brings the counts back.
   Output: `results/<cell>/standard_timing/`.

Reproduce from "Isolated L2_3 circuit/" (after the PR 11 fits exist):

```
python experiments/spike_timing/fit_branches.py --cell <cell>
python experiments/spike_timing/fit_spikes.py --cell <cell> --start standard_branches --mode smooth --evals 150 --tag standard_timing
python experiments/spike_timing/fit_branches.py --cell <cell> --set v2 --evals 400 --tag branches_v2
python experiments/real_cells/code/compare.py
python experiments/spike_timing/final_table.py
```

## What did not work

* **Spike-forced front-end fine-tune by gradient** (`fit_timing.py` stage A, 150 Adam steps through the
  spike-forced loop): with the PR 11 branches it barely moved (val RMSE 5.5 → 5.1 mV); starting from the
  corrected branches, the validation error *grew* (318808427 2.6 → 6.6 mV in 50 steps) and the runs were
  stopped. Gradients through the threshold/reset loop are not reliable here, as in earlier spiking work.
* **Tighter smoothed spike loss** (3 ms, `fit_timing.py` stage B): the cells fire 18–43 % less and Γ drops.
* **Direct Γ objective on free runs** (`fit_spikes.py --mode free`): Nelder–Mead finds near-silent models
  (val Γ 0.37 → 0.0 on 382982932).
* **Γ objective on spike-forced runs** (`--mode forced`, with or without `--coinc 4`): train Γ 0.6–0.7 given the
  true history, but in free runs the cells fire 1.5–2× too much.
* **Count-weighted smoothed loss with SK free** (from v2): counts fixed, Γ back to 0.13.

## Interpretation and next steps

The main fault was SK strength, and fixing it is what moves Γ. After the fix, the spike-forced Vm in noise is
still off by about 4–6 mV RMS 10–80 ms after a spike (now slightly too depolarised), and no spike-parameter
objective turns the better Vm into both good timing and correct counts. GLIF gets 0.40 with a noise Vm RMSE of
2.0 mV; our best is 2.5 mV (v2) to 3.2 mV. So the remaining gap is the inter-spike Vm trajectory in noise. It
needs a front-end refit that sees the noise sweeps with the true spike history, which the gradient route
did not deliver. Options: (a) many more noise minutes per cell (the PR 11 cells have only 1–1.3 min of
Noise 1, and the planned recordings could add more), (b) a derivative-free or teacher-forced front-end fit, or
(c) a GLIF-style threshold readout on top of the Standard's Vm.

513531471 runs down during the recording (Noise 2 repeats go from 21 to 8 spikes), so every model over-fires on
its test sweeps. It should be read with care.

## Files

`timing_lib.py` (spike-forced loop, loaders, Γ), `diagnose.py`, `residual_by_spike.py`, `fit_branches.py`,
`fit_spikes.py`, `fit_timing.py` (round 1, documented above), `summarize.py` (quick table from predictions),
`final_table.py` → `results_table.md`, `logs/`. Model folders are in `experiments/real_cells/results/<cell>/`
(`standard_branches`, `standard_timing`, `branches_v2`). `compare.py` now scores them too, and its figures show
the timing refit next to the PR 11 fit.
