# Population fit of the Standard (roadmap step 2)

The Standard model (CNN-LSTM front end, Izhikevich top layer, conductance branches; 1 ms, 64 ms current-only
input) fitted to every mouse VISp L2/3 spiny cell in the Allen Cell Types database (66 cells). This is a
**feasibility result**: it shows that the PR 11 + PR 13 pipeline runs unattended over a whole population and
yields a usable per-cell parameter library. It is not a headline accuracy number.

## Pipeline

| Step | Script | Output |
| --- | --- | --- |
| Cell list, soma depth | `select_cells.py` | `cells.csv` (66 cells; `depth_norm` = Allen normalized cortical depth, 0 = pia, 1 = white matter; 61 cells have it) |
| Download, 1 ms datasets | `real_cells/code/fetch_allen.py`, `build_dataset.py` | `real_cells/data/<cell>/` |
| Recording QC | `qc.py` | `qc.csv` |
| Fits, 2 cells in parallel | `run_population.py` | `real_cells/results/<cell>/{standard, standard_branches, standard_timing}` |
| Fit quality | `score.py` | `fit_quality.csv` |
| Library | `build_library.py` | `library/<cell>/{params.json, front_end.npz}`, `library/index.csv` |
| Figure | `plot_summary.py` | `summary_vs_depth.png` |

Per cell: `fit_standard` (PR 11, about 30 min) → `fit_branches` (PR 13) → `fit_spikes --mode smooth`
(PR 13 timing refit, saved as `standard_timing`). The library uses `standard_timing`. The whole run took
about 18 h of wall-clock time (2 fits in parallel, since 3 run out of memory). `run_population.py` skips any
step whose output exists, so it can be restarted at any point.

Two small fixes were needed for older Allen cells. Their 200 kHz NWBs store SI values with a wrong
conversion attribute (`build_dataset.py`), and some have no Noise 1 sweep, so `fit_spikes.py` now holds out
the longest spiking sweep instead.

## Recording QC (`qc.csv`)

- `rundown`: the last Noise repeat has < 60% of the first repeat's spikes, or the first-spike peak on Long
  Square drops > 10 mV from the first to the last spiking steps.
- `drift`: baseline moves > 5 mV across the recording (no cell hit this).
- `no_noise_test`: no Noise 2 sweep, so no Γ test (subthreshold and square metrics still exist).
- `sparse_noise`: < 10 spikes on Noise 2.

Result: 42 clean, 14 rundown, 7 no_noise_test, 2 rundown + no_noise_test, 1 rundown + sparse_noise.

## Fit flags (`fit_quality.csv`, held-out sweeps)

- `silent`: the model fires < 25% of the recorded test spikes.
- `overfire`: the model fires > 2× the recorded test spikes (at least 5).
- `prestim`: > 5 model spikes before stimulus onset.
- `poor_vm`: square subthreshold Vm RMSE > 5 mV.

Result: 49 clean, 9 overfire, 2 poor_vm, 2 overfire + poor_vm, 2 prestim + poor_vm,
1 overfire + prestim + poor_vm, 1 silent. Most flagged fits are on run-down recordings.

`use = 1` in `library/index.csv` means a clean fit on a clean recording (cells whose only flag is
`no_noise_test` are kept). **41 cells have `use = 1`**; 38 of them have a soma depth.

## Fit quality (medians, held-out sweeps)

| | All 66 | use = 1 (41) |
| --- | --- | --- |
| Noise 2 Γ (±4 ms) | 0.15 | 0.17 |
| Repeat-to-repeat Γ of the cell itself | 0.48 | 0.59 |
| Γ / repeat Γ | 0.35 | 0.32 |
| Noise spike-count bias (model − data, per sweep) | +24 | +14 |
| Model / data test spikes | | 1.18 |
| Noise subthreshold Vm RMSE (mV) | 2.7 | 2.5 |
| Square subthreshold Vm RMSE (mV) | 2.6 | 2.5 |
| Square steady-state error (mV) | 1.7 | 1.5 |
| Square absolute spike-count error | 2.5 | 2.2 |

These match the PR 13 numbers on the 6 original cells (Γ 0.16). The same caveats apply. Spike timing on
noise is still well below the Allen GLIF and all-active models, and the model fires somewhat too much on
noise (about 18% more spikes at the median usable cell).

## Heterogeneity and depth

Across the 41 usable cells the fitted parameters spread widely: Izhikevich V_th −29 ± 5 mV, reset c
−45 ± 5 mV, ln SK scale −1.5 ± 0.9, input rescaling k 0.75 ± 0.43 (mean ± SD). Only one usable cell is
silent on its test sweeps, so the library avoids the 3 barely-firing cells of the 8-cell set.

With depth (Spearman, n = 38 usable cells): recorded rheobase falls with depth (ρ = −0.34, p = 0.04) and
fitted V_th rises slightly (ρ = 0.28, p = 0.09). Fit quality does not depend on depth (Γ ρ = −0.17,
Vm RMSE ρ = 0.16, both p > 0.3). See `summary_vs_depth.png`. These are small-n exploratory numbers.

## Using the library

```python
import sys; sys.path.insert(0, "experiments/population")
import library
rows = library.index()                    # index.csv rows: depth_norm, qc/fit flags, metrics, key params, use
cell = library.load_cell("318808427")     # a StandardCell with this cell's parameters
cells, rows = library.load_population()   # all use == 1 cells (use_only=False for all 66)
```

Each `StandardCell` takes the raw injected current in pA (the per-cell input rescaling is folded into its
front-end weights) and runs like the Standard (`simulate`, or step by step as in `circuit/hybrid.py`).
For the depth-placed vs shuffled test, take `depth_norm` from `index.csv`.

The run logs are in `logs/` (`status.tsv` has one line per step per cell, with run time in seconds).
