# Five more Allen L2/3 PCs: square, ramp and noise simulations, clean and with patch noise

These datasets test how far the single-cell approach carries beyond the first cell
(specimen 487661754, `../data/`). Each of 5 further mouse VISp L2/3 pyramidal cells gets 100
simulations for each of the 3 Druckmann/Allen protocols, plus a copy with realistic
patch-clamp recording noise. Protocols, sampling and conventions are identical to the first
cell (`../code/generate_dataset.py` is imported, not copied), so the files can be loaded
with the same code.

## Cells

All are Allen Cell Types specimens from VISp layer 2/3 with spiny (pyramidal) dendrites.
The models are the **all-active** fits of Nandi et al. 2022 (Allen, *Cell Reports*), best
hall-of-fame model (#0). These cells have no perisomatic model like 488462783. All-active
models also carry active dendritic conductances, which makes them a harder test case than
the first cell.

| specimen | line | model rheobase (pA) | recorded rheobase (pA) | square F-I max (spikes/1 s) | ramp len (ms) |
|---|---|---|---|---|---|
| 318808427 | Nr5a1-Cre | 112 | 130 | 28 | 7500 |
| 486262299 | Cux2-CreERT2 | 484 | 490 | 7 | 26900 |
| 502359001 | Cux2-CreERT2 | 142 | 150 | 26 | 9900 |
| 513531471 | Cux2-CreERT2 | 142 | 150 | 15 | 10500 |
| 521409057 | Cux2-CreERT2 | 205 | 210 | 17 | 12200 |

Recorded rheobase is the Allen "threshold current" of the cell.

## Data (`data/<specimen>/`)

| file | contents |
|---|---|
| `square.npy` | (100, 2000, 2): 1 s steps, -110 pA ... rheobase + 160 pA |
| `ramp.npy` | (100, n_ms, 2): 25 pA/s ramps ended between the 1st and 27th ramp spike |
| `noise.npy` | (100, 4500, 2): pink noise (1-100 Hz, CV 0.2) on a 3 s square at 0.75-1.5 x rheobase |
| `<protocol>_noisy.npy` | same arrays with recording noise added to Vm (the current is left clean) |
| `<protocol>_params.csv` | amplitude, timing, seed and spike count of each simulation |
| `<protocol>_noise_params.csv` | the recording session, seed and realised RMS of the noise added to each trace |
| `calibration.json` | model rheobase, rest, the ramp spike amplitudes used for the ramp bounds |
| `metadata.json` | sources, build, integration settings, summary |

* `[..., 0]` = injected current (pA), `[..., 1]` = somatic Vm (mV), one sample per 1 ms.
* Vm uses the Allen recording convention: v_sim minus the junction potential (-14 mV), the same as the first cell.
* NEURON 9.0.2, fixed step dt = 0.01 ms (as for the first cell), 34 °C, steady-state initialisation at zero current.
* The ramp length differs per cell because the end amplitude depends on the cell's own rheobase.

## Model sources and build (`models/`)

The Allen API and brain-map.org can't be reached from the build machine, so the same files
were taken from public mirrors:

* fit parameters: `github.com/AllenInstitute/All-active-Manuscript` `assets/mouse_models/<id>/hof_models/hof_param_<id>_0.json`
* morphologies: `github.com/YaxuanLi-cn/MorphRep` `benchmark_dataset/allen_cell_type_processed/` (the Allen SWCs; the copy of the first cell is byte-identical to the Allen download)
* mechanisms (`models/modfiles/`): from the same all-active manuscript repo

`code/allactive_cell.py` reproduces AllenSDK `AllActiveUtils` with the **stub axon**
(2 × 30 µm, 1 µm diameter, at soma(0.5)), which is what the all-active fitting workflow
(ateamopt `stub_axon`) used. The "truncated" axon option (SWC diameters) was tried first and
gave broken models (depolarisation block on the ramp, self-sustained firing after the step).
With the stub axon, all 5 model rheobases are within about 20 pA of the recorded ones (see table).

## Recording noise model (`noise_model/`)

Measured by `code/measure_noise.py` from real Allen whole-cell current-clamp recordings
(DANDI 000020, Allen patch-seq, 8 Slc17a7 excitatory-cell sessions listed in
`sessions_downloaded.tsv`; MultiClamp 700B, 10 kHz Bessel, 50 kHz sampling):

* Baseline Vm at zero command, at least 300 ms after any command change, spike-free, point-sampled at 1 ms like the simulations, so aliased high-frequency noise is included.
* One median Welch PSD per session (0-500 Hz, 1.95 Hz bins). It holds everything a real baseline has: amplifier and thermal noise, 60 Hz hum and harmonics, and spontaneous membrane fluctuations of the cell in the slice.
* Sessions with resting Vm above -65 mV (2 of 8, unhealthy cells) are excluded, which leaves 6.
* Measured baseline RMS is 0.04-0.15 mV and 60 Hz hum is about 0.01 mV. Allen recordings are very clean.

`code/add_recording_noise.py` draws one session per trace and synthesises Gaussian noise
with that PSD and random phases. Each cell uses its own seed (20261122 + specimen id).
Below 1.95 Hz the PSD is held at its 1.95 Hz value, so slow drift is not modelled and the
synthetic RMS is a little below the measured one.

## Plots (`plots/`)

* `example_traces.png`: an example of each protocol for each cell, clean vs noisy.
* `fi_curves.png`: F-I from the square sims, with the first cell for comparison.
* `noise_model.png`: the measured baseline PSDs and a synthesised noise trace.

## Reproduce

```bash
cd models && nrnivmodl modfiles && cd ../code
python generate_dataset.py --cell ../models/521409057 --mech ../models --out ../data/521409057   # 1.5-4 h per cell on 4 cores
python add_recording_noise.py --data ../data/521409057 --psd ../noise_model/noise_psd.npz
python plot_cells.py --data ../data --first ../../data --psd ../noise_model/noise_psd.npz --out ../plots
```

`N_SIM=10 python generate_dataset.py ...` gives a quick smoke run.

## Caveats

* These are models of the 5 cells, not the cells themselves. The all-active fits match the recorded features well (Nandi et al.), but the exact traces are model output.
* The noise is stationary and Gaussian. It has no slow drift, no access-resistance changes and no bridge-balance error. Real recordings can also have spike-shape distortion from series resistance, which is not modelled.
* 486262299 has a high rheobase (about 490 pA) and fires at most 7 spikes per 1 s step, so its ramps are much longer (26.9 s) and only 22 of its 100 squares spike.
* 318808427's F-I curve drops slightly at the top of the range (28 to 26 spikes), the start of depolarisation block.
* The noise is small next to the signal (RMS about 0.06-0.09 mV), as in the Allen recordings, so it is barely visible on the example plots. It matters mostly for subthreshold and threshold fitting.
* Wall time on 4 shared cores: 1.5-4.2 h per cell.
