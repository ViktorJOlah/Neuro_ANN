# Fano factor: measured behavioural-state drive of the LGN input

The model's trial-to-trial variability was far below in vivo (L2/3 PC Fano 0.34 vs 2.65). The fix had to be
biological, with no fitted parameter. The chosen candidate: slow shared modulation of the input by behavioural
state (running, pupil), with its size and timescale measured in LGN from the same Allen sessions.

## 1. In vivo, by state (44 Allen Visual Coding Neuropixels sessions)

`code/extract.py` + `code/fetch_all.sh` pull spikes, gratings, running and pupil from each session NWB.
`code/invivo_fano.py`: preferred grating at contrast 0.8, 2 s counts, RS units with mean count >= 1.

| population | all trials | still (< 1 cm/s) | running (> 5 cm/s) | slow drift removed |
|---|---|---|---|---|
| V1 L2/3 RS (476 units) | 2.69 | 2.26 | 2.24 | 2.21 |
| V1 L4 RS | 3.10 | 2.27 | 2.28 | 2.33 |
| V1 L5 RS | 3.04 | 2.06 | 2.00 | 2.19 |
| V1 L6a RS | 3.16 | 2.26 | 2.13 | 2.20 |
| LGd (1021 units) | 5.68 | 4.62 | 3.55 | 4.48 |

`code/state_gain.py`: shared gain per trial (split-half, removes Poisson noise), regressed on running and pupil;
timescale from spontaneous activity. Medians over sessions:

| population | shared gain SD per trial | fraction explained by running + pupil | timescale |
|---|---|---|---|
| LGd | 0.21 | 0.56 | 3.5 s |
| L2/3 | 0.29 | 0.19 | 0.5 s |
| L4 | 0.22 | 0.38 | 0.5 s |
| L5 | 0.19 | 0.52 | 0.5 s |
| L6 | 0.20 | 0.78 | 1.0 s |

`code/invivo_noisecorr.py`: L2/3 RS noise correlation (same condition, 2 s counts) 0.17 on all trials and 0.17 on
still trials.

State is a small part of the in vivo Fano: a still animal keeps L2/3 Fano at about 2.2.

## 2. Model with the measured LGN state drive

`code/make_state_drive.py`: for session 756029989 (median state-predicted gain SD among sessions, 40% running
trials), the LGd shared gain is regressed on running and pupil, and the fitted value of 10 reps of one grating, in
time order, is the LGN gain of trials 0-9 (mean 1, SD 0.16). Each LGN trial is the pool of the 10 Billeh LGN
trials thinned with p = g/10, so every train stays Poisson. The gain drives the whole network, so L4-L6 inherit it.
`run_state_trials.sh` runs the full GLIF network and the kernel-mode hybrid at the step 1 settings.
`code/model_fano.py` and `code/plot_fano_state.py` give `results/model_fano_state_756029989.json` and
`results/fig_fano_state.png`.

| | Fano | noise correlation | top-10% share | mean rate (Hz) |
|---|---|---|---|---|
| in vivo, all / still | 2.69 / 2.26 | 0.17 / 0.17 | 0.50 | |
| GLIF, Poisson LGN | 0.35 | 0.01 | 0.32 | 1.59 |
| GLIF, LGN state drive | 0.67 | 0.43 | 0.32 | 1.56 |
| ANN-PC, Poisson LGN | 0.34 | 0.01 | 0.29 | 1.51 |
| ANN-PC, LGN state drive | 0.46 | 0.20 | 0.29 | 1.48 |

- The measured state drive brings ANN-PC noise correlations to the in vivo level (0.20 vs 0.17); GLIF overshoots.
- Fano rises only from 0.34 to 0.46. With in vivo shared gain variance (0.085) and mean count (4.5), the shared
  part of in vivo Fano is about 0.4, so most of the in vivo Fano (about 1.8) is private to each cell. The model's
  private variability is sub-Poisson (0.34): its cells fire too regularly for a given input.
- The rate tail is unchanged (top-10% share 0.29), as expected for a gain shared by all cells.
