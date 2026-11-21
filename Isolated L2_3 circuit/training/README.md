# Staged training pipeline

A single, configurable replacement for the hand-edited fitting scripts
(`single compartmental model/fit_CNN_LSTM_latest.py`,
`multicompartmental models/fit_multicompartmental.py`,
`multicompartmental models/L2_3 PC/final_L2_3.ipynb`). The original scripts are
left untouched. The CNN-LSTM architectures are the same layer for layer, so old
`.h5` weights load as `model.init_weights`.

```bash
pip install -r training/requirements.txt
python -m pytest training/tests -q                                  # ~1 min on CPU
python -m training.train --config training/configs/l23_two_stage.yaml
python -m training.train --config ... --stages subthreshold         # run one stage
python -m training.train --config ... --resume                      # continue after a crash or disconnect
python -m training.train --config ... --set training.batch_size=128 # override any value
python -m training.train --config ... --eval-only --split test      # closed-loop scores
```

On Colab use `training/notebooks/train_colab.ipynb`, which clones the repo, writes
checkpoints to Drive and runs each stage in its own cell.

## What changed compared with the original scripts, and why

| Problem in the original code | Effect | Fix here |
|---|---|---|
| Vm normalized with min/max of whichever file is loaded | stage 2 sees differently scaled inputs than stage 1 | one `Scaler` fit on the training traces of **all** pools, saved to `scaler.json`, used by every stage |
| Stage 2 restarts a fresh optimizer at full LR on the pretrained weights | large first updates can undo stage 1 | per-stage LR with linear warmup; stage 2 starts from stage 1's **best** checkpoint |
| `data["ina"][i] = ...` chained assignment | no-op under pandas copy-on-write (pandas ≥ 2.2 warns, 3.0 ignores): compression/clipping silently skipped | vectorized numpy preprocessing (`data.preprocess`) |
| random split of overlapping windows; checkpoint on training loss; float16 validation | val loss ≈ train loss, no overfitting signal; loss floor ≈ float16 resolution | split **by trace**, float32, best checkpoint on validation loss |
| `i % 10000` skip with the wrong index (and none in the L2/3 path) | training windows straddle two unrelated traces | windows are built per trace and never cross a boundary |
| header stripping with `pop` while iterating, output opened in append mode | lines skipped; re-running duplicates the dataset | headers detected by field count while reading; source files are never written |
| `keras.layers.convolutional`, `tqdm._tqdm_notebook`, `Nadam(lr=)`, `period=` | fail on current Colab (Keras 3) | Keras 3 API only |
| no protection against divergence | NaN or exploding loss wastes a run | non-finite gradients are zeroed and counted; if an epoch's loss is non-finite or > `blowup_factor` × best, the best weights are restored and the LR is backed off (up to `max_retries`) |

## Stages and the smooth curriculum

Each stage trains on a mixture of the `subthreshold` and `spiking` trace pools:

```yaml
stages:
  - {name: subthreshold, epochs: 150, lr: 1.0e-3, warmup_steps: 1000, spiking_fraction: 0.0}
  - {name: spiking,      epochs: 300, lr: 1.0e-4, warmup_steps: 2000, spiking_fraction: 1.0}
```

Any of `spiking_fraction` and `spike_loss_weight` can be ramped instead of
switched (`configs/l23_curriculum.yaml`):

```yaml
  - name: curriculum
    spiking_fraction: {start: 0.1, end: 0.7, ramp_epochs: 60, shape: cosine}
    spike_loss_weight: {start: 1.0, end: 5.0, ramp_epochs: 60}
```

`spike_loss_weight` up-weights samples whose target is within
`training.spike_window` rows of a spike, so the ~1 % of spike samples are not
drowned out by the subthreshold samples. The best checkpoint of a stage is chosen on the
validation loss at the stage's *final* mixture and weighting, so the best checkpoint is
comparable across epochs even while the ramp moves.

Pools come either from separate files (`data.files.subthreshold`,
`data.files.spiking`) or from one mixed file list (`data.files.all`) split by
spike count (`data.min_spikes_spiking`).

## Closed-loop evaluation

Training is one step ahead, but simulation feeds predictions back. Every
`eval.rollout_every` epochs, and on the test traces at the end of each stage,
`evaluate.py` runs the real closed loop on held-out traces and reports
subthreshold variance explained, RMSE (mV), spike precision, recall and F1, the
spike-count ratio, and the fraction of traces that diverged. These are the
`cl_*` columns in `history.csv` and the values in `eval_test.json`.

## Opt-in closed-loop robustness (changes model behaviour, off by default)

* `training.state_noise_std`: Gaussian noise on the fed-back state channels of
  the input window, so the model learns to correct small errors in its own
  history.
* `training.rollout_steps` / `rollout_loss_weight`: adds a k-step closed-loop
  loss (backpropagation through the model's own predictions). This is the direct remedy
  for exposure bias, and should reduce the need for post-hoc linear
  correction of predictions.

Turn these on for one cell and compare `eval_test.json` against the default run.

## Run directory

```
<output_dir>/<run_name>/
  config.yaml  scaler.json  data_split.json  state.json
  <stage>/best.weights.h5  last.weights.h5  best_model.keras  best_model.h5
          history.csv  eval_test.json
```

Resuming restores the weights, epoch, best score and LR scale. The optimizer
moments restart, and the stage warmup covers that.

## Synthetic data

`python -m training.synthetic out.txt --traces 20 --rate-scale 0.12` writes
Izhikevich-cell traces in the `calibrate.hoc` format (`--currents` gives the
single-compartment `vm ik ina inp0 inp1` layout). The tests use it, and it is handy
for checking a config before pointing it at real data.
