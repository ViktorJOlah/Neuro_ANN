# Isolated L2/3 circuit

Everything for the R21 isolated L2/3 circuit work lives in this folder. The plan is to build
single-cell ANNs of L2/3 pyramidal cells from the three Druckmann protocols, then model an
isolated L2/3 circuit. The circuit compares homogeneous cells with cells that have the measured
heterogeneity, and checks the resulting firing-rate distributions against in vivo data. (The
folder is named `L2_3` because `/` can't be used in a folder name.)

| folder | contents |
|---|---|
| `simulations/` | 3000 NEURON simulations of the Allen L2/3 PC model 488462783 (square, ramp, noise), with the code that generated them. A copy of `simulations/allen_L23_PC_protocols/`. |
| `training/` | The ANN training pipeline (CNN-LSTM models, curriculum and two-stage training). A copy of the repo-level `training/`. |
| `experiments/izhikevich_subthreshold/` | Izhikevich layer alone fitted to the spike-free simulations (current in, Vm out): code, metrics and plots for two variants. |
| `circuit/` | Step 5: the L2/3 circuit cut out of the Allen V1 point network, with the other layers, LGN and background replayed as fixed spike streams. See `circuit/README.md`. |
| `experiments/kernel_rerun/` | The rerun of steps 0-4 with the synaptic kernel layer, which is the default architecture from 2026-11-26. |
| `izhikevich/` | A batched, differentiable Izhikevich (2003) top layer, checked to be bit-identical to the published model. See `izhikevich/README.md`. |

The repo-level `simulations/` and `training/` folders are unchanged, so old paths and Colab
links still work. New work goes here.

Run everything from this folder:

```bash
cd "Isolated L2_3 circuit"
pip install -r training/requirements.txt
python -m pytest izhikevich/tests training/tests -q
python -m training.train --config training/configs/l23_two_stage.yaml
```
