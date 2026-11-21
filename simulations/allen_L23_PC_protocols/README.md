# Allen L2/3 PC: square, ramp and noise simulations

1000 NEURON simulations per protocol (3000 total) of the Allen Cell Types Database
perisomatic biophysical model **488462783** (specimen **487661754**,
Cux2-CreERT2;Ai14 L2/3 pyramidal cell, VISp). Model files are the Allen download already in
`multicompartmental models/L2_3 PC/NEURON/orig_file/`.

## Data (`data/`)

| file | shape | contents |
|---|---|---|
| `square.npy` | (1000, 2000, 2) | 1 s current steps |
| `ramp.npy` | (1000, 7800, 2) | 25 pA/s ramps, terminated at varying amplitudes |
| `noise.npy` | (1000, 4500, 2) | 3 s pink-noise epochs on a square offset |

* `[..., 0]` = injected current (pA), `[..., 1]` = somatic Vm (mV). One sample per 1 ms.
* Vm uses the Allen recording convention (v_sim + 14 mV liquid junction potential), the same
  as AllenSDK output, so it can be compared to Allen recordings directly.
* `<protocol>_params.csv`: amplitude, timing, noise seed and spike count of every simulation.
* `metadata.json`: model, integration settings, the model's measured rheobase and all choices.

Load with `x = np.load("square.npy"); I, V = x[..., 0], x[..., 1]`.

## Protocols and amplitude bounds

Bounds come from the Allen Cell Types *Technical White Paper: Electrophysiology*
(Oct 2017, v.5, Fig. 3 and Appendix "Stimulus Details"). Allen defined amplitudes relative to
each cell's rheobase, so the model's own rheobase (1 s step, bisection) is used.

* **Long square**: 1 s step after 500 ms baseline, 500 ms after. 1000 amplitudes evenly spaced
  from -110 pA to rheobase + 160 pA (white paper: "-110 pA ... to rheobase + 160 pA").
* **Ramp**: 25 pA/s from 0 pA after 500 ms baseline (white paper: "slope of 25 pA/s ...
  terminated after the neuron fires a short series of action potentials"). The termination
  amplitude is varied, evenly spaced from the model's 1st ramp spike to its 27th ramp spike
  (27 is the most spikes Allen recorded on this cell's ramp sweeps, sweep 5; sweep 6 had 14).
  After termination current is 0 for the rest of the fixed-length trace.
* **Noise**: pink noise with CV = 0.2 (SD = 0.2 x offset) riding on a 3 s square (white paper:
  "3 x 3 s noise epochs riding on top of square pulses at 0.75, 1, and 1.5 times rheobase").
  Each simulation is one epoch, offset evenly spaced from 0.75 to 1.5 x rheobase, with its own
  random seed (Allen used 2 fixed seeds). 500 ms before, 1 s after.
  The noise band (1/f power from 1 to 100 Hz) is an assumption: the white paper doesn't state it.

## Simulation details

* `code/allen_cell.py` rebuilds the cell exactly as `allensdk.model.biophysical.utils.Utils`
  (perisomatic) does: SWC import, axon replaced by a 60 um stub, passive and channel parameters
  from `fit_parameters.json`, 34 C. It was checked against AllenSDK's own builder (same rest,
  same responses).
* Current is injected at soma(0.5) through an IClamp; dt = 0.01 ms (Allen's runner uses 0.005 ms;
  0.01 ms shifts spike times by <0.1 ms over 1 s). Current is defined at 1 ms points and played
  with linear interpolation, so the stored current is the exact input.
* Each run starts from the zero-current steady state (rest = -82.0 mV in recording convention).
* Vm is sampled (not averaged) every 1 ms, so spike peaks are clipped by the sampling.

## Reproduce

```bash
pip install neuron numpy
cd "multicompartmental models/L2_3 PC/NEURON/orig_file" && nrnivmodl modfiles && cd -
python simulations/allen_L23_PC_protocols/code/generate_dataset.py \
  --model "multicompartmental models/L2_3 PC/NEURON/orig_file" \
  --out simulations/allen_L23_PC_protocols/data --progress progress.txt
```

About 80 minutes on 4 cores (works on Colab CPU too). `N_SIM=4` env var gives a quick test run.

## Caveats

* The model's rheobase (83.8 pA) is above the recorded cell's (~70 pA), and its hyperpolarizing
  response is larger than the recording (-110 pA gives about -132 mV vs about -101 mV recorded).
  These are properties of Allen's perisomatic fit, not of this code.
