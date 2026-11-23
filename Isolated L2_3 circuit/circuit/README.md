# Isolated L2/3 circuit (roadmap step 5)

Code that cuts layer 2/3 out of the Allen V1 point-neuron network (Billeh et al. 2020) and runs
it on its own, with every removed input replayed as a fixed spike stream. This is the "Extract
L2/3" and "Make the input streams" part of step 5, plus the 5a reference check, all with GLIF
cells. Swapping in the model PCs (5b) comes next.

## What is kept and what is replayed

| kept, unchanged | replayed as spike streams |
|---|---|
| all L2/3 cells (`pop_name` `e23*` / `i23*`: E, Pvalb, Sst, Htr3a) with their GLIF models, positions and tuning angles | every other V1 cell that synapses onto L2/3 (L1, L4, L5, L6), as the virtual population `v1ext` |
| every V1 → V1 edge with both ends in L2/3 (weights, delays, receptor ports as in the original files) | LGN and the 1 kHz background input, from the same spike files as the full run |

Edges from the replayed sources onto L2/3 are copied unchanged, so each replayed spike reaches
the same synapse with the same weight and delay as in the full network. The isolated circuit
therefore reproduces the full network's L2/3 spikes exactly. Any later difference comes from
the cells you swap in, not from the isolation.

## Files

| file | what it does |
|---|---|
| `extract_l23.py` | reads the SONATA network and writes `network/l23_*`, `v1ext_*`, `lgn_l23_*`, `bkg_l23_*` files, `id_maps.h5` (new ids → original V1 ids) and `extraction_summary.json`. `--radius` keeps only L2/3 cells within that distance of the column axis (the rest are replayed too). |
| `simulate.py full` | runs the original Allen config (BMTK PointNet / NEST) and saves all V1 spikes |
| `simulate.py streams` | turns full-network spikes into `v1ext_spikes.h5` (the replay stream) and `l23_reference_spikes.h5`. `--poisson-control` also writes a rate-matched Poisson stream with the timing removed. |
| `simulate.py isolated` | writes and runs the isolated config: L2/3 + replayed `v1ext` + the full run's LGN and bkg spike files |
| `compare_rates.py` | 5a check: rates per class, per-cell correlation, KS distance, fraction of matched spikes, and a plot |
| `plot_overview.py` | population rates over time, the L2/3 E rate distribution (with the top-10% spike share) and a raster |
| `synaptic_layer.py` | the fixed synaptic layer for model cells: presynaptic spikes → summed alpha PSC current (pA), exactly as Allen's `glif_psc` cells do it. Closed-form (`alpha_current`, `binned_current`) and stateful per-step (`AlphaPSC`) forms. |
| `demo_two_cells.py` | two-cell demo: a GLIF pair with a NEST synapse next to a pair of Standard ANN cells joined by the synaptic layer |
| `toy_network.py` | a 350-cell Billeh-like network with the same file layout, used by the tests |
| `sonata_io.py` | small h5py readers and writers for SONATA nodes, edges and spikes |

## Getting the Allen network

The GLIF network is the `simulations/v1_point` folder of the Allen V1 model Dropbox (linked from
the [Allen V1 model page](https://portal.brain-map.org/explore/models/mv1-all-layers)). Download
it as a zip by changing `dl=0` to `dl=1` in the folder's share link (8.9 GB). The zip holds
`network/`, `components/`, the LGN grating spikes (`lgn_stimulus/`, 315°, 80% contrast,
10 trials; the first 0.5 s is gray screen), and the background spike trials (`bkg_inputs/`).
`network_dynsyns/` isn't needed. Note that the bundled `output/spikes.h5` was made with LGN and
background files that aren't in the download (270°, 100% contrast), so it can't be the reference.
Run the full network instead.

## Running it

```bash
cd "Isolated L2_3 circuit"
pip install nest-simulator bmtk h5py scipy matplotlib
export PYTHONPATH=$PWD
V1=/path/to/v1_point; X=/path/to/l23_extract; R=/path/to/runs/gratings_t0

python -m circuit.extract_l23 --network-dir $V1/network --out $X
python -m circuit.simulate full --config $V1/config.json --out $R/full --threads 4
python -m circuit.simulate streams --full-spikes $R/full/spikes.h5 --extract-dir $X --out $R/streams --poisson-control --tstop 3000
python -m circuit.simulate isolated --config $V1/config.json --extract-dir $X --v1ext-spikes $R/streams/v1ext_spikes.h5 --out $R/isolated --threads 1
python -m circuit.compare_rates --extract-dir $X --reference-spikes $R/streams/l23_reference_spikes.h5 --isolated-spikes $R/isolated/spikes.h5 --out $R/check --t1 3000
python -m pytest circuit/tests -q
```

On a 4-core, 16 GB machine the full 230,924-cell network takes about 4 minutes per 3 s of
simulation (13 GB peak), and the isolated circuit about 3 minutes on one thread (5 GB). Run the
isolated circuit with `--threads 1`: with 4 NEST threads, wiring the 100k replayed sources runs
out of memory. `simulate.py` also replaces BMTK's pandas node-id lookup with an array lookup
(`patch_bmtk_gid_lookup`), which gives the same ids with far less memory.

## Results on the Allen network (gratings 315°, LGN and background trial 0)

Numbers and figures are in `results/` (`extraction_summary.json`, and per condition
`replay/`, `poisson_control/`, `overview.png`, `runs.json`). In short: 65,949 L2/3 cells (56,057 E, 2,927 Pvalb,
2,120 Sst, 4,845 Htr3a) and 15.0M recurrent edges are kept. 101,792 other V1 cells (13.3M edges,
mostly from L4 E), 756k LGN edges and 66k background edges are replayed. Recurrent L2/3 synapses
are 52% of all synapses onto L2/3. Replaying the recorded streams reproduces every one of the
351,504 L2/3 spikes of the full network.

| L2/3 class | full network (Hz) | isolated, replay (Hz) | isolated, Poisson control (Hz) |
|---|---|---|---|
| E | 1.38 | 1.38 | 1.37 |
| Pvalb | 5.70 | 5.70 | 3.93 |
| Sst | 4.78 | 4.78 | 3.72 |
| Htr3a | 2.73 | 2.73 | 2.69 |

Rates are over the whole 3 s (0.5 s gray, then 2.5 s grating). In the full network, during the
grating, 30% of L2/3 E cells are silent and the top 10% fire 33% of the E spikes.

The Poisson control keeps each replayed source's mean rate but removes its timing and
correlations. E rates barely change, but Pvalb and Sst drop by 31% and 22%, so the
interneurons depend on synchronous L4 input. That makes the replay, not Poisson drive, the right
input for steps 5b and 6.

## Synaptic input to model cells (step 5b)

The Standard cell model was trained on injected current only. In the Allen point network,
synapses are current based: a spike arriving after the edge delay adds
I(t) = w · (t/τ) · e^(1 − t/τ) to the soma, peaking at the weight w (pA) at t = τ, with
τ = 5.5 / 8.5 / 2.8 / 5.8 ms for ports 1–4 (E→E, I→E, E→I, I→I). A cell sums these linearly.
`synaptic_layer.py` implements exactly that as a fixed, parameter-free layer in front of the
ANN, so the ANN's trained weights are never touched: synaptic input becomes an injected current,
binned to the ANN's 1 ms steps.

Checks:
- It matches NEST's own `I_syn` of a `glif_psc` cell to 1e-7 pA (`tests/test_synaptic_layer.py`).
  NEST rounds the delay to its 0.25 ms grid and starts the current one step later, so a 1.6 ms edge
  acts as 1.75 ms. `nest_effective_delay` reproduces this.
- The stateful form agrees with the closed form to 1e-7 pA.

`demo_two_cells.py` (`results/two_cell_demo/`) runs a GLIF pair and a Standard ANN pair under the
same drive. Both pairs use one E→E synapse (w = 60 pA, about 3.5 of Allen's strongest L2/3 E→E
synapse, so the effect shows on a single connection) and both use the Allen GLIF model of the same
L2/3 PC (node type 487661754) that the Standard model was fitted to. In both pairs B fires more
(GLIF 1.2 → 3.0 Hz, ANN 3.2 → 3.8 Hz), and B's spikes cluster after A's: 5–20 ms for GLIF and
10–30 ms for the ANN. In the GLIF pair, driving an unconnected B with the layer's current
reproduces the synapse-driven B's spikes within 2 ms (92% of spikes). The ANN pair is
feed-forward, so A is simulated first. A recurrent circuit steps `AlphaPSC` together with the
cells.

```bash
python -m circuit.demo_two_cells --glif-dir $V1/components/cell_models/nest_models \
    --standard-dir Standard --out results/two_cell_demo
```
