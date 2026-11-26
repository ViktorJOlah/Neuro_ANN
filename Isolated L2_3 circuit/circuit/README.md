# Isolated L2/3 circuit (roadmap step 5)

Code that cuts layer 2/3 out of the Allen V1 point-neuron network (Billeh et al. 2020) and runs
it on its own, with every removed input replayed as a fixed spike stream. This is the "Extract
L2/3" and "Make the input streams" part of step 5, plus the 5a reference check, all with GLIF
cells. The hybrid circuit with ANN PCs (5b) is described at the end.

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
| `pc_layer.py` | all PCs as one batch of Standard cells, advanced one 1 ms step at a time (same output as `StandardCell.simulate`) |
| `hybrid.py` | the hybrid circuit: GLIF interneurons in NEST, ANN PCs batched every ms (`prepare`, `run`, `tune`, `report`) |
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

## Hybrid circuit: ANN PCs with GLIF interneurons (`hybrid.py`)

Every L2/3 interneuron stays the Allen GLIF cell and runs in NEST at 0.25 ms. All 56,057 L2/3 PCs
are the Standard ANN cell (node type 487661754, which is also every PC of the GLIF network, so both
networks are homogeneous in their PCs). At each whole millisecond all PCs read their synaptic
current for the past millisecond and take one batched 1 ms step. The two halves trade spikes
every millisecond:

1. PC input for (k, k+1] = the replayed input (L1/L4/L5/L6, LGN, background; it does not depend
   on the circuit, so `prepare` computes it once) plus the PC and interneuron spikes arriving in
   that millisecond, through the fixed synaptic layer (Allen alpha PSCs at 0.25 ms, averaged over
   the millisecond).
2. One Standard step for all PCs (`pc_layer.BatchedStandard`). PC spikes are stamped at k+1.
3. PC spikes become spike-generator times in NEST, which then advances the interneurons from k to
   k+1. Their spikes go to the PCs' synaptic layer with each edge's NEST delay.

Every edge keeps its Allen weight, delay and receptor port. A spike always reaches its target at
least 0.25 ms (PC → PC) or 1.25 ms (interneuron → PC) after it is fired, so it lands in a later
millisecond than the one that produced it.

**Gain.** `--gain a` multiplies every synapse onto a PC (excitatory and inhibitory, replayed and
recurrent), so each PC's E/I ratio is unchanged. `--pc-to-in-gain` sets the factor on PC →
interneuron synapses separately: a number, or one per class (`Pvalb=1.0,Sst=1.0,Htr3a=1.0`). Left
out, it equals `--gain` (the rule of the first version, PR 9). Use `--pc-to-in-gain 1.0`: the
interneurons are unchanged GLIF cells, so the synapses onto them keep Allen's weights (see
"Interneuron rates" below). `tune` searches a until the mean PC rate matches the GLIF network,
with `--pc-to-in-gain` held fixed.

**Synapses (default from 2026-11-26: kernel layer).** `run`/`tune --synapses kernel` (the default) turns each port's
Allen alpha PSC into a conductance (normalised at −55 mV; E_exc 0, E_inh −80 mV). The conductance enters the Izhikevich
layer through the per-cell membrane correction (`--corr`, or each model's `corr` in a population file), not through the
CNN-LSTM, whose input is then zero. `--synapses current` is the older path (PR 9/14). Tuned kernel settings: `--gain 1.18
--pc-to-in-gain "Pvalb=1.06,Sst=1.04,Htr3a=1.04"`. See `experiments/kernel_rerun/README.md`.

**Checks.**
- Loop check: replaying the GLIF network's own PC spikes through the loop (`run --replay-pc`)
  reproduces all 120,075 interneuron spikes of the GLIF network exactly. The PC input current
  matches NEST's `I_syn` of the same PCs to under 0.01 pA per millisecond (refractory bins masked).
- `tests/test_pc_layer.py`: the batched per-step layer gives exactly the spikes of `StandardCell.simulate`
  (needs `STANDARD_DIR`). `tests/test_hybrid_bank.py`: the PCs' synaptic layer against the closed form.

**Results** (gratings 315°, trial 0, 3 s; `results/hybrid_ann_pcs/`):

| | GLIF network | ANN PCs, gain 1 (0–1.5 s) | ANN PCs, gain 0.82 |
|---|---|---|---|
| L2/3 E (Hz) | 1.38 | 2.17 (GLIF 1.13) | 1.28 |
| Pvalb (Hz) | 5.70 | | 3.37 |
| Sst (Hz) | 4.78 | | 2.98 |
| Htr3a (Hz) | 2.73 | | 2.23 |
| E silent during grating | 30% | | 29% |
| E spikes fired by the top 10% | 33% | | 32% |

At gain 1 the ANN PCs are hyperactive (about 2× the GLIF rate, and the interneurons follow at
2.3×). At gain 0.82 the PC rate is within 7% of the GLIF network over 3 s (1.5 s tuning run:
1.125 vs 1.126 Hz) and the rate distribution matches: the same silent fraction, the same top-10%
share, and each PC's rate correlates with its GLIF-network rate at r = 0.96. Two differences remain:
the ANN network has no PCs above about 10 Hz (the GLIF network reaches 13 Hz), and the
interneurons fire 18–41% less. The second one is fixed below.

**Interneuron rates** (`results/interneuron_rate_fix/`). The interneurons were low because the
gain of 0.82 was also applied to every PC → interneuron synapse, so the unchanged GLIF
interneurons got 18% less excitation per PC spike than in the Allen network. Open-loop tests
(`run --replay-pc` with fixed PC spikes, no feedback; `diagnose_interneurons.py`) separate the causes:

| PC spikes fed to the interneurons | PC → IN gain | Pvalb | Sst | Htr3a |
|---|---|---|---|---|
| GLIF network's own | 0.82 | −36% | −32% | −12% |
| hybrid (PR 9) | 0.82 | −41% | −38% | −18% |
| hybrid (PR 9) | 1.0 | −8% | −10% | −7% |
| hybrid, each spike moved 0–0.75 ms earlier | 1.0 | −8% | −9% | −7% |

So the gain explains most of the gap, the rest follows the 7% lower PC rate, and the 1 ms
spike grid of the ANN PCs does not matter. With PC → interneuron synapses at Allen weights,
the gain onto PCs has to rise a little (more inhibition comes back). The 1.5 s `tune` gave
0.827, but over 3 s that leaves PCs 11% low (the second half of the grating is more
inhibited), so the gain was set on 3 s runs: 0.848.

| 0–3 s, Hz | GLIF network | PR 9 (gain 0.82 onto and from PCs) | gain 0.827 onto PCs, PC → IN 1.0 | **gain 0.848 onto PCs, PC → IN 1.0** |
|---|---|---|---|---|
| L2/3 E | 1.38 | 1.28 (−7%) | 1.23 (−11%) | **1.36 (−1%)** |
| Pvalb | 5.70 | 3.37 (−41%) | 4.82 (−15%) | **5.66 (−1%)** |
| Sst | 4.78 | 2.98 (−38%) | 4.03 (−16%) | **4.69 (−2%)** |
| Htr3a | 2.73 | 2.23 (−18%) | 2.50 (−8%) | **2.66 (−3%)** |

With gain 0.848 every class is within 3% of the GLIF network, so no per-class factor is needed.
During the grating 28% of PCs are silent (GLIF 30%), the top 10% fire 30% of PC spikes (GLIF 33%),
and each PC's rate correlates with its GLIF-network rate at r = 0.97 (Spearman 0.98). The short
PC upper tail (no PCs above about 10 Hz) is unchanged, and the largest synchronous interneuron
peaks are still somewhat smaller than in the GLIF network.

Runtime on 4 CPUs: about 0.85 s per simulated millisecond (43 min per 3 s), almost all of it the
batched CNN-LSTM front end. NEST and spike routing add 0.03 s per ms.

```bash
H=/path/to/hybrid; STD=/path/to/Standard
python -m circuit.hybrid prepare --config $V1/config.json --extract-dir $X --v1ext-spikes $R/streams/v1ext_spikes.h5 --out $H/prep
python -m circuit.hybrid run --prep $H/prep --standard-dir $STD --gain 1.0 --tstop 300 --out $H/check \
    --replay-pc $R/streams/l23_reference_spikes.h5 --probe 20      # loop check, no ANN
python -m circuit.hybrid tune --prep $H/prep --standard-dir $STD --reference $R/streams/l23_reference_spikes.h5 \
    --pc-to-in-gain 1.0 --gains 0.86 --tstop 3000 --tol 0.03 --out $H/tune
python -m circuit.hybrid run --prep $H/prep --standard-dir $STD --gain 0.848 --pc-to-in-gain 1.0 --tstop 3000 --out $H/final
python -m circuit.hybrid report --prep $H/prep --extract-dir $X --reference $R/streams/l23_reference_spikes.h5 \
    --hybrid-spikes $H/final/spikes.h5 --tune-json $H/tune/tune.json --out results/interneuron_rate_fix \
    --previous-spikes $H/pr9_gain0.82/spikes.h5
```
