# Aim 2 controls in silico (feasibility)

These runs exercise the full Aim 2 analysis pipeline in the hybrid L2/3 circuit before the new recordings
exist. They ask whether the pipeline works and what kind of answer it gives. With only 8 fitted cells, the
numbers below are **not** the R21's result.

The four controls are:

- **(a)** homogeneous vs heterogeneous PCs at a matched mean rate, compared with Neuropixels V1 L2/3 RS units
- **(b)** a rank test: are the top firers the most intrinsically excitable cells, or the best-driven ones?
- **(c)** depth-placed vs shuffled heterogeneity
- **(d)** homogeneous vs diverse interneurons

## Where the heterogeneity comes from

The heterogeneity comes from the **8 existing Standard fits**. It is not a parametric spread fitted to the
Allen population. Each fit is one Allen L2/3 PC specimen, and each model drives 1/8 of the 56,057 PCs
(`models.json`):

| specimen | fit | rheobase of the model, 1 s steps | normalized depth |
|---|---|---|---|
| 487661754 | the Standard itself (simulated perisomatic) | 90 pA | 0.17 |
| 513531471 | real recording (PR 11) | 150 pA | 0.11 |
| 521409057 | real recording (PR 11) | 220 pA | 0.14 |
| 486262299 | simulated all-active (PR 10) | 500 pA | 0.21 |
| 502359001 | simulated all-active (PR 10) | 160 pA | 0.26 |
| 318808427 | real recording (PR 11) | 120 pA | 0.34 |
| 382982932 | real recording (PR 11) | 120 pA | 0.36 |
| 502269786 | real recording (PR 11) | 200 pA | 0.37 |

485468180 is left out because its fit fails. Each fit has its rheobase rescaling folded into its weights, so
every model reads the same raw pA input. `results/fig0_models.png` shows the f-I curves and the rates under
fluctuating input (`characterize_models.py`).

## Network settings

- **Interneuron fix (PR 14):** PC→IN synapses stay at the Allen weights (`--pc-to-in-gain 1.0`) in every
  condition.
- **Gain:** `--gain` scales only the synapses onto PCs. It is tuned per condition on 3 s runs, so that the
  mean PC rate over 0–3 s matches the homogeneous network at gain 0.848 (1.36 Hz).
- **Tuned gains:** heterogeneous 1.284 (1.34 Hz). Rheobase-matched 0.878 (1.39 Hz). Homogeneous PCs with
  homogeneous interneurons 0.848 (1.32 Hz).
- **Excitability confound:** the Standard is the most excitable of the 8 models, so a heterogeneous network
  needs more input gain to reach the same mean.
- **Rate window:** rates are measured over the grating window (0.5–3 s), as in the earlier in vivo comparison.
  In that window, the heterogeneous networks run about 11% below the homogeneous one (1.40 vs 1.58 Hz).

## Results (`results/summary.json`)

| condition | mean (Hz) | silent | top-10% share | p99 (Hz) | max (Hz) |
|---|---|---|---|---|---|
| in vivo, Neuropixels L2/3 RS (n = 595) | 1.44 | 0.34 | **0.50** | **12.1** | 23.3 |
| GLIF network | 1.58 | 0.30 | 0.33 | 6.4 | 12.8 |
| homogeneous ANN PCs | 1.58 | 0.28 | 0.30 | 6.0 | 10.4 |
| heterogeneous, shuffled | 1.40 | 0.59 | **0.45** (0.41–0.50) | **7.6** (6.8–8.0) | 17.2 |
| heterogeneous, depth-placed | 1.40 | 0.59 | 0.45 | 7.6 | 13.2 |
| heterogeneous, rheobase-matched | 1.61 | 0.35 | 0.35 (0.33–0.38) | 7.2 (6.4–8.4) | 16.8 |
| homogeneous PCs, homogeneous IN | 1.53 | 0.28 | 0.31 | 5.6 | 10.0 |
| heterogeneous PCs, homogeneous IN | 1.37 | 0.60 | 0.45 | 7.2 | 17.2 |

The ranges in parentheses are 95% intervals of the statistic in random subsamples of 595 cells.

**(a) Heterogeneity moves the distribution toward in vivo, but in the wrong way.** The top-10% share rises
from 0.30 to 0.45, close to in vivo (0.50). The 99th percentile only rises from 6.0 to 7.6 Hz, against
12.1 Hz in vivo. Most of the gain in skew comes from extra **silent** cells: 59% are silent vs 34% in vivo,
because 5 of the 8 models fire under 0.5 Hz in the network. A long upper tail does not appear. Real
heterogeneity, with many more cells, would need both a spread of low-rheobase cells and a high-gain minority.
With 8 models this cannot be told apart (`fig1_distributions.png`).

**(b) Rank test: in this circuit, intrinsic identity wins.** The model a PC is assigned explains 67% of the
variance in log rate. Its input drive (its rate in the homogeneous network) explains 13%, and both together
explain 80%. The most excitable model (lowest rheobase) supplies 62% of the top-10% cells. Rate vs −rheobase
gives Spearman 0.76. The top-10% cells still sit at the 84th percentile of input drive, so both factors
matter (`fig2_rank_test.png`). With 8 discrete models, this test can only resolve between-model effects. The
R21 library of 60–80 cells is what makes it a real test.

**Rheobase-matched control: most of the effect is rheobase.** Each PC's input is scaled by its model's
rheobase / 90 pA, which equalizes threshold while keeping each model's gain, adaptation and spike timing.
The top-10% share then falls from 0.45 to 0.35, against 0.30 homogeneous. The silent fraction returns to
35%, close to in vivo. The rank test flips: input drive explains 73% of log rate and model identity 12%.
Rate vs −rheobase gives Spearman −0.04, and the top-10% cells sit at the 94th percentile of input drive. So
in these 8 fits, threshold differences produce most of the skew, and the remaining f-I differences add a
little (0.30 → 0.35, with p99 6.0 → 7.2 Hz). For the R21, the rheobase spread of the real library is the
quantity to measure and report.

**(c) Depth placement is null by construction.** In the Allen V1 model, the input from outside L2/3 does not
depend on depth (external current vs depth, Spearman 0.0001; homogeneous rate vs depth, 0.0005). Depth-placed
and shuffled populations therefore give the same distribution (`fig3_depth.png`). A meaningful depth control
needs depth-dependent inputs, such as L4 vs L1 drive by depth. The Allen model does not provide them.

**(d) Interneuron diversity does not shape the PC tail here, and this control is confounded.** Giving every
interneuron its class's most common GLIF model changes the PC statistics by less than the subsample noise
(top-10% share 0.45 vs 0.45; p99 7.6 vs 7.2 Hz). However, the most common Htr3a model is almost silent with
the network's input (0.01–0.04 Hz vs 3.0 Hz with diverse models), and Pvalb and Sst rise 13–25%. A clean
version would rescale each class's homogeneous model to keep its class rate. As run, this is a pipeline
check only (`fig4_interneurons.png`).

**Interneuron rates with heterogeneous PCs:** Pvalb 4.4, Sst 4.2, Htr3a 2.7 Hz. The homogeneous network gives
6.7, 5.6 and 3.0 Hz. With PC→IN fixed, interneurons follow the lower PC rate during the grating and the
changed PC spike pattern. The R21 runs should report interneuron rates for every condition.

## What this says for the R21

- The pipeline runs end to end: populations from fitted cells, tuning, rate statistics at the in vivo
  sample size, the rank test, and depth and interneuron controls. A 3 s run takes about 1 h on 4 CPUs.
- The question is well posed. Intrinsic heterogeneity of fitted cells changes the share of the top 10%
  strongly (0.30 → 0.45), and rheobase carries most of that. Whether it also makes the long upper tail is
  open, and that depends on the spread of the real library.
- The depth and interneuron controls need changes before they mean anything: depth-dependent inputs, and
  rate-matched homogeneous interneurons.

## Preview with the earlier PC→IN setting

A matched pair was first run with every PC→IN synapse at 0.82, the PR 9 rule before the PR 14 fix
(`old_pcin082_*`, not in `runs.json`). It gave the same picture: top-10% share 0.32 → 0.50, p99 6.0 → 7.2 Hz,
63% silent, and an R² of 0.66 for model identity vs 0.13 for input.

## Reproduce

```
python characterize_models.py                        # results/models/excitability.json
python build_populations.py                          # populations/
python make_homogeneous_interneurons.py --prep PREP --out PREP_HOMOG_IN
cd ../.. && export PYTHONPATH=$PWD
python -m circuit.hybrid tune --prep PREP --standard-dir Standard --reference GLIF_SPIKES \
    --population experiments/aim2_controls/populations/shuffled.json --pc-to-in-gain 1.0 \
    --target 1.36 --tstop 3000 --gains 1.24 --tol 0.03 --out RUNS/tune3_shuffled
python -m circuit.hybrid run --prep PREP --standard-dir Standard --gain 1.2837 --pc-to-in-gain 1.0 \
    --population experiments/aim2_controls/populations/depth.json --tstop 3000 --out RUNS/final_depth
python experiments/aim2_controls/analyze.py --runs experiments/aim2_controls/runs.json --prep PREP \
    --pops experiments/aim2_controls/populations --invivo npx_l23_rs_rates_all.npy --out experiments/aim2_controls/results
python experiments/aim2_controls/plot_results.py --analysis experiments/aim2_controls/results --out experiments/aim2_controls/results
```

`runs.json` lists the spike files used. They live outside git and are copied to the project files under
`aim2_controls/`. The rheobase-matched control (`populations/rheo_matched.json`) gives each PC an input scale of
its model's rheobase / 90 pA, which removes the threshold difference (results above).
