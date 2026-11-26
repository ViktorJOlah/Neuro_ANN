# Experimental fully active L2/3 PC: in vivo constrained

This is the version whose tuft Ca²⁺ plateaus come easily (≥10 Hz top-down input in vivo). The sibling folder
`../In vitro constrained/` has the same code with a slow tuft K⁺ brake (`tuft_kd_scale` 4 instead of 0), so
control plateaus are rare and dendritic 4-AP unmasks the full wave, as in rodent slice data. See `../README.md`.

A detailed NEURON model of a mouse V1 L2/3 pyramidal cell with active dendrites, built to match what
Viktor's review draft ("What we really know about L2/3 PCs", v3) and Olah, Wu, Kaczmarek & Rowan 2025
(eLife 96002) report about L2/3 dendritic integration, and tested with in vivo-like input from the Allen
V1 model. It sits next to `Standard/` but is a separate, experimental line of work: it is not an ANN and
is not part of the R21 pipeline.

First working version. Everything below is a first fit by hand, not an optimisation.

## Layout

| path | what |
|---|---|
| `morphology/502359001.swc` | Allen Cux2 L2/3 PC reconstruction (axon replaced by hillock, AIS, myelin) |
| `mechanisms/` | Allen mod files, plus `Ih_Olah.mod`, `NMDA_Ca.mod`, `KA_L23.mod` (new) |
| `model/l23_cell.py`, `model/params.json` | cell builder, all densities and distance rules in one json |
| `validation/` | targets (`targets.json`, with sources), protocols and `run_validation.py` |
| `invivo/` | input builder from the Billeh V1 network, runner, analysis, `run_all.sh` |
| `results/validation/` | fig1-fig7 and `summary.json` |
| `results/invivo/` | `summary.json` and space-time figures (the raw npz files are not committed; `run_all.sh` remakes them in ~4 min) |

```
cd "Isolated L2_3 circuit/Experimental fully active/In vivo constrained"
nrnivmodl mechanisms
python validation/run_validation.py --out results/validation     # ~9 min
python invivo/build_inputs.py        # needs the v1_point network files (see the script header)
bash invivo/run_all.sh && python invivo/analyze_invivo.py results/invivo
```

## Design

* **HCN (Olah 2025).** `Ih_Olah.mod` uses the paper's fitted time constant
  tau(v) = 25.46·exp(-0.5((v+100.28)/16.2)²). Density is a Gaussian centred on the soma (sd 42.5 µm, i.e.
  ~50 µm half-width), the paper's Fig 5 best fit. Half-activation is -85 mV with tau divided by 3 (the
  recordings were at 30 °C, the model runs at 34 °C). The paper prints minf = 1/(1+exp((v+21.945)/6.85)),
  which does not match its own Fig 2i (V1/2 near -106 mV), so V1/2 was refitted (see open questions).
* **NMDA (Olah 2025).** `NMDA_Ca.mod`: Jahr-Stevens Mg block with a voltage shift (-10 mV, the paper's
  fitted "NMDA activation voltage"), 10% of the current carried by Ca²⁺ into `cai`. The NMDA:AMPA ratio
  is a soma-centred Gaussian (0.3 far away, 1.5 at the soma, sd 127 µm), the paper's Fig 4e best fit.
* **Na⁺.** NaTs2 falls exponentially along the apical dendrite (λ 150 µm) so bAPs decline gradually.
  The trunk is widened to ≥1.5 µm (the reconstruction is ~0.9 µm), which the trunk needs to couple
  dendritic spikes to the soma.
* **Ca²⁺ electrogenesis and its K⁺ brake.** HVA/LVA peak at ~80 µm (bAP Ca²⁺ hot spot) and form a
  tuft Ca²⁺ zone. A Kv4-like A current (`KA_L23.mod`) and Kv3.1 are 3x denser in the tuft and hold that
  zone back. `4APd` (dendritic 4-AP: KA, K_T, K_P x0.2, Kv3.1 x0.3) unmasks it, as Gooch 2022 showed in
  human L2/3. In mouse this is a model prediction.
* **Tuft knobs** in `params.json`: `tuft_ca_scale` (tuft HVA/LVA), `tuft_ka_scale`, `tuft_kv3_scale`,
  `tuft_kd_scale` (tuft KA, Kv3.1 and slow K_P brakes). Here 1/1/1/0.
* **Basal dendrites** have no A current. With it, clustered basal input was always sublinear; without it,
  the Branco & Häusser supralinearity appears and nothing else moved (Rin 72.7→74.1 MΩ, EPSP widths <0.5 ms).

Pharmacology in the code: `HCN`, `NMDA`, `AMPA`, `TTX`, `Ca`, `4AP` (whole cell), `4APd` (dendrites only),
`SK`, `Im`; combine with `+`, e.g. `HCN+NMDA`.

## Validation: model vs data

| measure | data | model |
|---|---|---|
| Rin control / HCN block (MΩ) | 74.8 / 105.0 | 74.1 / 116.5 |
| Vrest, ΔVrest with HCN block (mV) | -76.8, -3.5 | -77.3, -3.6 |
| tau, HCN block (ms) | ~13.5 | 12.0 |
| L4 EPSP half-width ctrl / Cs / Cs+MK801 / MK801 (ms) | 10.1 / 23.6 / 11.8 / 9.0 | 16.1 / 25.3 / 13.5 / 9.8 |
| L1 EPSP half-width ctrl / Cs / Cs+MK801 / MK801 (ms) | 14.3 / 16.2 / 19.9 / 13.6 | 15.0 / 20.6 / 17.4 / 13.1 |
| bAP amplitude at 150 µm (mV) | ~75 | 78 (50 at 250 µm, 28 at 340 µm) |
| bAP velocity (m/s) | 0.21-0.23 | 0.20 |
| bAP Ca²⁺ peak distance (µm) | 50-100 | 191 (miss) |
| current for a soma AP from a brief trunk EPSC (nA) | 1.14 | 2.6 (miss) |
| soma APs per dendritic spike | 1-2, never ≥3 | 1 |
| TTX abolishes dendritic spike | yes | yes |
| critical frequency (Hz) | 128 | tuft Ca²⁺ jumps 2.5x from 80 to 100 Hz and peaks at 140 Hz |
| Ca²⁺ boost from 4-AP at 1 nA (mV) | unmasked (human) | 11.5 → 22.5 (4APd) |
| basal NMDA supralinearity, proximal / distal (%) | 128 / 209 | 151 / 184 |

Figures: `results/validation/fig1`-`fig7`. The Ca²⁺ space-time panel (fig6) shows the three kinds of
dendritic Ca²⁺ event the model makes: a bAP-train wave that reaches into the tuft above the critical
frequency, a local tuft NMDA spike from 20 clustered synapses (Ca²⁺ in 19% of the tree, 2.3x the
NMDA-blocked response, no soma AP), and a tuft-wide Ca²⁺ plateau from a trunk-end current step.

## In vivo-like input

`invivo/build_inputs.py` takes every synapse onto a GLIF L2/3 E cell in the Billeh 2020 V1 network
(3 s of drifting gratings) together with its presynaptic spike trains (from the replay files in
`/mnt/project-files/isolated_l23`). Each edge becomes 3 synapses placed by source: LGN, L4 and L5 near
the soma; L2/3 anywhere outside the tuft; background everywhere; Pvalb perisomatic; Sst distal apical;
L1 Htr3a in the tuft. The Billeh network has no top-down cortical input, so a synthetic tuft stream
(100 Poisson inputs at 0-20 Hz) is swept. The conductance scale (3.6e-5 µS per pA) was set so cell 82141
fires at its GLIF rate (1.33 Hz). The other two cells then come out at 0.67 Hz (GLIF 0.67) and
3.3 Hz (GLIF 4.0) with no further tuning.

| cell 82141, 3 s | rate (Hz) | soma Vm | tuft Ca²⁺ events/s | tuft plateaus |
|---|---|---|---|---|
| control | 1.33 | -66.6 ± 3.6 | 1.0 (peak 144 nM) | none |
| top-down 5 Hz | 2.33 | -65.9 | 1.3 (224 nM) | none |
| top-down 10 Hz | 11.3 | -64.9 | 1.7 (959 nM) | yes, 300 ms tuft plateau driving a burst |
| top-down 20 Hz | 25.0 | -61.4 | 1.3 (2.7 µM) | yes |
| HCN block | 1.0 | -66.9 | 0.7 | none |
| HCN block, top-down 10 Hz | 11.7 | -65.4 | 1.7 | yes |
| NMDA block (any top-down) | 0 | -74.5 | 0 | none |
| dendritic 4-AP | 28.7 | -63.9 | plateaus | constant |

What this first test says:

1. **NMDA carries most of the drive.** With NMDA blocked the cell is silent at every top-down rate.
   That follows from the paper's large perisomatic NMDA:AMPA ratio, together with a drive calibrated with
   NMDA present. Whether this is right in vivo is an open question.
2. **Every dendritic Ca²⁺ event is tied to a somatic AP.** None of the tuft or basal Ca²⁺ events are
   isolated (no AP within -10..+30 ms). Without top-down input, the Ca²⁺ is bAP-driven and spread over
   85-90% of the tree.
3. **Top-down input sets a threshold for tuft Ca²⁺ plateaus.** At 10 Hz top-down input, a ~300 ms tuft
   plateau appears (µM Ca²⁺ across the tuft) and drives a high-frequency burst at the soma
   (`results/invivo/spacetime_cell82141_control_td10_s0.png`). At 5 Hz there are none. This is the
   "dendritic Ca²⁺ wave" regime. The rate jump (2.3 → 11 Hz) is steep.
4. **HCN block barely changes in vivo firing** (1.33 → 1.0 Hz), even though it widens unitary EPSPs.
5. **Dendritic 4-AP makes the cell hyperexcitable** (29-38 Hz, tuft plateaus all the time). So the K⁺
   brake is what keeps the tuft Ca²⁺ zone quiet in this model.

## Known gaps and limitations

* L4 control EPSPs are too wide (16 vs 10 ms) and the L1 HCN effect is too big (20.6 vs 16.2 ms).
* The bAP Ca²⁺ peak sits at ~190 µm (the tuft Ca²⁺ zone), not 50-100 µm.
* The dendritic current threshold is 2.6 nA, not 1.14 nA. No doublets come from brief dendritic currents,
  but bursts and doublets do appear in vivo during plateaus.
* Once triggered, the tuft Ca²⁺ plateau is close to all-or-none and lasts as long as its drive. The review
  says rodent L2/3 Ca²⁺ electrogenesis is weak, so the tuft Ca zone may be too strong in control.
* NMDA proximal bias (Olah) vs Branco's base-to-tip gradient: the model has distal supralinearity
  (184%) above proximal (151%), but its proximal branches saturate higher.
* The in vivo-like rate is steep in the conductance scale (3.6e-5 → 1.3 Hz, 4.2e-5 → 3.3 Hz). Only 3 s of
  input; one cell morphology; inputs are placed by rule, not by a connectome.

## Open questions for Viktor

1. Which HCN V1/2 did the eLife model use? The printed minf (-21.9 mV) can't be right, and the measured
   curve is near -106. -85 mV (with tau/3) fits Rin, ΔVrest and the EPSP effects here.
   **Answer (Viktor): -85 mV is correct.**
2. Is a model that is silent without NMDA acceptable? **Answer (Viktor): yes.**
3. Should the tuft Ca²⁺ zone be weaker in control (rodent)? **Answer (Viktor): build both;** the weaker one
   is `../In vitro constrained/`.
