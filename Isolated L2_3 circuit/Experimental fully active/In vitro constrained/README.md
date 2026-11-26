# Experimental fully active L2/3 PC: in vitro constrained

Same code as `../In vivo constrained/`; only one parameter differs: `tuft_kd_scale` is 4 here (0 there).
That adds a slow, 4-AP-sensitive K⁺ conductance (K_P, 0.04 S/cm²) to the tuft. It barely touches brief
events (bAPs, the critical-frequency boost, tuft NMDA spikes) but cuts tuft Ca²⁺ plateaus short. So, as
in rodent slices (Larkum 2007, Waters 2003), control plateaus are rare and dendritic 4-AP unmasks the full
Ca²⁺ wave. Design, sources and the full description of every protocol are in the in vivo constrained README;
this one lists only what differs.

Why a slow K⁺ brake and not more Kv4: making the tuft A-type current 3-10x denser also silenced the
control plateau, but it stopped bAPs from entering the tuft (no tuft Ca²⁺ at any bAP frequency, so no
critical frequency). Scaling the tuft Ca²⁺ channels down instead weakened the 4-AP plateau as much as
the control one.

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
cd "Isolated L2_3 circuit/Experimental fully active/In vitro constrained"
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

## Validation: what differs from the in vivo constrained version

Passive, EPSP pathways, bAP amplitudes and basal NMDA supralinearity are identical (the tuft K_P is shut
at rest). Differences:

| measure | in vivo constrained | in vitro constrained |
|---|---|---|
| 0.5 nA, 50 ms step at trunk end: tuft Ca²⁺, soma APs | 4.7 µM plateau outlasting the step, 9 APs | 3.0 µM spike cut short at ~50 ms, 2 APs |
| same step with dendritic 4-AP | 4.5 µM, 13 APs | 4.8 µM plateau, 10 APs (full wave unmasked) |
| Ca²⁺ boost at 1 nA (mV), control → 4APd | 11.5 → 22.5 | 5.1 → 20.0 |
| soma APs, 1 nA 50 ms trunk step, control / 4APd | 8 / 10 | 2 / 8 |
| tuft Ca²⁺ vs bAP frequency (integral, 80/100/120/140 Hz) | 2.0 / 5.2 / 10.3 / 15.6 | 1.4 / 2.8 / 5.0 / 7.0 |
| 5 bAPs at 143 Hz, peak tuft Ca²⁺ | 748 nM | 531 nM |
| tuft NMDA spike (20 clustered syn) | 203 nM, local | 200 nM, local |

The critical-frequency boost is kept (tuft Ca²⁺ rises 5x from 80 to 140 Hz) but is more gradual.

## In vivo-like input

Same inputs and placement as the in vivo constrained version. The conductance scale is 3.65e-5 µS/pA.
The rate moves in whole spikes over 3 s, so cell 82141 fires at 1.0 Hz (GLIF 1.33); the next step up,
3.675e-5, gives 1.67 Hz. Cells 81559 and 80733 come out at 0.67 and 3.0 Hz (GLIF 0.67 and 4.0).

| cell 82141, 3 s | rate (Hz) | tuft Ca²⁺ peak | tuft plateaus |
|---|---|---|---|
| control | 1.0 | 88 nM | none |
| top-down 5 Hz | 1.33 | 111 nM | none |
| top-down 10 Hz | 2.33 | 163 nM | none (in vivo constrained: 300 ms plateau, 11 Hz) |
| top-down 20 Hz | 3.67 | 298 nM | none (in vivo constrained: 2.7 µM, 25 Hz) |
| HCN block, top-down 10 Hz | 2.33 | 172 nM | none |
| NMDA block (any top-down) | 0 | – | none |
| dendritic 4-AP, top-down 0/5/10 Hz | 26 / 26 / 30 | 3.3-3.6 µM | constant |

So in this version, top-down input up to 20 Hz raises the rate gradually and never sets off a tuft
plateau. Dendritic 4-AP unmasks plateaus and a 26-30 Hz bursting mode, as in the in vivo constrained one.
As there, every dendritic Ca²⁺ event comes with a somatic AP.

Figures: `results/validation/fig1`-`fig7`, `results/invivo/spacetime_*.png`.

## Known gaps

Same as the in vivo constrained version (L4 control EPSP width, bAP Ca²⁺ peak distance, dendritic
current threshold), except the all-or-none tuft plateau, which is what this version changes.
