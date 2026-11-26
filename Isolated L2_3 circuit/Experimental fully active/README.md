# Experimental fully active L2/3 PC

A detailed NEURON L2/3 PC with active dendrites (Olah et al. 2025 eLife HCN and NMDA, tuft Ca²⁺ zone),
built on Viktor's review of L2/3 dendritic integration and tested with in vivo-like input from the Allen
V1 model. Two versions, side by side, with identical code. Only the tuft Ca²⁺ zone's K⁺ brake differs:

| folder | tuft Ca²⁺ zone | constrained by |
|---|---|---|
| `In vivo constrained/` | plateaus come easily: a trunk-end step or ≥10 Hz top-down input sets off a ~300 ms tuft Ca²⁺ plateau and a somatic burst | in vivo dendritic Ca²⁺ imaging (tuft events with top-down input) |
| `In vitro constrained/` | plateaus are rare in control (cut short by a slow 4-AP-sensitive K⁺ brake); dendritic 4-AP unmasks the full wave | rodent slice data (weak L2/3 Ca²⁺ electrogenesis; Larkum 2007, Waters 2003; Gooch 2022 4-AP) |

The knobs are in each `model/params.json`: `tuft_ca_scale` (tuft HVA/LVA), `tuft_ka_scale`,
`tuft_kv3_scale`, `tuft_kd_scale` (tuft KA, Kv3.1, slow K_P). In vivo constrained: 1/1/1/0. In vitro
constrained: 1/1/1/4. Anything in between gives an intermediate cell.

Each folder is self-contained: `nrnivmodl mechanisms`, then `validation/run_validation.py` and
`invivo/run_all.sh` (see its README).
