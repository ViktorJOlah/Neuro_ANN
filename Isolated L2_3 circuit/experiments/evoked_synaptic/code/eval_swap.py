"""Swap test: are the input transform D and the somatic model S separable?

Chimera <A>_soma<B> = cell A's morphology and dendritic (apical + basal) parameters with cell B's somatic and
axonal parameters, stimulated with A's synapses at A's stimulus strengths (evoked_sim.py --chimera-soma).
If D captures input processing and S the somatic transform, D_A -> S_B should predict the chimera better than
the three other pairings. All four D/S are fitted on the parent cells only; nothing is fitted on the chimera.

    python eval_swap.py --a 521409057 --b 513531471 --model tcn256
Writes ../results/swap_<model>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import fit_input_transform as F

HERE = os.path.dirname(os.path.abspath(__file__))


def load_D(cell, model):
    D = F.build_D(model)
    D.load_weights(os.path.join(HERE, "..", "results", cell, f"{model}_all", "D.weights.h5"))
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="521409057")
    ap.add_argument("--b", default="513531471")
    ap.add_argument("--model", default="tcn256")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    cells = (args.a, args.b)
    D = {c: load_D(c, args.model) for c in cells}
    S = {c: F.load_somatic(c) for c in cells}
    out = {}
    for dend, soma in ((args.a, args.b), (args.b, args.a)):
        tag = f"{dend}_soma{soma}"
        data = F.load_data(tag)
        stim, dc, V = data["stim"].astype(np.float32), data["dc"].astype(np.float32), data["V"].astype(np.float32)
        res = {}
        for dd in cells:
            for ss in cells:
                vm = F.simulate(S[ss], D[dd], stim, dc, V)
                r = F.score(vm, V, stim)
                res[f"D{'A' if dd == args.a else 'B'}_S{'A' if ss == args.a else 'B'}"] = r
                print(tag, f"D_{dd} S_{ss}", {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        # the parents themselves on the same stimuli are not simulated; report the parents' own test scores instead
        out[tag] = dict(dendrites_from=dend, soma_from=soma, n_traces=int(len(V)), scores=res,
                        expected_best=f"D{'A' if dend == args.a else 'B'}_S{'A' if soma == args.a else 'B'}")
    json.dump(dict(A=args.a, B=args.b, model=args.model, chimeras=out),
              open(os.path.join(HERE, "..", "results", f"swap_{args.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
