"""Spike-forced residual (model - recording, mV) by time since the last recorded spike, PR 11 fits.
python residual_by_spike.py [cell ...] [--folder-tag standard]"""
import sys, json, os
import numpy as np
import timing_lib as tl

tag, stim = "standard", ""
args = [a for a in sys.argv[1:]]
if "--tag" in args:
    i = args.index("--tag"); tag = args[i + 1]; del args[i:i + 2]
if "--stim" in args:
    i = args.index("--stim"); stim = args[i + 1]; del args[i:i + 2]
bins = [5, 10, 20, 40, 80, 160, 320, 1e10]
out = {}
for cell in args or tl.CELLS:
    m, p = tl.load_fitted(cell, os.path.join(tl.RC, "results", cell, tag))
    sw = [s for s in tl.load_sweeps(cell) if s["split"] == "train" and s["n_spikes"] > 0 and stim in s["stimulus"]][:8]
    E = {b: [] for b in bins}
    for s in sw:
        I, V = s["I"][None], s["V"][None]
        vc, sc = tl.simulate(m, I, V, tl.spike_clamp(V, tl.SEED_MS))
        td = tl.gt_spike_ms(V[0])
        last = np.full(V.shape[1], 1e9)
        for t0 in td:
            last[t0:] = np.arange(V.shape[1] - t0)
        ok = (np.arange(V.shape[1]) >= tl.SEED_MS) & (V[0] < -45) & tl.away_mask(V.shape[1], td, 3, 5) & ~sc[0]
        for t in np.flatnonzero(ok):
            b = next(b for b in bins if last[t] < b)
            E[b].append(vc[0, t] - V[0, t])
    out[cell] = {str(b): (float(np.mean(e)), float(np.sqrt(np.mean(np.square(e)))), len(e)) for b, e in E.items() if e}
    print(cell, {k: (round(v[0], 2), round(v[1], 2), v[2]) for k, v in out[cell].items()}, flush=True)
