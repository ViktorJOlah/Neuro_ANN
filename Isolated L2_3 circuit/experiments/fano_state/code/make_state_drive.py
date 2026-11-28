"""Measured behavioural-state drive for the LGN input (no free parameters).

1. For one Allen Neuropixels session, regress the LGd shared gain of every grating trial on running (>5 cm/s)
   and pupil z (state_gain.py output, gain_trials.csv). The fitted value is the state-predicted LGN gain.
2. Take the reps of one grating condition in time order (as in vivo Fano is computed) and their predicted gains,
   renormalised to mean 1 over the chosen trials so the mean LGN rate is unchanged.
3. Build each modulated LGN trial by pooling the 10 Billeh LGN Poisson trials and thinning with p = g_k / 10.
   This keeps every train Poisson with rate g_k times the original; the gain is constant within a 3 s trial
   (the measured LGd state timescale is 3.5 s, and running bouts last longer).
usage: make_state_drive.py gain_trials.csv session lgn_dir out_dir [n_trials] [seed]
"""
import sys, os, json
import numpy as np, pandas as pd

gt, sid, lgn_dir, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 10
rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
d = pd.read_csv(gt)
d = d[(d["pop"] == "LGd") & (d.session == sid)].sort_values("start").reset_index(drop=True)
X = np.c_[np.ones(len(d)), (d.run > 5).astype(float), d.pupil_z.fillna(0)]
beta = np.linalg.lstsq(X, d.gain.values, rcond=None)[0]
d["g_state"] = X @ beta
# condition with the most reps; its reps in time order
cond = d.cond.value_counts().index[0]
sel = d[d.cond == cond].head(n).copy()
g = sel.g_state.values / sel.g_state.values.mean()
os.makedirs(out, exist_ok=True)
pool = [pd.read_csv(os.path.join(lgn_dir, f"spikes.trial_{k}.csv"), sep=" ") for k in range(10)]
pool = pd.concat(pool, ignore_index=True)
for k, gk in enumerate(g):
    keep = rng.random(len(pool)) < gk / 10
    p = pool[keep].sort_values(["node_ids", "timestamps"])
    p.to_csv(os.path.join(out, f"spikes.trial_{k}.csv"), sep=" ", index=False)
sel.assign(g_used=g).to_csv(os.path.join(out, "trials.csv"), index=False)
json.dump(dict(session=sid, cond=str(cond), beta=beta.tolist(), gains=g.tolist(), gain_sd=float(g.std()),
               run_frac=float((sel.run > 5).mean())), open(os.path.join(out, "drive.json"), "w"), indent=1)
print(json.dumps(dict(gains=np.round(g, 3).tolist(), sd=float(g.std()), beta=np.round(beta, 3).tolist())))
