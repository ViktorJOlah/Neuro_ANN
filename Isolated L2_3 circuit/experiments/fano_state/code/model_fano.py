"""Fano factor, noise correlation and tail of L2/3 PCs over 10 trials: Poisson-input baseline (kernel_rerun trials)
vs the measured LGN state drive (run_state_trials.sh). 1-3 s counts, cells with mean count >= 1.
    python code/model_fano.py [drive]  -> results/model_fano_<drive>.json
"""
import json, os, sys
import h5py, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from circuit.hybrid import _read_spikes  # noqa: E402
V1 = os.environ.get("V1W", "/home/user/v1"); drive = sys.argv[1] if len(sys.argv) > 1 else "state_756029989"
pc = h5py.File(f"{V1}/H/ids.h5")["pc_to_l23"][()]

def counts(path, a=1000, b=3000):
    i, t = _read_spikes(path, "l23"); m = (t >= a) & (t < b)
    return np.bincount(i[m], minlength=65949)[:65949][pc]

def stats(C, g=None, rng=np.random.default_rng(0)):
    mu = C.mean(0); act = mu >= 1; ff = C.var(0, ddof=1)[act] / mu[act]
    A = C[:, act]; z = (A - A.mean(0)) / (A.std(0) + 1e-12)
    ii = rng.integers(0, A.shape[1], (20000, 2)); ii = ii[ii[:, 0] != ii[:, 1]]
    nc = np.nanmean(z[:, ii[:, 0]] * z[:, ii[:, 1]], 0) * len(C) / (len(C) - 1)
    r = np.sort(mu / 2)[::-1]; top = r[: len(r) // 10].sum() / r.sum()
    out = dict(fano_median=float(np.median(ff)), fano_iqr=np.percentile(ff, [25, 75]).tolist(), n_cells=int(act.sum()),
               noise_corr_mean=float(np.nanmean(nc)), top10_share=float(top), mean_rate=float(mu.mean() / 2),
               pop_count_per_trial=C.sum(1).tolist())
    if g is not None:
        out["corr_pop_count_vs_lgn_gain"] = float(np.corrcoef(C.sum(1), g)[0, 1])
    return out, ff

g = np.array(json.load(open(f"{V1}/v1_point/lgn_stimulus/{drive}/drive.json"))["gains"])
sets = {"GLIF, Poisson LGN": [f"{V1}/runs/l23_reference_spikes.h5"] + [f"{V1}/trials/t{k}/streams/l23_reference_spikes.h5" for k in range(1, 10)],
        "ANN-PC, Poisson LGN": [f"{V1}/K/standard_inadj3/spikes.h5"] + [f"{V1}/trials/t{k}/hyb/spikes.h5" for k in range(1, 10)],
        "GLIF, LGN state drive": [f"{V1}/{drive}/t{k}/streams/l23_reference_spikes.h5" for k in range(10)],
        "ANN-PC, LGN state drive": [f"{V1}/{drive}/t{k}/hyb/spikes.h5" for k in range(10)]}
res, ffs = {}, {}
for k, fl in sets.items():
    res[k], ffs[k] = stats(np.stack([counts(f) for f in fl]), g if "state" in k else None)
os.makedirs(f"{HERE}/results", exist_ok=True)
json.dump(dict(drive=drive, lgn_gains=g.tolist(), results=res), open(f"{HERE}/results/model_fano_{drive}.json", "w"), indent=1)
np.savez(f"{HERE}/results/model_fano_{drive}.npz", **{k.replace(" ", "_").replace(",", ""): v for k, v in ffs.items()})
for k, v in res.items():
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "pop_count_per_trial"})
