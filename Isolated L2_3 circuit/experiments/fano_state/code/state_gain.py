"""Shared gain of each population (LGd, VISp layers, RS units) under drifting gratings (contrast 0.8): per trial and
250 ms bin, gain = observed spikes / condition-mean spikes summed over the population's units. Split-half covariance
(random halves of units, 20 splits) gives the variance of the shared gain without Poisson noise. Its dependence on
running speed and pupil, and its timescale (from the spontaneous gray-screen blocks, 0.5 s bins), are reported.
-> out/state_gain.json, out/gain_trials.csv"""
import glob, json, os
import numpy as np, pandas as pd

D = "/home/user/npx"; OUT = os.path.join(D, "out"); os.makedirs(OUT, exist_ok=True)
BIN, NB, WIN = 0.25, 8, 2.0
POPS = {"LGd": lambda u: u["area"] == "LGd", "L2/3": lambda u: u["layer"] == "L2/3", "L4": lambda u: u["layer"] == "L4",
        "L5": lambda u: u["layer"] == "L5", "L6": lambda u: u["layer"] in ("L6a", "L6b")}
rng = np.random.default_rng(0)


def split_cov(C, E, splits=20):
    """C, E: (units, samples) observed / expected counts. Returns shared gain variance and per-sample gain estimate."""
    n = C.shape[0]; cov = []
    for _ in range(splits):
        p = rng.permutation(n); a, b = p[: n // 2], p[n // 2:]
        ga = C[a].sum(0) / E[a].sum(0); gb = C[b].sum(0) / E[b].sum(0)
        ok = np.isfinite(ga) & np.isfinite(gb)
        cov.append(np.cov(ga[ok], gb[ok])[0, 1])
    return float(np.mean(cov)), C.sum(0) / E.sum(0)


res, trial_rows = {}, []
for fn in sorted(glob.glob(os.path.join(D, "ext", "*.npz"))):
    z = np.load(fn); meta = json.loads(str(z["meta"])); sid = meta["session"]
    dg = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith("dg_")})
    dg = dg[np.isclose(dg.contrast.astype(float), 0.8)].sort_values("start_time").reset_index(drop=True)
    cond = (dg.orientation.astype(str) + "_" + dg.temporal_frequency.astype(str)).values
    edges = dg.start_time.values[:, None] + np.arange(NB + 1) * BIN                  # (trials, NB+1)
    tmid = (edges[:, :-1] + edges[:, 1:]) / 2
    run = np.interp(tmid, z["run_t"], z["run_v"])
    pup = np.interp(tmid, z["pup_t"], z["pup_v"]) if len(z["pup_t"]) else np.full(tmid.shape, np.nan)
    pz = (pup - np.nanmean(pup)) / np.nanstd(pup)
    # spontaneous blocks for the timescale
    sp = [(a, b) for a, b in zip(z["sp_start_time"], z["sp_stop_time"]) if b - a > 60] if "sp_start_time" in z.files else []
    res[sid] = {}
    for pop, f in POPS.items():
        us = [u for u in meta["units"] if f(u) and (pop == "LGd" or u["duration"] > 0.4)]
        if len(us) < 6:
            continue
        C = np.stack([np.diff(np.searchsorted(z[f"u{u['id']}"], edges), axis=1) for u in us]).astype(float)   # (units, trials, NB)
        E = np.zeros_like(C)
        for c in np.unique(cond):
            m = cond == c; E[:, m] = C[:, m].mean(1, keepdims=True)
        keep = E.sum((1, 2)) / C.shape[1] >= 0.5                                       # >= 0.5 spike per trial
        C, E = C[keep], E[keep]
        if len(C) < 6:
            continue
        var_bin, g_bin = split_cov(C.reshape(len(C), -1), E.reshape(len(C), -1))
        var_trial, g_trial = split_cov(C.sum(2), E.sum(2))
        # state dependence of the (noisy) trial gain: regress on trial-mean running and pupil
        rm, pm = run.mean(1), np.nanmean(pz, 1)
        X = np.c_[np.ones(len(rm)), rm > 5, np.nan_to_num(pm)]
        ok = np.isfinite(g_trial)
        beta = np.linalg.lstsq(X[ok], g_trial[ok], rcond=None)[0]
        pred = X[ok] @ beta
        state_var = float(np.var(pred))                                              # variance explained by state
        # timescale from spontaneous activity: split-half cross-correlation of the gain, 0.5 s bins
        tau = None
        if sp:
            ac = []
            for a, b in sp:
                eb = np.arange(a, b, 0.5)
                Cs = np.stack([np.diff(np.searchsorted(z[f"u{u['id']}"], eb)) for u, k in zip(us, keep) if k]).astype(float)
                Es = Cs.mean(1, keepdims=True) * np.ones_like(Cs)
                p = rng.permutation(len(Cs)); A, B = p[: len(p) // 2], p[len(p) // 2:]
                ga = Cs[A].sum(0) / Es[A].sum(0) - 1; gb = Cs[B].sum(0) / Es[B].sum(0) - 1
                L = min(120, len(ga) // 2)
                ac.append([np.mean(ga[: len(ga) - l] * gb[l:]) for l in range(L)])
            L = min(len(x) for x in ac); ac = np.mean([x[:L] for x in ac], 0)
            acn = ac / ac[0] if ac[0] > 0 else ac
            below = np.nonzero(acn < np.exp(-1))[0]
            tau = float(below[0] * 0.5) if len(below) else float(L * 0.5)
        res[sid][pop] = dict(n_units=int(len(C)), shared_var_bin=var_bin, shared_var_trial=var_trial,
                             shared_sd_trial=float(np.sqrt(max(var_trial, 0))), state_explained_var=state_var,
                             frac_state=float(state_var / var_trial) if var_trial > 0 else None,
                             beta_running=float(beta[1]), beta_pupil_sd=float(beta[2]), timescale_s=tau,
                             frac_running_trials=float((rm > 5).mean()))
        for i in range(len(g_trial)):
            trial_rows.append(dict(session=sid, pop=pop, trial=i, start=float(dg.start_time[i]), cond=cond[i], gain=float(g_trial[i]),
                                   run=float(rm[i]), pupil_z=float(pm[i])))
pd.DataFrame(trial_rows).to_csv(os.path.join(OUT, "gain_trials.csv"), index=False)
json.dump({str(k): v for k, v in res.items()}, open(os.path.join(OUT, "state_gain.json"), "w"), indent=1)
summ = {}
for pop in POPS:
    v = [r[pop] for r in res.values() if pop in r]
    if v:
        summ[pop] = {k: float(np.nanmedian([x[k] for x in v if x[k] is not None])) for k in
                     ("shared_sd_trial", "shared_var_bin", "frac_state", "beta_running", "beta_pupil_sd", "timescale_s")} | {"sessions": len(v)}
print(json.dumps(summ, indent=1))
