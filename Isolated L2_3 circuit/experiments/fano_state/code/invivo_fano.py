"""In vivo Fano factors of V1 units under drifting gratings, split by behavioural state, and the state modulation
of each population (LGd, V1 layers). Reads ext/<session>.npz (extract.py). -> out/fano_invivo.json, out/units.csv"""
import glob, json, os, sys
import numpy as np, pandas as pd

D = "/home/user/npx"; OUT = os.path.join(D, "out"); os.makedirs(OUT, exist_ok=True)
STILL, RUN = 1.0, 5.0          # cm/s, trial-mean running speed: still < 1, running > 5
WIN = 2.0                      # s, counting window from grating onset (= Allen fano_dg)
MIN_COUNT = 1.0                # mean spikes per trial, as in the model analysis


def interp_state(z, t0, t1, n=40):
    tt = np.linspace(t0, t1, n)
    run = np.interp(tt, z["run_t"], z["run_v"])
    pup = np.interp(tt, z["pup_t"], z["pup_v"]) if len(z["pup_t"]) else np.full(n, np.nan)
    return run, pup


def fano(c):
    c = np.asarray(c, float)
    return np.nan if len(c) < 5 or c.mean() <= 0 else c.var(ddof=1) / c.mean()


rows = []
for fn in sorted(glob.glob(os.path.join(D, "ext", "*.npz"))):
    z = np.load(fn, allow_pickle=False)
    meta = json.loads(str(z["meta"])); sid = meta["session"]
    dg = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith("dg_")})
    dg = dg[np.isclose(dg.contrast.astype(float), 0.8)].sort_values("start_time").reset_index(drop=True)
    cond = dg.orientation.astype(str) + "_" + dg.temporal_frequency.astype(str)
    st = np.array([interp_state(z, a, a + WIN) for a in dg.start_time])            # (trials, 2, n)
    run_m = np.nanmean(st[:, 0], 1)
    pup_m = np.nanmean(st[:, 1], 1)
    pup_z = (pup_m - np.nanmean(pup_m)) / np.nanstd(pup_m) if np.isfinite(pup_m).any() else pup_m
    for u in meta["units"]:
        s = z[f"u{u['id']}"]
        c = np.searchsorted(s, dg.start_time + WIN) - np.searchsorted(s, dg.start_time)
        m = pd.Series(c).groupby(cond.values).mean()
        pref = m.idxmax(); sel = (cond == pref).values
        cp, rp, pp = c[sel], run_m[sel], pup_z[sel]
        # slow drift removed: residual from the mean of the 4 nearest repeats of the same condition (in time)
        k = np.arange(len(cp)); det = []
        for i in k:
            nb = np.argsort(np.abs(k - i))[1:5]
            det.append(cp[i] - cp[nb].mean())
        det = np.array(det) / np.sqrt(1 + 1 / 4)            # residual var = var * (1 + 1/4) for independent trials
        rows.append(dict(session=sid, unit=u["id"], area=u["area"], layer=u["layer"], rs=u["duration"] > 0.4,
                         n_trials=int(sel.sum()), mean_count=float(cp.mean()), fano_all=fano(cp),
                         fano_still=fano(cp[rp < STILL]), n_still=int((rp < STILL).sum()),
                         fano_run=fano(cp[rp > RUN]), n_run=int((rp > RUN).sum()),
                         fano_detrended=float(det.var(ddof=1) / cp.mean()) if cp.mean() > 0 else np.nan,
                         frac_trials_running=float((rp > RUN).mean()),
                         rate_mod_run=float(cp[rp > RUN].mean() / cp[rp < STILL].mean()) if (rp > RUN).sum() >= 3 and (rp < STILL).sum() >= 3 and cp[rp < STILL].mean() > 0 else np.nan))
U = pd.DataFrame(rows); U.to_csv(os.path.join(OUT, "units.csv"), index=False)
res = {}
for (area, layer), g in U[U.rs & (U.mean_count >= MIN_COUNT)].groupby(["area", "layer"]):
    r = {}
    for k in ("fano_all", "fano_still", "fano_run", "fano_detrended"):
        v = g[k].dropna(); r[k] = dict(median=float(v.median()) if len(v) else None, n=int(len(v)))
    r["rate_mod_run_median"] = float(g.rate_mod_run.dropna().median()) if g.rate_mod_run.notna().any() else None
    res[f"{area} {layer}".strip()] = r
json.dump(res, open(os.path.join(OUT, "fano_invivo.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
