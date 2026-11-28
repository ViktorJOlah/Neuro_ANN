"""In vivo noise correlations (2 s spike counts, per grating condition at contrast 0.8, z-scored within condition)
for VISp L2/3 RS unit pairs in the same session, all trials and still trials (< 1 cm/s). Pairs where both units
have mean count >= 1 in that condition. -> out/noisecorr_invivo.json"""
import glob, json, os
import numpy as np, pandas as pd
D = "/home/user/npx"; WIN = 2.0
out = dict(all=[], still=[]); per_session = []
for fn in sorted(glob.glob(os.path.join(D, "ext", "*.npz"))):
    z = np.load(fn, allow_pickle=False); meta = json.loads(str(z["meta"]))
    us = [u for u in meta["units"] if u["area"] == "VISp" and u["layer"] == "L2/3" and u["duration"] > 0.4]
    if len(us) < 2: continue
    dg = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith("dg_")})
    dg = dg[np.isclose(dg.contrast.astype(float), 0.8)].sort_values("start_time").reset_index(drop=True)
    cond = (dg.orientation.astype(str) + "_" + dg.temporal_frequency.astype(str)).values
    run = np.array([np.nanmean(np.interp(np.linspace(a, a + WIN, 20), z["run_t"], z["run_v"])) for a in dg.start_time])
    C = np.stack([np.searchsorted(z[f"u{u['id']}"], dg.start_time + WIN) - np.searchsorted(z[f"u{u['id']}"], dg.start_time) for u in us], 1)
    sess = dict(all=[], still=[])
    for c in np.unique(cond):
        for key, m in (("all", cond == c), ("still", (cond == c) & (run < 1))):
            X = C[m].astype(float)
            if len(X) < 5: continue
            ok = (X.mean(0) >= 1) & (X.std(0) > 0); X = X[:, ok]
            if X.shape[1] < 2: continue
            R = np.corrcoef(X.T); iu = np.triu_indices(len(R), 1); sess[key] += R[iu].tolist()
    for k in out: out[k] += sess[k]
    per_session.append(dict(session=meta["session"], n_units=len(us), all=float(np.nanmean(sess["all"])) if sess["all"] else None,
                            still=float(np.nanmean(sess["still"])) if sess["still"] else None))
res = {k: dict(mean=float(np.nanmean(v)), median=float(np.nanmedian(v)), n_pair_conditions=len(v)) for k, v in out.items()}
res["per_session_median_all"] = float(np.nanmedian([s["all"] for s in per_session if s["all"] is not None]))
res["per_session_median_still"] = float(np.nanmedian([s["still"] for s in per_session if s["still"] is not None]))
res["sessions"] = per_session
json.dump(res, open(os.path.join(D, "out", "noisecorr_invivo.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "sessions"}, indent=1))
