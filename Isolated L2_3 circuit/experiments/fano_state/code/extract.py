"""Extract from one Allen Visual Coding Neuropixels session NWB: QC'd VISp (with CCF layer) and LGd unit spikes,
drifting-grating and spontaneous intervals, running speed and pupil area. -> ext/<session>.npz"""
import sys, json, numpy as np, pandas as pd, h5py
sid, nwb, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
V = pd.read_pickle("/home/user/npx/visp_units.pkl"); G = pd.read_pickle("/home/user/npx/lgd_units.pkl")
V = V[V.session == sid]; G = G[G.session == sid]
f = h5py.File(nwb, "r")
u = f["units"]; ids = u["id"][()]; idx = u["spike_times_index"][()]; st = u["spike_times"]
pos = {int(i): k for k, i in enumerate(ids)}
keep = [(int(i), "VISp", str(V.loc[i, "L"]), float(V.loc[i, "duration"])) for i in V.index if i in pos] + \
       [(int(i), "LGd", "", float(G.loc[i, "duration"])) for i in G.index if i in pos]
spikes = {}
for i, area, L, d in keep:
    k = pos[i]; a = idx[k - 1] if k else 0
    spikes[f"u{i}"] = st[a:idx[k]].astype(np.float64)
def table(name, cols):
    g = f["intervals"].get(name)
    if g is None: return None
    return {c: (g[c][()].astype(str) if g[c].dtype == object else g[c][()]) for c in cols if c in g}
dg = table("drifting_gratings_presentations", ["start_time", "stop_time", "orientation", "temporal_frequency", "contrast"]) or \
     table("drifting_gratings_75_repeats_presentations", ["start_time", "stop_time", "orientation", "temporal_frequency", "contrast"])
sp = table("spontaneous_presentations", ["start_time", "stop_time"])
r = f["processing/running/running_speed"]; run_t = r["timestamps"][()]; run_v = r["data"][()]
if run_t.ndim == 2: run_t = run_t.mean(1)
try:
    p = f["processing/filtered_gaze_mapping/pupil_area"]; pup_t, pup_v = p["timestamps"][()], p["data"][()]
except KeyError:
    pup_t, pup_v = np.zeros(0), np.zeros(0)
meta = dict(session=sid, units=[dict(id=i, area=a, layer=L, duration=d) for i, a, L, d in keep])
np.savez_compressed(out, meta=json.dumps(meta), run_t=run_t, run_v=run_v, pup_t=pup_t, pup_v=pup_v,
                    **{f"dg_{k}": v for k, v in dg.items()}, **{f"sp_{k}": v for k, v in (sp or {}).items()}, **spikes)
print(sid, "units", len(keep), "dg", len(dg["start_time"]), "pupil", len(pup_t), "run", len(run_t))
