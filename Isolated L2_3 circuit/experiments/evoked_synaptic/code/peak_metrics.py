"""Per-event EPSP peak metrics on held-out traces: isolated pulses (no other pulse of either pathway within
100 ms before / 150 ms after), no recorded spike from -20 to +150 ms. Baseline = mean Vm over the 10 ms before
the pulse; peak = max over 0-60 ms (L2/3) or 0-100 ms (L1). Model events where the model spikes are counted
separately (false spikes) and their peak is clipped at the spike threshold.

    python peak_metrics.py <results_dir>   (results_dir holds test_predictions.npz, data from data_ampa/<cell>)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = {0: 100, 1: 60}


def events(V, stim, vm, test, min_int=0.0):
    rows = []
    for k, i in enumerate(test):
        for j in (0, 1):
            for t in np.nonzero(stim[i, :, j] > min_int)[0]:
                if t < 400 or t + 150 > V.shape[1]:
                    continue
                w = stim[i, t - 100:t + 150].copy(); w[100, j] = 0
                if w.sum() > 0 or (V[i, t - 20:t + 150] > -20).any():
                    continue
                b, bm = V[i, t - 10:t].mean(), vm[k, t - 10:t].mean()
                pk = V[i, t:t + WIN[j]].max() - b
                mm = np.minimum(vm[k, t:t + WIN[j]], -20.0)
                rows.append(dict(path=j, inten=float(stim[i, t, j]), base=float(b), peak=float(pk),
                                 peak_model=float(mm.max() - bm), base_model=float(bm),
                                 tpk=int(np.argmax(V[i, t:t + WIN[j]])), tpk_model=int(np.argmax(mm)),
                                 false_spike=bool((vm[k, t:t + 150] > -20).any())))
    return rows


def summarize(rows):
    out = {}
    for j, p in ((0, "L1"), (1, "L23")):
        r = [x for x in rows if x["path"] == j]
        if not r:
            continue
        pk, pm = np.array([x["peak"] for x in r]), np.array([x["peak_model"] for x in r])
        big = pk > np.percentile(pk, 50)
        out[p] = dict(n=len(r), peak_mean=float(pk.mean()), peak_model_mean=float(pm.mean()),
                      ratio=float(pm.mean() / pk.mean()), ratio_big_half=float(pm[big].mean() / pk[big].mean()),
                      slope=float(np.polyfit(pk, pm, 1)[0]), peak_rmse=float(np.sqrt(np.mean((pm - pk) ** 2))),
                      tpk=float(np.median([x["tpk"] for x in r])), tpk_model=float(np.median([x["tpk_model"] for x in r])),
                      false_spike_frac=float(np.mean([x["false_spike"] for x in r])))
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    m = json.load(open(os.path.join(d, "metrics.json")))
    data = os.path.join(HERE, "..", sys.argv[2] if len(sys.argv) > 2 else "data_ampa", m["cell"], "evoked.npz")
    z = np.load(data); p = np.load(os.path.join(d, "test_predictions.npz"))
    rows = events(z["V"], z["stim"], p["vm"], p["test"])
    s = summarize(rows)
    json.dump(dict(summary=s, events=rows), open(os.path.join(d, "peaks.json"), "w"))
    print(d, json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in s.items()}))
