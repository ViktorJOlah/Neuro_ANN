"""How long does an evoked response last? Directly from the NEURON traces (no ANN).

For isolated pulses (single-pulse protocol, no other pulse within 400 ms after or 300 ms before, no somatic spike)
average the somatic response per pathway in intensity bins, with and without NMDA (control traces), and report the
peak, time to peak, time back to 20 % / 10 % of peak, and the fraction of the response area beyond 64 ms.
    python response_kinetics.py   ->  ../results/response_kinetics.json
"""
SET = ('_' + __import__('os').environ['EVOKED_SET']) if __import__('os').environ.get('EVOKED_SET') else ''   # e.g. EVOKED_SET=ampa -> data_ampa/, results_ampa/
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data" + SET)


def responses(V, stim, rows, pre=20, post=400):
    out = {0: [], 1: []}
    for i in rows:
        for j in (0, 1):
            for t in np.nonzero(stim[i, :, j])[0]:
                if t - 300 < 0 or t + post > V.shape[1]:
                    continue
                win = stim[i, t - 300:t + post].copy(); win[300, j] = 0
                if win.sum() > 0 or (V[i, t - pre:t + post] > -20).any():
                    continue
                base = V[i, t - pre:t].mean()
                out[j].append((stim[i, t, j], V[i, t:t + post] - base))
    return out


def stats(r):
    r = np.mean(r, 0)
    pk = r.max(); tp = int(r.argmax())
    after = r[tp:]
    t20 = tp + int(np.argmax(after < 0.2 * pk)) if (after < 0.2 * pk).any() else len(r)
    t10 = tp + int(np.argmax(after < 0.1 * pk)) if (after < 0.1 * pk).any() else len(r)
    pos = np.clip(r, 0, None)
    return dict(peak_mV=float(pk), t_peak_ms=tp, t_20pct_ms=t20, t_10pct_ms=t10,
                area_beyond_64ms=float(pos[64:].sum() / pos.sum()))


res = {}
for cell in ("521409057", "513531471"):
    z = np.load(os.path.join(DATA, cell, "evoked.npz"))
    V, stim, dc, ctrl = z["V"], z["stim"], z["dc"], list(z["ctrl_idx"])
    rows = [i for i in range(len(V)) if dc[i] == 0]
    R = responses(V, stim, rows)
    cr = [k for k, i in enumerate(ctrl) if dc[i] == 0]
    Rc = responses(V[ctrl][cr], stim[ctrl][cr], range(len(cr)))
    Rb = responses(z["V_nmda_block"][cr], stim[ctrl][cr], range(len(cr)))
    res[cell] = {}
    for j, p in enumerate(("L1", "L23")):
        strong = [r for n, r in R[j] if n >= 0.6]
        weak = [r for n, r in R[j] if n < 0.4]
        d = dict(n_strong=len(strong), strong=stats(strong) if strong else None,
                 n_weak=len(weak), weak=stats(weak) if weak else None)
        if Rc[j] and Rb[j]:
            d["control_subset"] = stats([r for _, r in Rc[j]])
            d["nmda_blocked_same_pulses"] = stats([r for _, r in Rb[j]])
            d["n_ctrl"] = len(Rc[j])
        res[cell][p] = d
json.dump(res, open(os.path.join(HERE, "..", "results" + SET, "response_kinetics.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
