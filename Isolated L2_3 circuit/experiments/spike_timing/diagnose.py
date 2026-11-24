"""Where does the Standard lose spike timing on real noise? Free vs spike-forced runs of the fitted models."""
import sys, json, time
import numpy as np
import timing_lib as tl

out = {}
for cell in sys.argv[1:] or tl.CELLS:
    m, p = tl.load_fitted(cell)
    sw = [s for s in tl.load_sweeps(cell) if "Noise" in s["stimulus"]]
    res = []
    for s in sw:
        I, V = s["I"][None], s["V"][None]
        T = V.shape[1]
        x = m.current(I)
        td = tl.gt_spike_ms(V[0])
        t0 = time.time()
        vf, sf = tl.simulate(m, I, V, tl.spike_clamp(V, tl.SEED_MS, 0, -1), x=x)  # seed only
        vc, sc = tl.simulate(m, I, V, tl.spike_clamp(V, tl.SEED_MS), x=x)
        tf_ = np.flatnonzero(sf[0]); tc = np.flatnonzero(sc[0])
        td_ = td[td >= tl.SEED_MS]
        aw = tl.away_mask(T, np.concatenate([td, tf_])); aw[:tl.SEED_MS] = False
        awc = tl.away_mask(T, np.concatenate([td, tc])); awc[:tl.SEED_MS] = False
        Vcl = np.minimum(V[0], -40)
        r = dict(sweep=s["sweep"], stim=s["stimulus"], split=s["split"], n_gt=len(td_), n_free=len(tf_), n_forced=len(tc),
                 g_free=tl.gamma(td_, tf_, T - tl.SEED_MS), g_forced=tl.gamma(td_, tc, T - tl.SEED_MS),
                 rmse_free=float(np.sqrt(np.mean((vf[0][aw] - V[0][aw]) ** 2))),
                 rmse_forced=float(np.sqrt(np.mean((vc[0][awc] - V[0][awc]) ** 2))),
                 bias_forced=float(np.mean(vc[0][awc] - V[0][awc])), sec=time.time() - t0)
        # model Vm just before recorded spikes (forced run) vs model threshold
        pre = [vc[0][t - 2] for t in td_ if t - 2 >= 0 and not sc[0][t - 6:t].any()]
        r["vm_pre2_med"] = float(np.median(pre)) if pre else None
        r["V_th"] = p["branches"]["V_th"]
        res.append(r)
        print(cell, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out[cell] = res
json.dump(out, open(f"/tmp/claude-0/diag_{'_'.join(sys.argv[1:]) or 'all'}.json", "w"), indent=1)
