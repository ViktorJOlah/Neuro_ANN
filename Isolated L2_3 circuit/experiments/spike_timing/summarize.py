"""Quick table: Noise 2 Γ, spike-count bias and Vm RMSE away from spikes on the test sweeps, per cell and variant."""
import sys, os, numpy as np
import timing_lib as tl
variants = sys.argv[1:] or ["standard", "standard_branches", "smooth_from_branches"]
rows = {}
for c in tl.CELLS:
    test = [s for s in tl.load_sweeps(c) if s["split"] == "test"]
    for v in variants:
        f = os.path.join(tl.RC, "results", c, v, "predictions.npz")
        if not os.path.exists(f):
            continue
        p = np.load(f)
        g, cb, rm, gsq, rms = [], [], [], [], []
        for s in test:
            T = len(s["V"]); td = tl.gt_spike_ms(s["V"]); td = td[td >= tl.SEED_MS]
            tm = p[f"s{s['sweep']}_spk"]; tm = tm[tm >= tl.SEED_MS]
            vm = p[f"s{s['sweep']}_vm"]
            aw = np.ones(T, bool); aw[:tl.SEED_MS] = False
            for t in np.concatenate([td, tm]).astype(int):
                aw[max(0, t - 5):t + 30] = False
            r = float(np.sqrt(np.mean((vm[aw] - s["V"][aw]) ** 2))) if aw.any() else np.nan
            if s["stimulus"] == "Noise 2":
                g.append(tl.gamma(td, tm, T - tl.SEED_MS)); cb.append(len(tm) - len(td)); rm.append(r)
            elif s["stimulus"] == "Long Square":
                rms.append(r)
        rows[(c, v)] = (np.mean(g), np.mean(cb), np.mean(rm), np.nanmean(rms))
print(f"{'variant':26s} " + " ".join(f"{c:>22s}" for c in tl.CELLS) + "   median Γ")
for v in variants:
    vals = [rows.get((c, v)) for c in tl.CELLS]
    print(f"{v:26s} " + " ".join(f"Γ{x[0]:5.2f} b{x[1]:+4.0f} {x[2]:3.1f}/{x[3]:3.1f}" if x else " " * 22 for x in vals)
          + f"   {np.median([x[0] for x in vals if x]):.3f}")
print("cells: Γ = Noise 2 Γ, b = spike-count bias per Noise 2 sweep, a/b = Vm RMSE away from spikes noise/long-square (mV)")
