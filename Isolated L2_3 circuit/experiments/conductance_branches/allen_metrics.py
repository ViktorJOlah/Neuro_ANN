"""Allen-style summary metrics for the fitted models (continuous prediction, test set).

Explained variance ratio (GLIF papers, Teeter et al. 2018): spike trains of model and data are each
convolved with a Gaussian (sigma 10 ms); EV = 1 - var(model - data) / var(data), divided by the EV
between repeated recordings of the cell. The ground truth here is a deterministic NEURON model, so
the data-to-data EV is 1 and the ratio equals the EV. Computed per noise test trace (as Allen uses
its noise sweeps) and averaged over traces with at least one NEURON spike.

Feature errors (as in the Allen biophysical "feature average"): spike features on the square test
steps (step 500-1500 ms), model vs NEURON per trace: spike count, latency to first spike, mean ISI,
ISI CV, adaptation index (eFEL adaptation_index2), AHP depth (mean Vm trough between spikes).
Allen divides each feature error by the experimental across-sweep SD and averages; those SDs come
from the real cell's repeated sweeps and are not available here, and AP height/width do not exist
for an Izhikevich spike. So the errors are given in natural units and as a fraction of a reference
spread: the SD of the NEURON feature across square sweeps within +-20 pA of the same amplitude
(a stand-in for sweep-to-sweep variability; floored at 5% of the NEURON value or FLOOR), and the mean of those z-scores as a "feature average".
Run from "Isolated L2_3 circuit/":  python experiments/conductance_branches/allen_metrics.py results_slow_k [results_fast_k ...]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb
sp, fs = sb.sp, sb.fs
MODELS = ("dense3", "cnnlstm", "dense_cnnlstm")
SIG = 10.0
ON, OFF = 500, 1500
FLOOR = dict(count=0.5, latency=1.0, mean_isi=1.0, isi_cv=0.02, adaptation=0.01, ahp_depth=0.5)


def ev(a, b):
    k = np.arange(-int(4 * SIG), int(4 * SIG) + 1); g = np.exp(-0.5 * (k / SIG) ** 2)
    sa, sb_ = np.convolve(a.astype(float), g, "same"), np.convolve(b.astype(float), g, "same")
    return 1 - np.var(sb_ - sa) / np.var(sa) if np.var(sa) > 0 else np.nan


def feats(vm, spk, inspk=None):
    inspk = spk if inspk is None else inspk      # samples to skip when looking for the AHP trough
    t = np.flatnonzero(spk[ON:OFF]) + ON
    f = dict(count=float(len(t)), latency=float(t[0] - ON) if len(t) else np.nan)
    isi = np.diff(t)
    f["mean_isi"] = float(isi.mean()) if len(isi) else np.nan
    f["isi_cv"] = float(isi.std() / isi.mean()) if len(isi) > 1 else np.nan
    f["adaptation"] = float(np.mean((isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1]))) if len(isi) > 2 else np.nan
    tr = [vm[a + 2:b - 1][~inspk[a + 2:b - 1]].min() for a, b in zip(t[:-1], t[1:]) if b - a > 4]
    f["ahp_depth"] = float(np.mean(tr)) if tr else np.nan
    return f


def main():
    sets = fs.load_split()
    nz = sets[("noise", "test")]; Vn = nz["V"]; Sn, _ = sp.gt_masks(Vn); Sn = Sn > 0.5
    sq = sets[("square", "test")]; Vs = sq["V"]; Ss, Ms = sp.gt_masks(Vs); Ss = Ss > 0.5; Ms = Ms > 0.5
    amp = sq["I"][:, 1000]
    gt_f = [feats(Vs[i], Ss[i], Ss[i] | Ms[i]) for i in range(len(Vs))]
    keys = list(gt_f[0])
    gt_arr = {k: np.array([f[k] for f in gt_f]) for k in keys}
    # reference spread: NEURON feature SD over sweeps within +-20 pA (stand-in for repeat variability)
    ref_sd = {k: np.array([np.nanstd(gt_arr[k][np.abs(amp - amp[i]) <= 20]) for i in range(len(amp))]) for k in keys}
    out = {}
    for d in sys.argv[1:] or ["results_slow_k"]:
        out[d] = {}
        for m in MODELS:
            z = np.load(os.path.join(HERE, d, m, "predictions.npz"))
            ns = z["noise_cont_spikes"].astype(bool)
            evs = [ev(Sn[i, sb.base.SEED_MS:], ns[i, sb.base.SEED_MS:]) for i in range(len(Vn)) if Sn[i].sum() > 0]
            vm, spk = z["square_cont_vm"], z["square_cont_spikes"].astype(bool)
            mf = [feats(vm[i], spk[i]) for i in range(len(vm))]
            res = {"ev_ratio_noise": float(np.nanmean(evs)), "features": {}}
            zs = []
            for k in keys:
                mk = np.array([f[k] for f in mf]); ok = np.isfinite(mk) & np.isfinite(gt_arr[k]) & (amp > 0)
                err = np.abs(mk - gt_arr[k])[ok]
                # floor: 5% of the NEURON value or a small absolute floor, so near-constant features don't explode
                sd = np.maximum(ref_sd[k][ok], np.maximum(0.05 * np.abs(gt_arr[k][ok]), FLOOR[k]))
                zk = err / sd
                res["features"][k] = dict(neuron_mean=float(np.mean(gt_arr[k][ok])), model_mean=float(np.mean(mk[ok])),
                                          mean_abs_err=float(err.mean()), z=float(np.median(zk)), n=int(ok.sum()))
                zs.append(np.median(zk))
            res["feature_average_z"] = float(np.mean(zs))
            out[d][m] = res
            print(d, m, "EV ratio (noise) %.3f" % res["ev_ratio_noise"], "feature avg %.2f" % res["feature_average_z"],
                  {k: (round(v["neuron_mean"], 2), round(v["model_mean"], 2), round(v["z"], 2)) for k, v in res["features"].items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "results_allen_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
