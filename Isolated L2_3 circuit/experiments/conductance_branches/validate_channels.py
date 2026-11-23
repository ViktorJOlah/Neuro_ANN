"""Check the 1 ms branch kinetics (channels.py) against NEURON (neuron_reference.py, train100).

The branches are driven by NEURON's own somatic voltage sampled at 1 ms, in the same way the
model drives them: at a spike onset sample the branches see the 1 ms spike pulse (V_SPIKE), and
on the following in-spike samples the reset voltage c (here -46 mV, the typical post-spike Vm).
Compares Na current below threshold, calcium, and SK current with NEURON's recordings.
"""
import json, os, sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spiking"))
import channels as ch
import spiking as sp

NREF = os.environ.get("NEURON_REF", "/tmp/nref")
OUT = os.path.join(HERE, "results", "validation")
os.makedirs(OUT, exist_ok=True)
V_SPIKE = sp.SPIKE_DISPLAY_MV - ch.JP


def drive(V, S, M, mode):
    n, T = V.shape
    st = ch.init_state(tf.constant(V[:, 0] - ch.JP))
    na, sk, ca = [], [], []
    for t in range(T):
        if mode == "pulse":
            vb = S[:, t] * V_SPIKE + (1 - S[:, t]) * (M[:, t] * (-46.0 - ch.JP) + (1 - M[:, t]) * (V[:, t] - ch.JP))
        else:
            vb = V[:, t] - ch.JP
        st, cur = ch.step(st, tf.constant(vb, tf.float32))
        na.append((cur["NaTs"] + cur["Nap"]).numpy()); sk.append(cur["SK"].numpy()); ca.append(st["cai"].numpy())
    return np.stack(na, 1), np.stack(sk, 1), np.stack(ca, 1)


res = {}
fig, ax = plt.subplots(3, 3, figsize=(15, 9))
for j, pr in enumerate(sp.PROTOS):
    z = np.load(os.path.join(NREF, f"train100_{pr}.npz"))
    V = z["control_v"]
    S, M = sp.gt_masks(V)
    nS = S[:, 450:].sum(1)
    na, sk, ca = drive(V, S, M, "pulse")
    na_raw, sk_raw, ca_raw = drive(V, S, M, "raw")
    sub = (V < -45) & (M < 0.5)
    na_n = z["control_ina_NaTs"] + z["control_ina_Nap"]
    d = dict(
        na_subthreshold_corr=float(np.corrcoef(na[sub], na_n[sub])[0, 1]),
        na_subthreshold_ratio=float(np.median(na[sub & (V > -55)] / na_n[sub & (V > -55)])),
        ca_rise_ratio_pulse=float(np.median((ca[:, -1] - ch.MIN_CAI)[nS > 0] / (z["control_cai"][:, -1] - ch.MIN_CAI)[nS > 0])),
        ca_rise_ratio_raw=float(np.median((ca_raw[:, -1] - ch.MIN_CAI)[nS > 0] / (z["control_cai"][:, -1] - ch.MIN_CAI)[nS > 0])),
        ca_corr_pulse=float(np.corrcoef(ca.ravel(), z["control_cai"].ravel())[0, 1]),
        sk_corr_pulse=float(np.corrcoef(sk[M < 0.5], z["control_ik_SK"][M < 0.5])[0, 1]),
        sk_ratio_pulse=float(np.sum(sk[M < 0.5]) / np.sum(z["control_ik_SK"][M < 0.5])),
        sk_peak_pA_neuron=float(np.percentile(z["control_ik_SK"] * ch.PA_PER_DENSITY, 99.9)))
    res[pr] = d
    print(pr, json.dumps({k: round(v, 3) for k, v in d.items()}), flush=True)
    i = int(np.argmax(nS))
    t = np.arange(V.shape[1]) / 1000
    ax[0, j].plot(t, V[i], "k", lw=0.6); ax[0, j].set_title(f"{pr}: NEURON Vm (trace with most spikes)")
    ax[1, j].plot(t, z["control_cai"][i] * 1e3, "k", label="NEURON"); ax[1, j].plot(t, ca[i] * 1e3, "C1", label="1 ms branch")
    ax[1, j].set_ylabel("[Ca]i (µM)"); ax[1, j].legend(fontsize=8)
    ax[2, j].plot(t, z["control_ik_SK"][i] * ch.PA_PER_DENSITY, "k", lw=0.8, label="NEURON")
    ax[2, j].plot(t, sk[i] * ch.PA_PER_DENSITY, "C1", lw=0.8, label="1 ms branch"); ax[2, j].set_ylabel("I_SK (pA)")
    ax[2, j].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "branches_vs_neuron.png"), dpi=130)
json.dump(res, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
