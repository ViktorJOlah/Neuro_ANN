"""Check the Kv3.1 branch (and look at the BK branch) against NEURON at 1 ms.

The branches are driven with NEURON's own 1 ms voltage in the model's convention: a spike onset
sample is the 1 ms spike pulse (V_SPIKE), the rest of the spike is the reset level, and the current
of the spike step itself is not passed on. Compared: the Kv3.1 after-current 1-10 ms after each spike
(spike-triggered average), NEURON vs branch. BK has no NEURON counterpart (Allen has no BK); its
spike-triggered average is shown for scale. Writes results_fast_k/validation/{fast_k_vs_neuron.png, validation.json}.
"""
import json, os, sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb
ch, sp = sb.ch, sb.sp
NREF = "/tmp/nref_kv"
OUT = os.path.join(HERE, "results_fast_k", "validation"); os.makedirs(OUT, exist_ok=True)
RESET = -48.6   # Vm the model resets to (SK-pinned fits: c = -48.6 to -49.4 mV)
LAGS = np.arange(-3, 16)


def drive(V, S, M):
    st = ch.init_state(tf.constant(V[:, 0] - ch.JP))
    kv, bk = [], []
    for t in range(V.shape[1]):
        vb = S[:, t] * sb.V_SPIKE + (1 - S[:, t]) * (M[:, t] * (RESET - ch.JP) + (1 - M[:, t]) * (V[:, t] - ch.JP))
        st, cur = ch.step(st, tf.constant(vb, tf.float32))
        kv.append(((cur["Kv3_1"] - ch.kv31_steady(np.minimum(vb, sb.V_REF - ch.JP))) * (1 - S[:, t])).numpy())
        bk.append((cur["BK"] * (1 - S[:, t])).numpy())
    return np.stack(kv, 1) * ch.PA_PER_DENSITY, np.stack(bk, 1) * ch.PA_PER_DENSITY


def sta(X, S):
    rows, cols = np.nonzero(S[:, 20:-20]); cols = cols + 20
    return np.mean([X[r, c + LAGS] for r, c in zip(rows, cols)], 0)


res = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for j, pr in enumerate(sp.PROTOS):
    z = np.load(os.path.join(NREF, f"train100_{pr}.npz"))
    V = z["control_v"][:40]
    S, M = sp.gt_masks(V)
    kv, bk = drive(V, S, M)
    kv_n = z["control_ik_Kv3_1"][:40] * ch.PA_PER_DENSITY
    # the model passes on the current from the step after the spike: align the branch output one step
    # later (it acts on Vm from t+1), compare with NEURON's sampled current at the same lags
    a_n, a_b, a_bk = sta(kv_n, S), sta(np.roll(kv, 1, 1), S), sta(np.roll(bk, 1, 1), S)
    post = (LAGS >= 1) & (LAGS <= 10)
    res[pr] = dict(kv_neuron_pA_lag1to5=a_n[(LAGS >= 1) & (LAGS <= 5)].round(0).tolist(),
                   kv_branch_pA_lag1to5=a_b[(LAGS >= 1) & (LAGS <= 5)].round(0).tolist(),
                   kv_ratio_1to10ms=float(a_b[post].sum() / a_n[post].sum()),
                   bk_branch_pA_lag1to5=a_bk[(LAGS >= 1) & (LAGS <= 5)].round(0).tolist())
    ax[j].plot(LAGS, a_n, "k-o", ms=3, label="NEURON Kv3.1")
    ax[j].plot(LAGS, a_b, "-o", color="#e67e22", ms=3, label="Kv3.1 branch (s = 1)")
    ax[j].plot(LAGS, a_bk, "-o", color="#16a085", ms=3, label="BK branch (s = 1, no NEURON counterpart)")
    ax[j].set_xlim(-3, 15); ax[j].set_ylim(0, max(a_n[LAGS >= 1].max(), a_b[LAGS >= 1].max()) * 1.2)
    ax[j].set_title(f"{pr}: spike-triggered outward current"); ax[j].set_xlabel("ms from spike onset"); ax[j].grid(alpha=0.25)
ax[0].set_ylabel("pA (outward +)"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fast_k_vs_neuron.png"), dpi=130)
json.dump(res, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
