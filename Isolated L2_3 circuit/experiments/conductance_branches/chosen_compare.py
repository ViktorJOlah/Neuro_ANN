"""The Chosen One (CNN-LSTM): onset doublets and summary metrics across fit variants.
Run from "Isolated L2_3 circuit/":  python experiments/conductance_branches/chosen_compare.py DIR [DIR ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb
s = sb.fs.load_split()[("square", "test")]
amp = s["I"][:, 1000]; S, _ = sb.sp.gt_masks(s["V"]); S = S > 0.5
strong = amp >= 180


def first_isi(sp):
    t = np.flatnonzero(sp[500:1500]); return t[1] - t[0] if len(t) > 1 else np.nan


gt = np.array([first_isi(S[i]) for i in range(len(S))])
print("NEURON: first ISI on strong steps (>=180 pA) median %.0f ms, min %.0f ms, n %d" % (np.nanmedian(gt[strong]), np.nanmin(gt[strong]), strong.sum()))
for d in sys.argv[1:]:
    r = json.load(open(os.path.join(HERE, d, "cnnlstm", "metrics.json"))); e = r["eval"]
    z = np.load(os.path.join(HERE, d, "cnnlstm", "predictions.npz")); sp = z["square_cont_spikes"].astype(bool)
    fi = np.array([first_isi(sp[i]) for i in range(len(sp))])
    t = r["trained"]
    print(f"{d}: first ISI strong median {np.nanmedian(fi[strong]):.0f} ms, doublets (<20 ms) {np.sum(fi[strong] < 20)}/{strong.sum()}, "
          f"all steps {np.sum(fi < 20)} | Γ cont " + ", ".join(f"{p} {e[p]['cont']['gamma']:.3f} (bias {e[p]['cont']['count_bias']:+.2f})" for p in e)
          + f" | spike-free {r['spike_free_cont_rmse_mV']} | V_th {t['V_th']:.1f} c {t['c_mV']:.1f} th_k {t.get('th_k', 0):.2f} loss {r['train_spike_loss']:.3f}")
