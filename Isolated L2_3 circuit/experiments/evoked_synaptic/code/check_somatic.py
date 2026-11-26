"""Somatic-injection check: with no evoked events the kernel layer adds exactly zero current, so the Standard model's
response to somatic current must be unchanged. Runs held-out square / ramp / noise traces of the Standard's own cell
through S.run and through synaptic_kernel.run_syn (fitted kernel loaded, zero events) and compares.

    python check_somatic.py <kernel results dir>   -> somatic_check.json there
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

import fit_input_transform as F
import synaptic_kernel as SK
import standard_model as sm

SIMS = os.environ.get("ALLEN_SIMS", "/mnt/project-files/allen_l23_sims")
SEED = 450


def main(d):
    kj = json.load(open(os.path.join(d, "kernel.json")))
    S = F.load_somatic("487661754")
    K = SK.KernelLayer(kj["ncomp"], kj["nmda"], kj["delay"], kj["init"], kj.get("signed", False))
    run_k = tf.function(lambda x, ev, V, Sm, M: SK.run_syn(S, K, x, ev, V, Sm, M, tf.constant(SEED), False))
    out = {}
    for prot in ("square", "ramp", "noise"):
        a = np.load(os.path.join(SIMS, f"{prot}.npy"), mmap_mode="r")
        idx = np.linspace(0, len(a) - 1, 8).astype(int)          # spread over the protocol's parameter range
        I, V = np.asarray(a[idx, :, 0], np.float32), np.asarray(a[idx, :, 1], np.float32)
        x = S.current(I)
        Sm, M = sm.gt_masks(V)
        v0, s0 = S.run(tf.constant(x), tf.constant(V), tf.constant(Sm), tf.constant(M), tf.constant(SEED), tf.constant(False))
        ev = tf.zeros(I.shape + (2,), tf.float32)
        v1, s1 = run_k(tf.constant(x), ev, tf.constant(V), tf.constant(Sm), tf.constant(M))
        out[prot] = dict(n=len(idx), max_abs_dVm=float(np.max(np.abs(v0.numpy() - v1.numpy()))),
                         spikes_standard=int(s0.numpy().sum()), spikes_with_layer=int(s1.numpy().sum()),
                         spike_mismatch=int(np.sum(s0.numpy() != s1.numpy())))
        print(prot, out[prot], flush=True)
    json.dump(out, open(os.path.join(d, "somatic_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
