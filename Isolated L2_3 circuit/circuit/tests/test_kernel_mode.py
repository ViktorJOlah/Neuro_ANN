"""Kernel mode of the batched PC layer matches the evoked-fit implementation (synaptic_kernel.run_syn)."""
import os
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "Standard"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "evoked_synaptic", "code"))


def test_kernel_step_matches_run_syn():
    import standard_model as sm
    import synaptic_kernel as SK
    from circuit.pc_layer import BatchedStandard
    cell = sm.load()
    corr = dict(k0=0.9, c1=-0.05, tau1=5.0, c2=0.1, tau2=50.0)
    n, T = 3, 400
    rng = np.random.default_rng(0)
    t = np.arange(T)
    I = np.zeros((n, T), np.float32)
    for i in range(n):
        for ts in rng.uniform(0, T, 12):
            k = t - ts
            I[i] += np.where(k >= 0, 60 * (k / 5.5) * np.exp(1 - k / 5.5), 0)
    I += 10.0                     # stays subthreshold: near threshold float noise moves spikes by a step
    # far-away reversals: driving-force factor 1, so the layer sees I unchanged, as run_syn's i_ext does
    layer = BatchedStandard(cell, n, kernel=dict(corr=corr, E_e=1e7, E_i=-1e7, V_ref=0.0), jit=False)
    vm_b = np.stack([(layer.step_kernel(I[:, k], np.zeros(n)), layer.vm.numpy())[1] for k in range(T)], 1)
    K = SK.KernelLayer(1); K.on.assign(0.0)
    x0 = np.repeat(layer.x0.numpy()[None, None], n, 0).repeat(T, 1).astype(np.float32)
    V = np.full((n, T), -82.0, np.float32)
    Sm, M = sm.gt_masks(V)
    vm_r = SK.run_syn(cell, K, tf.constant(x0), tf.zeros((n, T, 2)), tf.constant(V), tf.constant(Sm), tf.constant(M),
                      tf.constant(1), False, i_ext=tf.constant(I), corr={k: tf.constant(v) for k, v in corr.items()})[0].numpy()
    assert np.max(np.abs(vm_b - vm_r)) < 0.05
