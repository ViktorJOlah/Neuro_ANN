"""The batched per-step Standard layer must reproduce StandardCell.simulate exactly.

Needs the Standard model folder: set STANDARD_DIR, or keep it at "Isolated L2_3 circuit/Standard".
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STD = os.environ.get("STANDARD_DIR", os.path.join(HERE, "..", "..", "Standard"))
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(STD, "standard_model.py")),
                                reason="Standard model folder not found (set STANDARD_DIR)")


def test_step_matches_simulate():
    sys.path.insert(0, os.path.abspath(STD))
    import standard_model as sm
    from circuit.pc_layer import BatchedStandard
    cell = sm.load(STD)
    rng = np.random.default_rng(0)
    T, n = 400, 6
    I = 60 + 120 * rng.standard_normal((n, T)).cumsum(1) / np.sqrt(np.arange(1, T + 1))
    I = np.clip(I, -100, 400).astype(np.float32)
    _, ref = cell.simulate(I)
    b = BatchedStandard(cell, n, I_first=I[:, 0])
    got = np.stack([b.step(I[:, t]) for t in range(T)], 1)
    assert ref.sum() > 5
    np.testing.assert_array_equal(got, ref)
