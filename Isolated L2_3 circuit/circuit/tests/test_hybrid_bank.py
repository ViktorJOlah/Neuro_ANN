"""PSCBank (the PCs' synaptic layer in the hybrid loop) against the closed-form alpha current."""
import numpy as np

from circuit.hybrid import PSCBank, RES
from circuit.synaptic_layer import TAU_SYN, alpha_current


def test_bank_matches_closed_form():
    rng = np.random.default_rng(1)
    n, T = 3, 60
    bank = PSCBank(n, ring=64)
    # arrivals: (time of PSC start on the 0.25 grid, target, weight, port)
    ev = [(rng.integers(4, 4 * (T - 10)) * RES, rng.integers(n), rng.normal(0, 20), rng.choice([1, 2])) for _ in range(40)]
    ev.sort()
    out = []
    for k in range(T):
        for a, trg, w, p in ev:
            step = int(round(a / RES)) - 1
            if 4 * k <= step < 4 * k + 4:
                bank.push_sparse(np.array([step]), np.array([trg]), np.array([w]), np.array([p]))
        out.append(bank.advance_ms())
    out = np.array(out)
    fine = np.arange(0, 4 * T + 1) * RES
    for c in range(n):
        sel = [e for e in ev if e[1] == c]
        I = alpha_current([e[0] for e in sel], [e[2] for e in sel], 0.0, [TAU_SYN[e[3]] for e in sel], fine)
        ref = (0.5 * I[0:-1:4] + I[1::4] + I[2::4] + I[3::4] + 0.5 * I[4::4]) / 4
        np.testing.assert_allclose(out[:, c], ref, atol=1e-6)
