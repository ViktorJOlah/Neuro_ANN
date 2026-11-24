COMMENT
NMDA receptor synapse: double-exponential conductance with Jahr & Stevens (1990) Mg2+ block.
Event weight = peak conductance (uS) of the unblocked double exponential. Added for the evoked
synaptic feasibility test (experiments/evoked_synaptic).
ENDCOMMENT
NEURON {
    POINT_PROCESS NMDA_JS
    RANGE tau1, tau2, e, i, g, mg
    NONSPECIFIC_CURRENT i
}
UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}
PARAMETER {
    tau1 = 2 (ms)
    tau2 = 50 (ms)
    e = 0 (mV)
    mg = 1 (mM)
}
ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    factor
}
STATE { A (uS) B (uS) }
INITIAL {
    LOCAL tp
    A = 0
    B = 0
    tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1/factor
}
BREAKPOINT {
    SOLVE state METHOD cnexp
    g = (B - A) / (1 + exp(0.062 * (-v)) * (mg / 3.57))
    i = g * (v - e)
}
DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
}
NET_RECEIVE(weight (uS)) {
    A = A + weight*factor
    B = B + weight*factor
}
