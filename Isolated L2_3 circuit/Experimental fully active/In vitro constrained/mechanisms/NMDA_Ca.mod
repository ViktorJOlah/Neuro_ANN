COMMENT
NMDA receptor synapse for the Experimental fully active L2/3 model.
Double-exponential conductance (tau1 rise, tau2 decay) with the Jahr & Stevens (1990) Mg2+ block,
as NMDA_JS.mod (experiments/evoked_synaptic). A fraction cafrac of the current is carried by Ca2+
and written to ica, so NMDA Ca2+ entry reaches cai (CaDynamics) like VGCC entry does. With a fixed
reversal of 0 mV this is an approximation of the GHK Ca2+ flux (fine for imaging-like readouts).
Event weight = peak conductance (uS) of the unblocked double exponential.
ENDCOMMENT
NEURON {
    POINT_PROCESS NMDA_Ca
    USEION ca WRITE ica
    RANGE tau1, tau2, e, i, g, mg, cafrac, inmda, gamma_mg, eta_mg, vshift
    NONSPECIFIC_CURRENT inon
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
    cafrac = 0.1
    gamma_mg = 0.062 (/mV)
    eta_mg = 3.57 (mM)
    vshift = 0 (mV)   : shifts the Mg2+-unblock curve (Olah 2025 fitted the NMDA activation voltage)
}
ASSIGNED {
    v (mV)
    i (nA)
    inmda (nA)
    inon (nA)
    ica (nA)
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
    g = (B - A) / (1 + exp(gamma_mg * (vshift - v)) * (mg / eta_mg))
    inmda = g * (v - e)
    i = inmda
    ica = cafrac * inmda
    inon = (1 - cafrac) * inmda
}
DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
}
NET_RECEIVE(weight (uS)) {
    A = A + weight*factor
    B = B + weight*factor
}
