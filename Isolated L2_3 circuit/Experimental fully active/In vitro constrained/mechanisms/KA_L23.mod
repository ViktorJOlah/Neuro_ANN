COMMENT
Dendritic A-type K+ current (Kv4-like) for the Experimental fully active L2/3 model.
Single activation gate m and inactivation gate h, gk = gbar * m * h (form of the Migliore et al.
1999 / Hoffman 1997 dendritic A-current). Activation V1/2 -25 mV, slope 8 (small window current at rest),
so the channel acts during subthreshold synaptic depolarisation (Biro 2018: Kv4 block enlarges
single-spine responses in L2/3) and as a brake on dendritic Ca2+ electrogenesis (Gooch 2022:
4-AP unmasks Ca2+ spikes in human L2/3). Inactivation V1/2 -74 mV (Kv4). Rates at 34 C.
Parameters vhalfm, km, vhalfh, kh, taum, tauh0, tauh_slope are RANGE so they can be fitted.
ENDCOMMENT
NEURON {
    SUFFIX KA_L23
    USEION k READ ek WRITE ik
    RANGE gbar, g, ik, vhalfm, km, vhalfh, kh, taum, tauh0, tauh_slope
}
UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}
PARAMETER {
    gbar = 0.01 (S/cm2)
    vhalfm = -25 (mV)
    km = 8 (mV)
    vhalfh = -74 (mV)
    kh = 7 (mV)
    taum = 0.5 (ms)
    tauh0 = 6 (ms)
    tauh_slope = 0.26 (ms/mV)
}
ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    g (S/cm2)
    minf
    hinf
    htau (ms)
}
STATE { m h }
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * m * h
    ik = g * (v - ek)
}
DERIVATIVE states {
    rates(v)
    m' = (minf - m) / taum
    h' = (hinf - h) / htau
}
INITIAL {
    rates(v)
    m = minf
    h = hinf
}
PROCEDURE rates(v (mV)) {
    UNITSOFF
    minf = 1 / (1 + exp(-(v - vhalfm) / km))
    hinf = 1 / (1 + exp((v - vhalfh) / kh))
    htau = tauh0
    if (v > -50) { htau = tauh0 + tauh_slope * (v + 50) }
    UNITSON
}
