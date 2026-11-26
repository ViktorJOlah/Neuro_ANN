COMMENT
HCN (Ih) current of mouse L2/3 PCs as in Olah, Wu, Kaczmarek & Rowan (eLife 2025, 96002, Methods):
single activation gate, no inactivation.
  tau(v) = 25.46 * exp(-0.5 * ((v + 100.28) / 16.2)^2)   (ms)   [paper's equation]
  minf(v) = 1 / (1 + exp((v - vhalf) / k))
The paper prints minf with (v + 21.945)/6.85, which does not match its own Fig 2i (Cs+-sensitive
conductance half-activated near -106 mV). vhalf is therefore a RANGE parameter (default from Fig 2i)
and k = 6.85 from the paper. tauscale divides tau (1 = paper). ehcn -45 mV.
ENDCOMMENT
NEURON {
    SUFFIX Ih_Olah
    NONSPECIFIC_CURRENT ihcn
    RANGE gbar, g, ihcn, vhalf, k, tauscale, ehcn, minf, mtau
}
UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}
PARAMETER {
    gbar = 0.0001 (S/cm2)
    ehcn = -45 (mV)
    vhalf = -106 (mV)
    k = 6.85 (mV)
    tauscale = 1
}
ASSIGNED {
    v (mV)
    ihcn (mA/cm2)
    g (S/cm2)
    minf
    mtau (ms)
}
STATE { m }
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * m
    ihcn = g * (v - ehcn)
}
DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
}
INITIAL {
    rates(v)
    m = minf
}
PROCEDURE rates(v (mV)) {
    UNITSOFF
    minf = 1 / (1 + exp((v - vhalf) / k))
    mtau = 25.46 * exp(-0.5 * ((v + 100.28) / 16.2)^2) / tauscale
    if (mtau < 0.05) { mtau = 0.05 }
    UNITSON
}
