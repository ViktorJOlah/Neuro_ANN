"""Somatic channel kinetics of Allen model 488462783, copied from its mod files, for a 1 ms step.

All voltages here are NEURON's simulation voltage (mV). The dataset uses the Allen recording
convention, Vm = v_sim - junction_potential = v_sim + 14 mV, so callers pass Vm - 14.
celsius = 34 (fit_parameters.json). Currents are densities in mA/cm2, NEURON sign convention
(inward negative). Gates use the exact exponential update over dt for a frozen voltage
(the cnexp solution); gates whose time constant is well under 1 ms are taken at steady state.
"""
import numpy as np
import tensorflow as tf

JP = 14.0                       # mV; Vm_recording = v_sim + JP
CELSIUS = 34.0
ENA, EK = 53.0, -107.0          # fit_parameters.json conditions
CAO, MIN_CAI = 2.0, 1e-4        # NEURON default cao (mM); CaDynamics minCai
FARADAY = 96485.309
RT_2F = 8.314462 * (273.15 + CELSIUS) / (2 * FARADAY) * 1000.0   # mV
SOMA_AREA_UM2 = 516.4723901317091   # soma[0](0.5).area() in the built model
PA_PER_DENSITY = SOMA_AREA_UM2 * 10.0  # mA/cm2 * um2 -> pA
# fitted somatic values (fit_parameters.json genome)
G = dict(NaTs=0.6838817113482546, Nap=1.04318930264212e-06, SK=0.0009437495816197669,
         Ca_HVA=5.06464725789868e-05, Ca_LVA=0.0019267653289255383)
CA_GAMMA, CA_DECAY, CA_DEPTH = 0.005385368415172307, 902.5077593614815, 0.1


def vtrap(x, y):
    small = tf.abs(x / y) < 1e-6
    safe = tf.where(small, tf.ones_like(x), x)
    return tf.where(small, y * (1 - x / y / 2), safe / (tf.exp(safe / y) - 1))


def relax(x, xinf, tau, dt=1.0):
    return xinf + (x - xinf) * tf.exp(-dt / tau)


# ---- NaTs (transient Na): m instantaneous (tau_m < 0.15 ms at 34 C), h exact
def nats_rates(v):
    qt = 2.3 ** ((CELSIUS - 23) / 10)
    ma, mb = 0.182 * vtrap(-(v + 40.0), 6.0), 0.124 * vtrap(v + 40.0, 6.0)
    ha, hb = 0.015 * vtrap(v + 66.0, 6.0), 0.015 * vtrap(-(v + 66.0), 6.0)
    return ma / (ma + mb), ha / (ha + hb), 1.0 / (ha + hb) / qt


# ---- Nap (persistent Na): m instantaneous by construction, h exact
def nap_rates(v):
    qt = 2.3 ** ((CELSIUS - 21) / 10)
    minf = 1.0 / (1 + tf.exp((v + 52.6) / -4.6))
    hinf = 1.0 / (1 + tf.exp((v + 48.8) / 10))
    ha, hb = 2.88e-6 * vtrap(v + 17, 4.63), 6.94e-6 * vtrap(-(v + 64.4), 2.63)
    return minf, hinf, 1.0 / (ha + hb) / qt


# ---- Ca_HVA: m tau < 1 ms around and above threshold, taken exact anyway; h exact (no q10 in mod)
def cahva_rates(v):
    ma, mb = 0.055 * vtrap(-27 - v, 3.8), 0.94 * tf.exp((-75 - v) / 17)
    ha, hb = 0.000457 * tf.exp((-13 - v) / 50), 0.0065 / (tf.exp((-v - 15) / 28) + 1)
    return ma / (ma + mb), 1.0 / (ma + mb), ha / (ha + hb), 1.0 / (ha + hb)


# ---- Ca_LVA: both gates exact (mod shifts v by +10 mV)
def calva_rates(v):
    qt = 2.3 ** ((CELSIUS - 21) / 10)
    w = v + 10.0
    minf = 1.0 / (1 + tf.exp((w + 30.0) / -6))
    mtau = (5.0 + 20.0 / (1 + tf.exp((w + 25.0) / 5))) / qt
    hinf = 1.0 / (1 + tf.exp((w + 80.0) / 6.4))
    htau = (20.0 + 50.0 / (1 + tf.exp((w + 40.0) / 7))) / qt
    return minf, mtau, hinf, htau


def sk_zinf(cai):
    cai = tf.where(cai < 1e-7, cai + 1e-7, cai)
    return 1.0 / (1 + (0.00043 / cai) ** 4.8)


def eca(cai):
    return RT_2F * tf.math.log(CAO / cai)


def init_state(v):
    """Steady-state gates at voltage v (n,), cai at minCai."""
    _, h_t, _ = nats_rates(v)
    _, h_p, _ = nap_rates(v)
    m_h, _, h_h, _ = cahva_rates(v)
    m_l, _, h_l, _ = calva_rates(v)
    cai = tf.fill(tf.shape(v), MIN_CAI)
    return dict(h_t=h_t, h_p=h_p, m_h=m_h, h_h=h_h, m_l=m_l, h_l=h_l, cai=cai, z=sk_zinf(cai))


def step(st, v, ca_scale=1.0, dt=1.0):
    """Advance the gates by dt at frozen voltage v (sim mV), return new state and the 1 ms
    currents (mA/cm2): NaTs, Nap, SK, and total Ca current (Ca enters only through cai)."""
    m_t, hinf_t, htau_t = nats_rates(v)
    _, hinf_p, htau_p = nap_rates(v)
    minf_h, mtau_h, hinf_h, htau_h = cahva_rates(v)
    minf_l, mtau_l, hinf_l, htau_l = calva_rates(v)
    h_t = relax(st["h_t"], hinf_t, htau_t, dt)
    h_p = relax(st["h_p"], hinf_p, htau_p, dt)
    m_h, h_h = relax(st["m_h"], minf_h, mtau_h, dt), relax(st["h_h"], hinf_h, htau_h, dt)
    m_l, h_l = relax(st["m_l"], minf_l, mtau_l, dt), relax(st["h_l"], hinf_l, htau_l, dt)
    minf_p = nap_rates(v)[0]
    i_nats = G["NaTs"] * m_t ** 3 * h_t * (v - ENA)
    i_nap = G["Nap"] * minf_p * h_p * (v - ENA)
    e_ca = eca(st["cai"])
    i_ca = G["Ca_HVA"] * m_h ** 2 * h_h * (v - e_ca) + G["Ca_LVA"] * m_l ** 2 * h_l * (v - e_ca)
    # CaDynamics: cai' = -10000 * ica * gamma / (2 F depth) - (cai - minCai) / decay   (exact for frozen ica)
    influx = -10000.0 * ca_scale * i_ca * CA_GAMMA / (2 * FARADAY * CA_DEPTH)
    k = np.exp(-dt / CA_DECAY)
    cai = MIN_CAI + influx * CA_DECAY + (st["cai"] - MIN_CAI - influx * CA_DECAY) * k
    cai = tf.maximum(cai, 1e-7)
    z = relax(st["z"], sk_zinf(cai), 1.0, dt)
    i_sk = G["SK"] * z * (v - EK)
    return dict(h_t=h_t, h_p=h_p, m_h=m_h, h_h=h_h, m_l=m_l, h_l=h_l, cai=cai, z=z), \
        dict(NaTs=i_nats, Nap=i_nap, SK=i_sk, Ca=i_ca)


def na_steady(v):
    """Steady-state NaTs + Nap current density at voltage v (sim mV)."""
    m_t, hinf_t, _ = nats_rates(v)
    minf_p, hinf_p, _ = nap_rates(v)
    return G["NaTs"] * m_t ** 3 * hinf_t * (v - ENA) + G["Nap"] * minf_p * hinf_p * (v - ENA)
