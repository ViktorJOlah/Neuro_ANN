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
         Ca_HVA=5.06464725789868e-05, Ca_LVA=0.0019267653289255383, Kv3_1=0.07834341196986132,
         BK=0.01)   # BK is not in the Allen model: a generic somatic density, scaled by the fit
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


# ---- Kv3_1 (fast delayed rectifier, Allen mod file, vshift 0): m exact; reported current uses the
# step-averaged gate, because m relaxes with tau 1-4 ms, i.e. within the 1 ms step after a spike
def kv31_rates(v):
    minf = 1.0 / (1 + tf.exp((v - 18.7) / -9.7))
    mtau = 0.2 * 20.0 / (1 + tf.exp((v + 46.56) / -44.14))
    return minf, mtau


# ---- BK (generic; the Allen model has none): voltage- and Ca-gated, Cox-Aldrich-like shift of the
# half-activation voltage with nanodomain Ca (about -80 mV per decade of Ca), tau 1 ms. The
# nanodomain Ca is driven by the same Ca current as the bulk pool, through a thin shell with no
# buffering (CA_ND_GAMMA), and clears in about 1 ms, so it only exists during and just after a spike.
BK_K, BK_TAU, CA_ND_TAU, CA_ND_GAMMA = 12.0, 1.0, 1.0, 10.0   # gamma 10 ~ a 10 nm shell: ~10 uM per spike
def bk_minf(v, ca_nd):
    ca_um = tf.maximum(ca_nd * 1000.0, 1e-3)
    vh = 40.0 - 80.0 * tf.math.log(ca_um) / np.log(10.0)
    return 1.0 / (1 + tf.exp(-(v - vh) / BK_K))


def mean_relax(x, xinf, tau, dt=1.0):
    """average over the step of the exact exponential relaxation"""
    return xinf + (x - xinf) * tau / dt * (1.0 - tf.exp(-dt / tau))


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
    ca_nd = tf.zeros_like(v)
    return dict(h_t=h_t, h_p=h_p, m_h=m_h, h_h=h_h, m_l=m_l, h_l=h_l, cai=cai, z=sk_zinf(cai),
                m_kv=kv31_rates(v)[0], ca_nd=ca_nd, m_bk=bk_minf(v, ca_nd))


def step(st, v, ca_scale=1.0, dt=1.0):
    """Advance the gates by dt at frozen voltage v (sim mV), return new state and the 1 ms
    currents (mA/cm2): NaTs, Nap, SK, total Ca current (Ca enters only through cai), and Kv3_1 and BK
    (step-averaged gates)."""
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
    minf_kv, mtau_kv = kv31_rates(v)
    m_kv = relax(st["m_kv"], minf_kv, mtau_kv, dt)
    i_kv = G["Kv3_1"] * mean_relax(st["m_kv"], minf_kv, mtau_kv, dt) * (v - EK)
    infl_nd = -10000.0 * ca_scale * i_ca * CA_ND_GAMMA / (2 * FARADAY * CA_DEPTH)
    ca_nd = tf.maximum(infl_nd * CA_ND_TAU + (st["ca_nd"] - infl_nd * CA_ND_TAU) * np.exp(-dt / CA_ND_TAU), 0.0)
    minf_bk = bk_minf(v, 0.5 * (st["ca_nd"] + ca_nd))
    m_bk = relax(st["m_bk"], minf_bk, BK_TAU, dt)
    i_bk = G["BK"] * mean_relax(st["m_bk"], minf_bk, BK_TAU, dt) * (v - EK)
    return dict(h_t=h_t, h_p=h_p, m_h=m_h, h_h=h_h, m_l=m_l, h_l=h_l, cai=cai, z=z, m_kv=m_kv, ca_nd=ca_nd, m_bk=m_bk), \
        dict(NaTs=i_nats, Nap=i_nap, SK=i_sk, Ca=i_ca, Kv3_1=i_kv, BK=i_bk)


def kv31_steady(v):
    return G["Kv3_1"] * kv31_rates(v)[0] * (v - EK)


def na_steady(v):
    """Steady-state NaTs + Nap current density at voltage v (sim mV)."""
    m_t, hinf_t, _ = nats_rates(v)
    minf_p, hinf_p, _ = nap_rates(v)
    return G["NaTs"] * m_t ** 3 * hinf_t * (v - ENA) + G["Nap"] * minf_p * hinf_p * (v - ENA)
