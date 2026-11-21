"""Generate square / ramp / noise current-clamp simulations of the Allen Cell Types
perisomatic L2/3 PC model 488462783 (specimen 487661754, Cux2-CreERT2 L2/3 PC, VISp).

Protocols follow the Allen Cell Types electrophysiology white paper (Oct 2017, v.5,
"Electrophysiology Overview", Fig. 3 and Appendix: Stimulus Details):
  * Long Square: 1 s steps from -110 pA to rheobase + 160 pA.
  * Ramp: 25 pA/s, terminated after the neuron fires a short series of spikes.
  * Noise 1 & 2: pink noise, CV = 0.2, 3 s epochs riding on squares at 0.75, 1 and
    1.5 x rheobase.
Rheobase is the model's own (1 s step, found by bisection), since Allen defined
amplitudes relative to each cell's rheobase.

Each protocol gets 1000 simulations with amplitudes evenly spaced across the bounds.
Output: <out>/<protocol>.npy, float32 array (1000, T, 2): [..., 0] = injected current (pA),
[..., 1] = somatic Vm (mV, Allen recording convention = v_sim - junction_potential),
both sampled every 1 ms. Plus <protocol>_params.csv and metadata.json.

Usage: python generate_dataset.py --model <allen model dir with compiled modfiles> --out <dir>
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_SIM = int(os.environ.get("N_SIM", 1000))
PRE_MS = 500           # baseline before stimulus onset
SQ_DUR, SQ_POST = 1000, 500
RAMP_SLOPE = 0.025     # pA per ms (= 25 pA/s)
RAMP_POST = 1000
RAMP_MAX_SPIKES = 27   # most spikes Allen recorded on this cell's ramp sweeps (sweep 5)
NOISE_DUR, NOISE_POST = 3000, 1000
NOISE_CV = 0.2
NOISE_BAND = (1.0, 100.0)  # Hz; Allen does not state the band in the white paper (assumption)
NOISE_SCALES = (0.75, 1.5)  # x rheobase
SPIKE_THR = -20.0      # mV, recording convention, for spike counting on 1 ms samples

_runner = None


def _init_worker(model_dir, dt):
    global _runner
    from allen_cell import Runner
    _runner = Runner(model_dir, dt=dt)


def _simulate(task):
    proto, idx, i_pA = task
    v = _runner.run(i_pA, 1.0)[: len(i_pA)]
    return proto, idx, v.astype(np.float32)


def count_spikes(v):
    return int(np.sum((v[1:] > SPIKE_THR) & (v[:-1] <= SPIKE_THR)))


def pink_noise(n, seed, band=NOISE_BAND, fs=1000.0):
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    amp = np.zeros_like(f)
    m = (f >= band[0]) & (f <= band[1])
    amp[m] = 1.0 / np.sqrt(f[m])  # power ~ 1/f
    phase = rng.uniform(0, 2 * np.pi, len(f))
    x = np.fft.irfft(amp * np.exp(1j * phase), n)
    return (x - x.mean()) / x.std()


def calibrate(model_dir, dt):
    """Model rheobase (1 s step) and ramp spike amplitudes, using exact spike times."""
    from allen_cell import Runner
    from neuron import h
    r = Runner(model_dir, dt=dt)
    nc = h.NetCon(h.soma[0](0.5)._ref_v, None, sec=h.soma[0])
    nc.threshold = SPIKE_THR + r.info["junction_potential"]  # raw-voltage threshold
    st = h.Vector()
    nc.record(st)

    def step(a):
        T = PRE_MS + SQ_DUR + SQ_POST
        t = np.arange(T)
        r.run(np.where((t >= PRE_MS) & (t < PRE_MS + SQ_DUR), a, 0.0), 1.0)
        return len(st)

    lo, hi = 0.0, 300.0
    while hi - lo > 0.25:
        mid = (lo + hi) / 2
        if step(mid) > 0:
            hi = mid
        else:
            lo = mid
    rheo = hi
    T = PRE_MS + 30000
    t = np.arange(T)
    r.run(np.clip((t - PRE_MS) * RAMP_SLOPE, 0, None), 1.0)
    amps = (np.array(st) - PRE_MS) * RAMP_SLOPE
    return dict(rheobase_pA=float(rheo), rest_mV=float(r.rest_raw - r.info["junction_potential"]),
                ramp_spike_amplitudes_pA=[float(a) for a in amps[:RAMP_MAX_SPIKES]],
                **r.info)


def build_tasks(cal):
    rheo = cal["rheobase_pA"]
    tasks, params = [], {"square": [], "ramp": [], "noise": []}

    # Long square
    T = PRE_MS + SQ_DUR + SQ_POST
    t = np.arange(T)
    for k, a in enumerate(np.linspace(-110.0, rheo + 160.0, N_SIM)):
        i = np.where((t >= PRE_MS) & (t < PRE_MS + SQ_DUR), a, 0.0)
        tasks.append(("square", k, i))
        params["square"].append(dict(idx=k, amplitude_pA=a, onset_ms=PRE_MS, duration_ms=SQ_DUR))

    # Ramp: 25 pA/s from 0, terminated at an end amplitude between the model's first
    # ramp spike and its 27th ramp spike (Allen's ramps on this cell gave 14 and 27 spikes).
    sp = cal["ramp_spike_amplitudes_pA"]
    a_lo, a_hi = sp[0], sp[-1]
    T = int(np.ceil((PRE_MS + a_hi / RAMP_SLOPE + RAMP_POST) / 100.0) * 100)
    t = np.arange(T)
    for k, a in enumerate(np.linspace(a_lo, a_hi, N_SIM)):
        dur = a / RAMP_SLOPE
        i = np.where((t >= PRE_MS) & (t <= PRE_MS + dur), (t - PRE_MS) * RAMP_SLOPE, 0.0)
        tasks.append(("ramp", k, i))
        params["ramp"].append(dict(idx=k, end_amplitude_pA=a, slope_pA_per_s=RAMP_SLOPE * 1000,
                                   onset_ms=PRE_MS, duration_ms=dur))

    # Noise: one Allen-style epoch per simulation, pink noise (CV 0.2) on a square whose
    # offset spans 0.75-1.5 x rheobase; a fresh seed per simulation.
    T = PRE_MS + NOISE_DUR + NOISE_POST
    t = np.arange(T)
    for k, s in enumerate(np.linspace(*NOISE_SCALES, N_SIM)):
        off = s * rheo
        seed = 10000 + k
        i = np.zeros(T)
        on = (t >= PRE_MS) & (t < PRE_MS + NOISE_DUR)
        i[on] = off * (1.0 + NOISE_CV * pink_noise(int(on.sum()), seed))
        tasks.append(("noise", k, i))
        params["noise"].append(dict(idx=k, rheobase_multiple=s, offset_pA=off,
                                    noise_sd_pA=NOISE_CV * off, cv=NOISE_CV, seed=seed,
                                    onset_ms=PRE_MS, duration_ms=NOISE_DUR))
    return tasks, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--progress", default=None, help="file to write 'done/total' into")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    cal = calibrate(a.model, a.dt)
    print("calibration:", json.dumps({k: v for k, v in cal.items() if k != "ramp_spike_amplitudes_pA"}), flush=True)
    tasks, params = build_tasks(cal)
    cur = {p: np.stack([x[2] for x in tasks if x[0] == p]).astype(np.float32) for p in params}
    vm = {p: np.zeros_like(cur[p]) for p in params}

    total = len(tasks)
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs, initializer=_init_worker, initargs=(a.model, a.dt)) as pool:
        for proto, idx, v in pool.imap_unordered(_simulate, tasks, chunksize=4):
            vm[proto][idx] = v
            done += 1
            if a.progress:
                with open(a.progress, "w") as f:
                    f.write("%d/%d %.0f\n" % (done, total, time.time() - t0))

    summary = {}
    for p in params:
        arr = np.stack([cur[p], vm[p]], axis=-1)
        np.save(os.path.join(a.out, "%s.npy" % p), arr)
        for row, v in zip(params[p], vm[p]):
            row["n_spikes"] = count_spikes(v)
        with open(os.path.join(a.out, "%s_params.csv" % p), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(params[p][0].keys()))
            w.writeheader()
            w.writerows(params[p])
        ns = [r["n_spikes"] for r in params[p]]
        summary[p] = dict(shape=list(arr.shape), n_spikes_min=min(ns), n_spikes_max=max(ns),
                          n_with_spikes=int(sum(n > 0 for n in ns)))

    meta = dict(
        model=dict(allen_neuronal_model_id=488462783, specimen_id=487661754,
                   name="Biophysical - perisomatic_Cux2-CreERT2;Ai14-211772.05.02.01",
                   source="Neuro_ANN repo: multicompartmental models/L2_3 PC/NEURON/orig_file (Allen download)",
                   build="replicates allensdk.model.biophysical.utils.Utils (perisomatic): 60 um axon stub"),
        simulation=dict(dt_ms=a.dt, sample_ms=1.0, celsius=34.0,
                        init="steady state at zero current (3 s, large-dt implicit settle) from v_init=e_pas",
                        current_interpolation="current defined at 1 ms samples, played with linear interpolation",
                        injection_site="soma[0](0.5)"),
        array_layout="(n_sim, n_ms, 2): [...,0]=injected current pA, [...,1]=somatic Vm mV",
        vm_convention="Vm = v_sim - junction_potential (+14 mV), same as allensdk output / Allen recordings",
        calibration=cal, protocols=summary,
        bounds_source="Allen Cell Types Database, Technical White Paper: Electrophysiology (Oct 2017 v.5), Fig. 3 + Appendix",
        noise_band_Hz=NOISE_BAND, noise_band_note="assumed; band not stated in the white paper",
        wall_time_s=time.time() - t0)
    with open(os.path.join(a.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done in %.0f s" % (time.time() - t0), json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
