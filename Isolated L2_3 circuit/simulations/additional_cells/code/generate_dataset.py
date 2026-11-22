"""Square / ramp / noise current-clamp simulations of additional Allen all-active L2/3 PC models.

Protocols, bounds, stimulus construction, 1 ms sampling, Vm convention and spike counting are
taken unchanged from ../../code/generate_dataset.py (the first cell, model 488462783), so the
datasets are directly comparable. Only the cell builder (all-active, allactive_cell.py), the
number of simulations (100 per protocol) and the calibration search ranges differ.

Bounds per cell (each cell's own model rheobase, as Allen defines amplitudes per cell):
  * square: -110 pA ... rheobase + 160 pA, 1 s step, 500 ms before, 500 ms after
  * ramp:   25 pA/s, ended between the model's 1st and 27th ramp spike (or its last spike if
            the model stops firing earlier, e.g. depolarisation block)
  * noise:  pink noise (1-100 Hz), CV 0.2, on a 3 s square at 0.75-1.5 x rheobase, seed 10000+k

Usage:
  python generate_dataset.py --cell ../models/521409057 --mech ../models --out ../data/521409057
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

# first cell's generator (protocols + constants); same file name, so load it by path
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "gen_first_cell", os.path.join(HERE, "..", "..", "code", "generate_dataset.py"))
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

N_SIM = int(os.environ.get("N_SIM", 100))
g0.N_SIM = N_SIM
RHEO_MAX = 3000.0        # pA, bisection upper bound
RAMP_CAL_MAX_MS = 80000  # calibration ramp length (up to 2 nA)

_runner = None


def _init_worker(cell, mech, dt):
    global _runner
    from allactive_cell import Runner
    _runner = Runner(cell, mech, dt=dt)


def _simulate(task):
    proto, idx, i_pA = task
    v = _runner.run(i_pA, 1.0)[: len(i_pA)]
    return proto, idx, v.astype(np.float32)


def calibrate(cell, mech, dt):
    from allactive_cell import Runner
    from neuron import h
    r = Runner(cell, mech, dt=dt)
    nc = h.NetCon(h.soma[0](0.5)._ref_v, None, sec=h.soma[0])
    nc.threshold = g0.SPIKE_THR + r.info["junction_potential"]
    st = h.Vector()
    nc.record(st)

    def step(a):
        T = g0.PRE_MS + g0.SQ_DUR + g0.SQ_POST
        t = np.arange(T)
        r.run(np.where((t >= g0.PRE_MS) & (t < g0.PRE_MS + g0.SQ_DUR), a, 0.0), 1.0)
        return len(st)

    lo, hi = 0.0, RHEO_MAX
    if step(hi) == 0:
        raise RuntimeError("no spikes at %g pA" % hi)
    while hi - lo > 0.25:
        mid = (lo + hi) / 2
        if step(mid) > 0:
            hi = mid
        else:
            lo = mid
    rheo = hi
    t = np.arange(g0.PRE_MS + RAMP_CAL_MAX_MS)
    v = r.run(np.clip((t - g0.PRE_MS) * g0.RAMP_SLOPE, 0, None), 1.0)
    amps = (np.array(st) - g0.PRE_MS) * g0.RAMP_SLOPE
    if len(amps) == 0:
        raise RuntimeError("no ramp spikes")
    return dict(rheobase_pA=float(rheo), rest_mV=float(r.rest_raw - r.info["junction_potential"]),
                ramp_spike_amplitudes_pA=[float(a) for a in amps[: g0.RAMP_MAX_SPIKES]],
                ramp_total_spikes_to_2nA=int(len(amps)),
                ramp_last_spike_pA=float(amps[-1]), **r.info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--mech", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--progress", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    cal = calibrate(a.cell, a.mech, a.dt)
    print("calibration:", json.dumps({k: v for k, v in cal.items() if k != "ramp_spike_amplitudes_pA"}), flush=True)
    json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w"), indent=2)
    tasks, params = g0.build_tasks(cal)
    cur = {p: np.stack([x[2] for x in tasks if x[0] == p]).astype(np.float32) for p in params}
    vm = {p: np.zeros_like(cur[p]) for p in params}
    tasks.sort(key=lambda x: -len(x[2]))  # longest first for load balance

    done, total = 0, len(tasks)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs, initializer=_init_worker, initargs=(a.cell, a.mech, a.dt)) as pool:
        for proto, idx, v in pool.imap_unordered(_simulate, tasks, chunksize=1):
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
            row["n_spikes"] = g0.count_spikes(v)
        with open(os.path.join(a.out, "%s_params.csv" % p), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(params[p][0].keys()))
            w.writeheader()
            w.writerows(params[p])
        ns = [r["n_spikes"] for r in params[p]]
        summary[p] = dict(shape=list(arr.shape), n_spikes_min=min(ns), n_spikes_max=max(ns),
                          n_with_spikes=int(sum(n > 0 for n in ns)))
    src = json.load(open(os.path.join(a.cell, "source.json")))
    meta = dict(
        model=dict(src, type="Allen all-active (Nandi et al. 2022 hall-of-fame #0)",
                   build="replicates allensdk AllActiveUtils, axon_type=stub (2 x 30 um, 1 um diameter, at soma(0.5))"),
        simulation=dict(dt_ms=a.dt, sample_ms=1.0, celsius=34.0,
                        init="steady state at zero current (3 s, large-dt implicit settle) from v_init",
                        current_interpolation="current defined at 1 ms samples, played with linear interpolation",
                        injection_site="soma[0](0.5)"),
        array_layout="(n_sim, n_ms, 2): [...,0]=injected current pA, [...,1]=somatic Vm mV",
        vm_convention="Vm = v_sim - junction_potential (+14 mV), same as allensdk output / Allen recordings",
        calibration=cal, protocols=summary, n_sim_per_protocol=N_SIM,
        protocol_source="../../code/generate_dataset.py (identical protocols to the first cell)",
        noise_band_Hz=g0.NOISE_BAND, wall_time_s=time.time() - t0)
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=2)
    print("done in %.0f s" % (time.time() - t0), json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
