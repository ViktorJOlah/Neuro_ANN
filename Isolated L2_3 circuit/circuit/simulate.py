"""Run the full V1 network and the isolated L2/3 circuit with BMTK PointNet (NEST).

Three steps, each a sub-command:

  full      run the original Allen config (or any BMTK PointNet config) and save all V1 spikes.
  streams   turn full-network spikes into the replay stream for L2/3: spikes of every external
            V1 source (population ``v1ext``), plus the L2/3 reference spikes re-indexed to the
            isolated circuit. Works on our own full run or on spikes that Allen distributes.
  isolated  write and run the config of the isolated circuit: the L2/3 network from
            ``extract_l23``, driven by the replayed v1ext stream and the same LGN and bkg spike
            files as the full run.

Each run happens in a fresh process because NEST cannot be reset cleanly inside BMTK.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np

from . import sonata_io as sio


# --------------------------------------------------------------------------------------------
def resolved_config(config_path: str) -> dict:
    """Load a BMTK config with the manifest substituted, as a plain dict."""
    from bmtk.simulator.core.simulation_config import SimulationConfig
    cfg = SimulationConfig.from_json(config_path)
    return json.loads(json.dumps({k: cfg[k] for k in cfg.keys()}, default=str))


def run_bmtk(config_path: str, threads: int = 1) -> float:
    """Run a PointNet config in a subprocess; returns wall-clock seconds."""
    code = (
        "import sys\n"
        "from bmtk.simulator import pointnet\n"
        "conf = pointnet.Config.from_json(sys.argv[1])\n"
        f"conf['run']['n_thread'] = {int(threads)}\n"
        "conf.build_env()\n"
        "net = pointnet.PointNetwork.from_config(conf)\n"
        "sim = pointnet.PointSimulator.from_config(conf, net)\n"
        "sim.run()\n"
    )
    t0 = time.time()
    subprocess.run([sys.executable, "-c", code, os.path.abspath(config_path)], check=True)
    return time.time() - t0


# --------------------------------------------------------------------------------------------
def run_full(config_path: str, out_dir: str, tstop: float | None = None, threads: int = 1) -> str:
    cfg = resolved_config(config_path)
    cfg.pop("manifest", None)
    cfg.setdefault("output", {})
    cfg["output"].update({"output_dir": os.path.abspath(out_dir), "spikes_file": "spikes.h5",
                          "overwrite_output_dir": True, "log_file": "log.txt"})
    cfg.pop("reports", None)  # membrane reports of 230k cells are not needed here
    if tstop is not None:
        cfg["run"]["tstop"] = float(tstop)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_full.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    wall = run_bmtk(path, threads)
    _write_timing(out_dir, wall, cfg["run"]["tstop"])
    return os.path.join(out_dir, "spikes.h5")


def make_streams(full_spikes: str, extract_dir: str, out_dir: str, v1_population: str | None = None) -> dict:
    """Split full-network V1 spikes into the v1ext replay stream and the L2/3 reference."""
    with h5py.File(os.path.join(extract_dir, "id_maps.h5"), "r") as f:
        l23_to_v1 = f["l23_to_v1"][()]
        ext_to_v1 = f["v1ext_to_v1"][()]
        v1_population = v1_population or f.attrs.get("v1_population", "v1")
        if isinstance(v1_population, bytes):
            v1_population = v1_population.decode()
    found = False
    for pop, ids, ts in sio.iter_spikes(full_spikes):
        if pop != v1_population:
            continue
        found = True
        n = int(max(ids.max(initial=0), l23_to_v1.max(initial=0), ext_to_v1.max(initial=0))) + 1
        to_ext = np.full(n, -1, dtype=np.int64)
        to_ext[ext_to_v1] = np.arange(len(ext_to_v1))
        to_l23 = np.full(n, -1, dtype=np.int64)
        to_l23[l23_to_v1] = np.arange(len(l23_to_v1))
        e, l = to_ext[ids], to_l23[ids]
        sio.write_spikes(os.path.join(out_dir, "v1ext_spikes.h5"), "v1ext", e[e >= 0], ts[e >= 0])
        sio.write_spikes(os.path.join(out_dir, "l23_reference_spikes.h5"), "l23", l[l >= 0], ts[l >= 0])
        info = {"source_spikes": os.path.abspath(full_spikes), "v1ext_spikes": int((e >= 0).sum()),
                "l23_reference_spikes": int((l >= 0).sum()), "n_v1ext": int(len(ext_to_v1)),
                "n_l23": int(len(l23_to_v1))}
    if not found:
        raise ValueError(f"no spikes for population {v1_population!r} in {full_spikes}")
    with open(os.path.join(out_dir, "streams.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def poisson_streams(v1ext_spikes: str, out_path: str, t0: float, t1: float, seed: int = 0) -> dict:
    """Control stream: each v1ext source fires as a homogeneous Poisson process at its own mean rate.

    Keeps every source's rate but removes timing, correlations and stimulus locking, which
    shows how much of the L2/3 response depends on the temporal structure of the inputs.
    """
    rng = np.random.default_rng(seed)
    (pop, ids, ts), = [x for x in sio.iter_spikes(v1ext_spikes) if x[0] == "v1ext"]
    m = (ts >= t0) & (ts < t1)
    n = int(ids.max(initial=-1)) + 1
    counts = np.bincount(ids[m], minlength=n)
    new_counts = rng.poisson(counts)
    new_ids = np.repeat(np.arange(n), new_counts)
    new_ts = rng.uniform(t0, t1, new_ids.size)
    sio.write_spikes(out_path, "v1ext", new_ids, new_ts)
    return {"n_sources": n, "spikes_original": int(m.sum()), "spikes_poisson": int(new_ids.size)}


def isolated_config(full_config: str, extract_dir: str, v1ext_spikes: str, out_dir: str,
                    tstop: float | None = None) -> str:
    """Write the BMTK config of the isolated L2/3 circuit, reusing the full run's settings."""
    cfg = resolved_config(full_config)
    cfg.pop("manifest", None)
    cfg.pop("reports", None)
    cfg.pop("node_sets_file", None)
    net = os.path.abspath(os.path.join(extract_dir, "network"))
    nodes, edges = [], []
    for p in ("l23", "v1ext", "lgn", "bkg"):
        if os.path.exists(os.path.join(net, f"{p}_nodes.h5")):
            nodes.append({"nodes_file": f"{net}/{p}_nodes.h5", "node_types_file": f"{net}/{p}_node_types.csv"})
    for p in ("l23_l23", "v1ext_l23", "lgn_l23", "bkg_l23"):
        if os.path.exists(os.path.join(net, f"{p}_edges.h5")):
            edges.append({"edges_file": f"{net}/{p}_edges.h5", "edge_types_file": f"{net}/{p}_edge_types.csv"})
    cfg["networks"] = {"nodes": nodes, "edges": edges}
    # keep the full run's LGN / bkg inputs unchanged and add the replayed V1 stream
    inputs = {k: v for k, v in cfg.get("inputs", {}).items()
              if v.get("node_set") in ("lgn", "bkg") or (isinstance(v.get("node_set"), dict)
                                                         and v["node_set"].get("population") in ("lgn", "bkg"))}
    inputs["V1EXT_replay"] = {"input_type": "spikes", "module": "sonata", "input_file": os.path.abspath(v1ext_spikes),
                              "node_set": "v1ext"}
    cfg["inputs"] = inputs
    cfg["output"] = {"output_dir": os.path.abspath(out_dir), "spikes_file": "spikes.h5",
                     "overwrite_output_dir": True, "log_file": "log.txt"}
    if tstop is not None:
        cfg["run"]["tstop"] = float(tstop)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_isolated.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_isolated(full_config: str, extract_dir: str, v1ext_spikes: str, out_dir: str,
                 tstop: float | None = None, threads: int = 1) -> str:
    path = isolated_config(full_config, extract_dir, v1ext_spikes, out_dir, tstop)
    wall = run_bmtk(path, threads)
    with open(path) as f:
        _write_timing(out_dir, wall, json.load(f)["run"]["tstop"])
    return os.path.join(out_dir, "spikes.h5")


def _write_timing(out_dir, wall, tstop_ms):
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump({"wall_s": wall, "tstop_ms": tstop_ms, "wall_s_per_sim_s": wall / (tstop_ms / 1000.0)}, f, indent=2)


# --------------------------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("full")
    a.add_argument("--config", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tstop", type=float)
    a.add_argument("--threads", type=int, default=1)
    b = sub.add_parser("streams")
    b.add_argument("--full-spikes", required=True)
    b.add_argument("--extract-dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--v1-population")
    b.add_argument("--poisson-control", action="store_true",
                   help="also write v1ext_poisson_spikes.h5 (rate-matched Poisson, no timing)")
    b.add_argument("--tstop", type=float, help="end of the Poisson control window (ms)")
    c = sub.add_parser("isolated")
    c.add_argument("--config", required=True, help="config of the full run (inputs and run settings are reused)")
    c.add_argument("--extract-dir", required=True)
    c.add_argument("--v1ext-spikes", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tstop", type=float)
    c.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    if args.cmd == "full":
        print(run_full(args.config, args.out, args.tstop, args.threads))
    elif args.cmd == "streams":
        info = make_streams(args.full_spikes, args.extract_dir, args.out, args.v1_population)
        if args.poisson_control:
            t1 = args.tstop
            if t1 is None:
                t1 = max(float(ts.max(initial=0)) for _, _, ts in sio.iter_spikes(args.full_spikes))
            info["poisson_control"] = poisson_streams(os.path.join(args.out, "v1ext_spikes.h5"),
                                                      os.path.join(args.out, "v1ext_poisson_spikes.h5"), 0.0, t1)
        print(json.dumps(info, indent=2))
    else:
        print(run_isolated(args.config, args.extract_dir, args.v1ext_spikes, args.out, args.tstop, args.threads))


if __name__ == "__main__":
    main()
