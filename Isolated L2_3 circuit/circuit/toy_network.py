"""Build a small Billeh-like V1 point network, used to test the L2/3 extraction end to end.

It has the same file layout, population names and model types as the Allen GLIF network
(``network/v1_nodes.h5`` with pop_names such as ``e23Cux2`` or ``i4Pvalb``, NEST ``glif_psc``
cells with four synaptic ports, virtual ``lgn`` and ``bkg`` populations), but only a few
hundred cells, so a full-network run and the isolated L2/3 replay take seconds.
"""
from __future__ import annotations

import json
import os

import numpy as np

# (pop_name, location, ei, n, model json)
POPS = [
    ("i1Htr3a", "VisL1", "i", 10, "htr3a.json"),
    ("e23Cux2", "VisL23", "e", 120, "exc.json"),
    ("i23Pvalb", "VisL23", "i", 16, "pvalb.json"),
    ("i23Sst", "VisL23", "i", 10, "sst.json"),
    ("i23Htr3a", "VisL23", "i", 8, "htr3a.json"),
    ("e4Rorb", "VisL4", "e", 100, "exc.json"),
    ("i4Pvalb", "VisL4", "i", 14, "pvalb.json"),
    ("e5Rbp4", "VisL5", "e", 60, "exc.json"),
    ("i5Pvalb", "VisL5", "i", 10, "pvalb.json"),
]

TAU_SYN = [5.5, 8.5, 2.8, 5.8]  # Billeh: exc->, ..., the four receptor ports


def _glif(V_th, g, C_m, t_ref, E_L=-70.0, asc=(-10.0, -100.0)):
    return {
        "V_th": V_th, "g": g, "E_L": E_L, "C_m": C_m, "t_ref": t_ref, "V_reset": E_L, "V_m": E_L,
        "asc_init": [0.0, 0.0], "asc_decay": [0.003, 0.1], "asc_amps": list(asc), "asc_r": [1.0, 1.0],
        "tau_syn": TAU_SYN, "after_spike_currents": True, "spike_dependent_threshold": False,
        "adapting_threshold": False,
    }


CELL_MODELS = {
    "exc.json": _glif(V_th=-50.0, g=6.0, C_m=80.0, t_ref=3.0),
    "pvalb.json": _glif(V_th=-52.0, g=8.0, C_m=50.0, t_ref=2.0, asc=(-2.0, -10.0)),
    "sst.json": _glif(V_th=-51.0, g=5.0, C_m=60.0, t_ref=2.5, asc=(-5.0, -30.0)),
    "htr3a.json": _glif(V_th=-51.0, g=5.0, C_m=50.0, t_ref=2.5, asc=(-5.0, -30.0)),
}


def build(out_dir: str, seed: int = 0, lgn_rate_hz: float = 8.0, bkg_rate_hz: float = 1000.0,
          tstop_ms: float = 1000.0) -> str:
    """Write the toy network, components, input spikes and a BMTK config. Returns the config path."""
    from bmtk.builder import NetworkBuilder
    from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

    rng = np.random.default_rng(seed)
    net_dir = os.path.join(out_dir, "network")
    comp = os.path.join(out_dir, "components")
    for d in (net_dir, os.path.join(comp, "cell_models"), os.path.join(comp, "synaptic_models"),
              os.path.join(out_dir, "inputs")):
        os.makedirs(d, exist_ok=True)
    for name, params in CELL_MODELS.items():
        with open(os.path.join(comp, "cell_models", name), "w") as f:
            json.dump(params, f, indent=1)
    for port in range(1, 5):
        with open(os.path.join(comp, "synaptic_models", f"receptor{port}.json"), "w") as f:
            json.dump({"receptor_type": port}, f)

    v1 = NetworkBuilder("v1")
    for pop, loc, ei, n, model in POPS:
        r = 200 * np.sqrt(rng.uniform(size=n))
        th = rng.uniform(0, 2 * np.pi, n)
        depth = {"VisL1": 50, "VisL23": 250, "VisL4": 400, "VisL5": 600}[loc]
        v1.add_nodes(N=n, pop_name=pop, location=loc, ei=ei, model_type="point_process",
                     model_template="nest:glif_psc", dynamics_params=model,
                     x=r * np.cos(th), y=-depth + rng.normal(0, 20, n), z=r * np.sin(th),
                     tuning_angle=rng.uniform(0, 360, n))

    # Receptor ports follow Billeh: 1 = E->E, 2 = I->E/I (slow), 3 = E->I, 4 = I->I (fast).
    def port(src_ei, trg_ei):
        return {("e", "e"): 1, ("i", "e"): 2, ("e", "i"): 3, ("i", "i"): 4}[(src_ei, trg_ei)]

    w = {("e", "e"): 18.0, ("e", "i"): 30.0, ("i", "e"): -60.0, ("i", "i"): -50.0}
    for spop, sloc, sei, _, _ in POPS:
        for tpop, tloc, tei, _, _ in POPS:
            p = 0.12 if sloc == tloc else (0.10 if (sloc, tloc) in {("VisL4", "VisL23"), ("VisL23", "VisL5")} else 0.03)

            def conn(src, trg, p=p):
                if src.node_id == trg.node_id:
                    return 0
                return int(rng.uniform() < p) * int(rng.integers(1, 4))

            v1.add_edges(source={"pop_name": spop}, target={"pop_name": tpop}, connection_rule=conn,
                         syn_weight=w[(sei, tei)] * rng.uniform(0.7, 1.3), delay=float(rng.choice([1.0, 1.5, 2.0])),
                         dynamics_params=f"receptor{port(sei, tei)}.json", model_template="static_synapse")
    v1.build()
    v1.save_nodes(output_dir=net_dir)
    v1.save_edges(output_dir=net_dir)

    lgn = NetworkBuilder("lgn")
    lgn.add_nodes(N=60, pop_name="tON", model_type="virtual", ei="e", location="LGN")
    for tpop, tloc, tei, _, _ in POPS:
        if tloc not in ("VisL4", "VisL23"):
            continue
        lgn.add_edges(source=lgn.nodes(), target=v1.nodes(pop_name=tpop),
                      connection_rule=lambda s, t: int(rng.uniform() < (0.15 if tloc == "VisL4" else 0.04)) * 2,
                      syn_weight=25.0, delay=1.7, dynamics_params=f"receptor{port('e', tei)}.json",
                      model_template="static_synapse")
    lgn.build()
    lgn.save_nodes(output_dir=net_dir)
    lgn.save_edges(output_dir=net_dir)

    bkg = NetworkBuilder("bkg")
    bkg.add_nodes(N=1, pop_name="bkg", model_type="virtual", ei="e", location="BKG")
    bkg.add_edges(source=bkg.nodes(), target=v1.nodes(), connection_rule=lambda s, t: 4,
                  syn_weight=11.0, delay=1.0, dynamics_params="receptor1.json", model_template="static_synapse")
    bkg.build()
    bkg.save_nodes(output_dir=net_dir)
    bkg.save_edges(output_dir=net_dir)

    for name, n_nodes, rate in (("lgn", 60, lgn_rate_hz), ("bkg", 1, bkg_rate_hz)):
        psg = PoissonSpikeGenerator(population=name, seed=seed + (1 if name == "lgn" else 2))
        psg.add(node_ids=list(range(n_nodes)), firing_rate=rate, times=(0.0, tstop_ms / 1000.0))
        psg.to_sonata(os.path.join(out_dir, "inputs", f"{name}_spikes.h5"))

    cfg = {
        "manifest": {"$BASE_DIR": os.path.abspath(out_dir), "$NETWORK_DIR": "$BASE_DIR/network",
                     "$COMPONENTS_DIR": "$BASE_DIR/components"},
        "run": {"tstop": tstop_ms, "dt": 0.25, "nsteps_block": 1000},
        "target_simulator": "NEST",
        "inputs": {
            "LGN_spikes": {"input_type": "spikes", "module": "sonata", "input_file": "$BASE_DIR/inputs/lgn_spikes.h5",
                           "node_set": "lgn"},
            "BKG_spikes": {"input_type": "spikes", "module": "sonata", "input_file": "$BASE_DIR/inputs/bkg_spikes.h5",
                           "node_set": "bkg"},
        },
        "output": {"output_dir": "$BASE_DIR/output_full", "spikes_file": "spikes.h5", "overwrite_output_dir": True,
                   "log_file": "log.txt"},
        "components": {"point_neuron_models_dir": "$COMPONENTS_DIR/cell_models",
                       "synaptic_models_dir": "$COMPONENTS_DIR/synaptic_models"},
        "networks": {
            "nodes": [{"nodes_file": f"$NETWORK_DIR/{p}_nodes.h5", "node_types_file": f"$NETWORK_DIR/{p}_node_types.csv"}
                      for p in ("v1", "lgn", "bkg")],
            "edges": [{"edges_file": f"$NETWORK_DIR/{p}_edges.h5", "edge_types_file": f"$NETWORK_DIR/{p}_edge_types.csv"}
                      for p in ("v1_v1", "lgn_v1", "bkg_v1")],
        },
    }
    path = os.path.join(out_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


if __name__ == "__main__":
    import sys
    print(build(sys.argv[1] if len(sys.argv) > 1 else "toy_v1"))
