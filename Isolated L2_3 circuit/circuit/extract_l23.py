"""Cut the L2/3 circuit out of the Allen V1 point-neuron network (Billeh et al. 2020).

What is kept, unchanged:
  * every L2/3 node (excitatory, Pvalb, Sst, Htr3a; pop_name matching ``--l23-pattern``),
    with its GLIF model, position and tuning angle,
  * every V1 -> V1 edge whose source and target are both kept L2/3 cells (weights, delays,
    nsyns and edge types as in the original file).

What is replaced by replayed spike streams:
  * every other V1 cell that synapses onto a kept L2/3 cell (L1, L4, L5, L6, and L2/3 cells
    outside ``--radius`` if one is given) becomes a node of the virtual population ``v1ext``,
  * LGN and background (bkg) nodes stay virtual, as they already are in the original network.
Their edges onto L2/3 are copied unchanged, so a replayed spike arrives at the same synapse,
with the same weight and delay, as in the full network.

Usage:
    python -m circuit.extract_l23 --network-dir <Allen GLIF network>/network --out <out dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from . import sonata_io as sio

L23_PATTERN = r"^[ei]23"
# Columns that come from node_types.csv and should not be copied into the h5 group.
TYPE_COLS_SKIP = {"node_id", "node_type_id", "node_group_id", "node_group_index"}


def classify(pop_name: str) -> str:
    """Return the Billeh cell class of a pop_name, e.g. 'e23Cux2' -> 'L2/3 E', 'i4Pvalb' -> 'L4 Pvalb'."""
    m = re.match(r"^([ei])(\d+)(.*)$", str(pop_name))
    if not m:
        return str(pop_name)
    ei, layer, sub = m.groups()
    layer = {"1": "L1", "23": "L2/3", "4": "L4", "5": "L5", "6": "L6"}.get(layer, "L" + layer)
    if ei == "e":
        return f"{layer} E"
    for k in ("Pvalb", "Sst", "Htr3a"):
        if k.lower() in sub.lower():
            return f"{layer} {k}"
    return f"{layer} I"


def _find(network_dir: str, pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(os.path.join(network_dir, pattern)))
    return hits[0] if hits else None


def find_files(network_dir: str) -> dict:
    """Locate the SONATA files of the Allen GLIF network (names as distributed)."""
    files = {
        "v1_nodes": _find(network_dir, "v1_nodes.h5"),
        "v1_node_types": _find(network_dir, "v1_node_types.csv"),
        "v1_v1_edges": _find(network_dir, "v1_v1_edges.h5"),
        "v1_v1_edge_types": _find(network_dir, "v1_v1_edge_types.csv"),
    }
    for ext in ("lgn", "bkg"):
        files[f"{ext}_nodes"] = _find(network_dir, f"{ext}_nodes.h5")
        files[f"{ext}_node_types"] = _find(network_dir, f"{ext}_node_types.csv")
        files[f"{ext}_v1_edges"] = _find(network_dir, f"{ext}_v1_edges.h5")
        files[f"{ext}_v1_edge_types"] = _find(network_dir, f"{ext}_v1_edge_types.csv")
    missing = [k for k in ("v1_nodes", "v1_node_types", "v1_v1_edges", "v1_v1_edge_types") if not files[k]]
    if missing:
        raise FileNotFoundError(f"{network_dir} is missing {missing}")
    return files


def select_l23(nodes: pd.DataFrame, pattern: str = L23_PATTERN, radius: Optional[float] = None) -> np.ndarray:
    keep = nodes["pop_name"].astype(str).str.match(pattern).to_numpy()
    if radius is not None:
        if not {"x", "z"} <= set(nodes.columns):
            raise ValueError("--radius needs x and z node coordinates")
        r = np.hypot(nodes["x"].to_numpy(float), nodes["z"].to_numpy(float))
        keep &= r <= radius
    return keep


def extract(network_dir: str, out_dir: str, pattern: str = L23_PATTERN, radius: Optional[float] = None,
            chunk: int = sio.EDGE_CHUNK, max_edges_per_file: int = 2_000_000) -> dict:
    files = find_files(network_dir)
    net_out = os.path.join(out_dir, "network")
    os.makedirs(net_out, exist_ok=True)

    v1_pop = sio.node_population_names(files["v1_nodes"])[0]
    nodes = sio.read_nodes(files["v1_nodes"], files["v1_node_types"], v1_pop)
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    keep = select_l23(nodes, pattern, radius)
    v1_ids = nodes["node_id"].to_numpy()
    l23_v1_ids = v1_ids[keep]
    n_v1 = int(v1_ids.max()) + 1
    v1_to_l23 = np.full(n_v1, -1, dtype=np.int64)
    v1_to_l23[l23_v1_ids] = np.arange(len(l23_v1_ids))
    is_l23 = v1_to_l23 >= 0
    print(f"[extract] {len(l23_v1_ids)} L2/3 cells kept out of {len(v1_ids)} V1 cells")

    # ---- L2/3 nodes ------------------------------------------------------------------------
    type_cols = set(pd.read_csv(files["v1_node_types"], sep=" ").columns)
    l23_nodes = nodes[keep].reset_index(drop=True)
    group_cols = {c: l23_nodes[c].to_numpy() for c in l23_nodes.columns
                  if c not in TYPE_COLS_SKIP and c not in type_cols and not c.endswith("_type")}
    sio.write_nodes(os.path.join(net_out, "l23_nodes.h5"), "l23", l23_nodes["node_type_id"].to_numpy(), group_cols)
    v1_types = pd.read_csv(files["v1_node_types"], sep=" ")
    v1_types[v1_types["node_type_id"].isin(l23_nodes["node_type_id"].unique())].to_csv(
        os.path.join(net_out, "l23_node_types.csv"), sep=" ", index=False)

    # ---- V1 -> V1 edges: split into recurrent L2/3 and external-to-L2/3 -------------------
    v1_edge_pop = sio.edge_population_names(files["v1_v1_edges"])[0]
    onto_l23 = sio.filter_edges(files["v1_v1_edges"], lambda s, t: is_l23[t], v1_edge_pop, chunk)
    src_is_l23 = is_l23[onto_l23["source_node_id"]]
    rec = {k: v[src_is_l23] for k, v in onto_l23.items()}
    ext = {k: v[~src_is_l23] for k, v in onto_l23.items()}
    rec["source_node_id"] = v1_to_l23[rec["source_node_id"]]
    rec["target_node_id"] = v1_to_l23[rec["target_node_id"]]
    sio.write_edges(os.path.join(net_out, "l23_l23_edges.h5"), "l23_to_l23", "l23", "l23", rec)
    shutil.copy(files["v1_v1_edge_types"], os.path.join(net_out, "l23_l23_edge_types.csv"))

    # The external V1 sources become the virtual population v1ext (re-indexed 0..M-1).
    ext_v1_ids = np.unique(ext["source_node_id"])
    v1_to_ext = np.full(n_v1, -1, dtype=np.int64)
    v1_to_ext[ext_v1_ids] = np.arange(len(ext_v1_ids))
    ext_nodes = nodes.set_index("node_id").loc[ext_v1_ids].reset_index()
    ext_cols = {"pop_name_orig": ext_nodes["pop_name"].astype(str).to_numpy()}
    for c in ("x", "y", "z", "tuning_angle"):
        if c in ext_nodes:
            ext_cols[c] = ext_nodes[c].to_numpy(float)
    sio.write_nodes(os.path.join(net_out, "v1ext_nodes.h5"), "v1ext", ext_nodes["node_type_id"].to_numpy(), ext_cols)
    ext_types = v1_types[v1_types["node_type_id"].isin(ext_nodes["node_type_id"].unique())]
    keep_cols = [c for c in ("node_type_id", "pop_name", "ei", "location") if c in ext_types]
    ext_types = ext_types[keep_cols].copy()
    ext_types["model_type"] = "virtual"
    ext_types.to_csv(os.path.join(net_out, "v1ext_node_types.csv"), sep=" ", index=False)
    ext["source_node_id"] = v1_to_ext[ext["source_node_id"]]
    ext["target_node_id"] = v1_to_l23[ext["target_node_id"]]
    sio.write_edges_split(os.path.join(net_out, "v1ext_l23_edges.h5"), "v1ext_to_l23", "v1ext", "l23", ext,
                          max_edges_per_file)
    shutil.copy(files["v1_v1_edge_types"], os.path.join(net_out, "v1ext_l23_edge_types.csv"))

    # ---- LGN and background edges onto L2/3 ------------------------------------------------
    virtual_pops = {}
    for name in ("lgn", "bkg"):
        if not files.get(f"{name}_v1_edges"):
            print(f"[extract] no {name}_v1_edges.h5 found, skipping {name}")
            continue
        src_pop, _, _ = sio.edge_endpoints(files[f"{name}_v1_edges"])
        e = sio.filter_edges(files[f"{name}_v1_edges"], lambda s, t: is_l23[t], None, chunk)
        e["target_node_id"] = v1_to_l23[e["target_node_id"]]
        sio.write_edges(os.path.join(net_out, f"{name}_l23_edges.h5"), f"{name}_to_l23", src_pop, "l23", e)
        shutil.copy(files[f"{name}_v1_edge_types"], os.path.join(net_out, f"{name}_l23_edge_types.csv"))
        shutil.copy(files[f"{name}_nodes"], os.path.join(net_out, f"{name}_nodes.h5"))
        shutil.copy(files[f"{name}_node_types"], os.path.join(net_out, f"{name}_node_types.csv"))
        virtual_pops[name] = {"population": src_pop, "edges_onto_l23": int(len(e["target_node_id"])),
                              "synapses_onto_l23": int(_nsyns(e).sum())}

    with h5py.File(os.path.join(out_dir, "id_maps.h5"), "w") as f:
        f.create_dataset("l23_to_v1", data=l23_v1_ids.astype(np.int64))
        f.create_dataset("v1ext_to_v1", data=ext_v1_ids.astype(np.int64))
        f.attrs["v1_population"] = v1_pop

    summary = summarize(nodes, keep, rec, ext, ext_nodes, virtual_pops, pattern, radius)
    summary["source_network_dir"] = os.path.abspath(network_dir)
    summary["v1_population"] = v1_pop
    with open(os.path.join(out_dir, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _nsyns(e: dict) -> np.ndarray:
    n = len(e["target_node_id"])
    return np.asarray(e["nsyns"], dtype=float) if "nsyns" in e and len(e["nsyns"]) == n else np.ones(n)


def summarize(nodes, keep, rec, ext, ext_nodes, virtual_pops, pattern, radius) -> dict:
    l23 = nodes[keep]
    cls = l23["pop_name"].map(classify)
    ext_class = ext_nodes["pop_name"].map(classify).to_numpy()[ext["source_node_id"]]
    syn_rec, syn_ext = _nsyns(rec), _nsyns(ext)
    by_src = pd.Series(syn_ext).groupby(ext_class).sum().sort_values(ascending=False)
    total = syn_rec.sum() + syn_ext.sum() + sum(v["synapses_onto_l23"] for v in virtual_pops.values())
    out = {
        "l23_pattern": pattern,
        "radius_um": radius,
        "n_l23_cells": int(keep.sum()),
        "n_v1_cells": int(len(nodes)),
        "l23_cells_by_class": {k: int(v) for k, v in cls.value_counts().sort_index().items()},
        "l23_pop_names": {k: int(v) for k, v in l23["pop_name"].value_counts().sort_index().items()},
        "recurrent_l23_edges": int(len(rec["target_node_id"])),
        "recurrent_l23_synapses": int(syn_rec.sum()),
        "n_v1ext_sources": int(len(ext_nodes)),
        "v1ext_edges_onto_l23": int(len(ext["target_node_id"])),
        "v1ext_synapses_onto_l23_by_source_class": {k: int(v) for k, v in by_src.items()},
        "virtual_inputs": virtual_pops,
        "fraction_of_l23_input_synapses_recurrent": float(syn_rec.sum() / total) if total else None,
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network-dir", required=True, help="folder with v1_nodes.h5, v1_v1_edges.h5, ...")
    p.add_argument("--out", required=True)
    p.add_argument("--l23-pattern", default=L23_PATTERN, help="regex on pop_name (default %(default)s)")
    p.add_argument("--radius", type=float, default=None,
                   help="keep only L2/3 cells within this radius (um) of the column axis; others are replayed")
    p.add_argument("--chunk", type=int, default=sio.EDGE_CHUNK)
    p.add_argument("--max-edges-per-file", type=int, default=2_000_000,
                   help="split the v1ext replay edges into files of this size (BMTK memory)")
    a = p.parse_args(argv)
    s = extract(a.network_dir, a.out, a.l23_pattern, a.radius, a.chunk, a.max_edges_per_file)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
