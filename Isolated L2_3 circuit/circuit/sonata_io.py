"""Small SONATA readers and writers used to cut L2/3 out of the Allen V1 network.

Only h5py, numpy and pandas are needed, so these work on the Billeh et al. (2020) point-neuron
network files as they are distributed (``network/v1_nodes.h5``, ``v1_v1_edges.h5`` and so on)
without building anything through BMTK. Edge files are read in chunks because the full V1 edge
file is too large to hold in memory at once on a workstation.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterator, Optional

import h5py
import numpy as np
import pandas as pd

EDGE_CHUNK = 5_000_000


# --------------------------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------------------------
def node_population_names(nodes_h5: str):
    with h5py.File(nodes_h5, "r") as f:
        return list(f["nodes"].keys())


def read_nodes(nodes_h5: str, node_types_csv: str, population: Optional[str] = None) -> pd.DataFrame:
    """Return one row per node: node_id, node_type_id, group columns and node-type columns.

    Node-type columns (pop_name, model_template, dynamics_params, ...) are merged in from the
    space-separated ``*_node_types.csv`` file, as SONATA defines it.
    """
    with h5py.File(nodes_h5, "r") as f:
        pops = list(f["nodes"].keys())
        population = population or pops[0]
        g = f["nodes"][population]
        df = pd.DataFrame({
            "node_id": g["node_id"][()].astype(np.int64),
            "node_type_id": g["node_type_id"][()].astype(np.int64),
            "node_group_id": g["node_group_id"][()].astype(np.int64),
            "node_group_index": g["node_group_index"][()].astype(np.int64),
        })
        for grp_name in [k for k in g.keys() if isinstance(g[k], h5py.Group) and k.isdigit()]:
            grp = g[grp_name]
            rows = df["node_group_id"] == int(grp_name)
            idx = df.loc[rows, "node_group_index"].to_numpy()
            for col in grp.keys():
                if isinstance(grp[col], h5py.Dataset) and grp[col].ndim == 1 and len(grp[col]) > 0:
                    vals = grp[col][()]
                    if vals.dtype.kind == "S":
                        vals = vals.astype(str)
                    if col not in df:
                        df[col] = np.nan if vals.dtype.kind in "fiu" else None
                    df.loc[rows, col] = vals[idx]
    types = pd.read_csv(node_types_csv, sep=" ")
    df = df.merge(types, on="node_type_id", how="left", suffixes=("", "_type"))
    df.attrs["population"] = population
    return df


def write_nodes(path: str, population: str, node_type_id: np.ndarray, group_cols: Dict[str, np.ndarray]):
    """Write a single-group SONATA nodes file with node_id = 0..N-1."""
    n = len(node_type_id)
    with h5py.File(path, "w") as f:
        f.attrs["magic"] = np.uint32(0x0A7A)
        f.attrs["version"] = [np.uint32(0), np.uint32(1)]
        g = f.create_group(f"nodes/{population}")
        g.create_dataset("node_id", data=np.arange(n, dtype=np.uint64))
        g.create_dataset("node_type_id", data=np.asarray(node_type_id, dtype=np.uint64))
        g.create_dataset("node_group_id", data=np.zeros(n, dtype=np.uint32))
        g.create_dataset("node_group_index", data=np.arange(n, dtype=np.uint64))
        grp = g.create_group("0")
        grp.create_group("dynamics_params")
        for col, vals in group_cols.items():
            vals = np.asarray(vals)
            if vals.dtype.kind in "OU":
                vals = vals.astype("S")
            grp.create_dataset(col, data=vals)


# --------------------------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------------------------
def edge_population_names(edges_h5: str):
    with h5py.File(edges_h5, "r") as f:
        return list(f["edges"].keys())


def edge_endpoints(edges_h5: str, population: Optional[str] = None):
    """Return (source population, target population, number of edges)."""
    with h5py.File(edges_h5, "r") as f:
        population = population or list(f["edges"].keys())[0]
        g = f["edges"][population]
        src = g["source_node_id"].attrs.get("node_population", b"")
        trg = g["target_node_id"].attrs.get("node_population", b"")
        src = src.decode() if isinstance(src, bytes) else str(src)
        trg = trg.decode() if isinstance(trg, bytes) else str(trg)
        return src, trg, len(g["target_node_id"])


def filter_edges(
    edges_h5: str,
    keep: Callable[[np.ndarray, np.ndarray], np.ndarray],
    population: Optional[str] = None,
    chunk: int = EDGE_CHUNK,
) -> Dict[str, np.ndarray]:
    """Read every edge for which ``keep(source_ids, target_ids)`` is True, chunk by chunk.

    Returns source/target ids, edge_type_id and every column of every edge group, resolved
    per edge (so the output can be rewritten as a single group).
    """
    out: Dict[str, list] = {"source_node_id": [], "target_node_id": [], "edge_type_id": []}
    group_cols: Dict[str, list] = {}
    with h5py.File(edges_h5, "r") as f:
        population = population or list(f["edges"].keys())[0]
        g = f["edges"][population]
        n = len(g["target_node_id"])
        groups = {int(k): g[k] for k in g.keys() if isinstance(g[k], h5py.Group) and k.isdigit()}
        # Load group columns lazily, but whole (they are indexed by edge_group_index).
        group_data = {gid: {c: grp[c] for c in grp.keys() if isinstance(grp[c], h5py.Dataset)}
                      for gid, grp in groups.items()}
        all_cols = sorted({c for cols in group_data.values() for c in cols})
        for c in all_cols:
            group_cols[c] = []
        cache: Dict[tuple, np.ndarray] = {}
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            s = g["source_node_id"][start:stop].astype(np.int64)
            t = g["target_node_id"][start:stop].astype(np.int64)
            m = keep(s, t)
            if not m.any():
                continue
            out["source_node_id"].append(s[m])
            out["target_node_id"].append(t[m])
            out["edge_type_id"].append(g["edge_type_id"][start:stop][m].astype(np.int64))
            gids = g["edge_group_id"][start:stop][m].astype(np.int64)
            gidx = g["edge_group_index"][start:stop][m].astype(np.int64)
            for c in all_cols:
                col = np.full(m.sum(), np.nan)
                for gid, cols in group_data.items():
                    if c not in cols:
                        continue
                    sel = gids == gid
                    if not sel.any():
                        continue
                    key = (gid, c)
                    if key not in cache:
                        cache[key] = cols[c][()]
                    col = col.astype(np.result_type(col.dtype, cache[key].dtype))
                    col[sel] = cache[key][gidx[sel]]
                group_cols[c].append(col)
    res = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64)) for k, v in out.items()}
    for c, v in group_cols.items():
        res[c] = np.concatenate(v) if v else np.zeros(0)
    return res


def write_edges(path: str, population: str, src_pop: str, trg_pop: str, edges: Dict[str, np.ndarray]):
    """Write a single-group SONATA edges file, sorted by target, with source and target indices."""
    from bmtk.builder.index_builders import create_index_in_memory

    order = np.lexsort((edges["source_node_id"], edges["target_node_id"]))
    n = len(order)
    with h5py.File(path, "w") as f:
        f.attrs["magic"] = np.uint32(0x0A7A)
        f.attrs["version"] = [np.uint32(0), np.uint32(1)]
        g = f.create_group(f"edges/{population}")
        ds = g.create_dataset("source_node_id", data=edges["source_node_id"][order].astype(np.uint64))
        ds.attrs["node_population"] = src_pop
        ds = g.create_dataset("target_node_id", data=edges["target_node_id"][order].astype(np.uint64))
        ds.attrs["node_population"] = trg_pop
        g.create_dataset("edge_type_id", data=edges["edge_type_id"][order].astype(np.uint64))
        g.create_dataset("edge_group_id", data=np.zeros(n, dtype=np.uint32))
        g.create_dataset("edge_group_index", data=np.arange(n, dtype=np.uint64))
        grp = g.create_group("0")
        for c, v in edges.items():
            if c in ("source_node_id", "target_node_id", "edge_type_id"):
                continue
            grp.create_dataset(c, data=np.asarray(v)[order])
    create_index_in_memory(path, f"edges/{population}", "target")
    create_index_in_memory(path, f"edges/{population}", "source")


def iter_spikes(spikes_h5: str) -> Iterator[tuple]:
    """Yield (population, node_ids, timestamps_ms) from a SONATA spikes file."""
    with h5py.File(spikes_h5, "r") as f:
        for pop in f["spikes"].keys():
            g = f["spikes"][pop]
            yield pop, g["node_ids"][()].astype(np.int64), g["timestamps"][()].astype(np.float64)


def write_spikes(path: str, population: str, node_ids: np.ndarray, timestamps: np.ndarray):
    """Write a SONATA spikes file (sorted by time), readable by BMTK as a spike-train input."""
    order = np.argsort(timestamps, kind="stable")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group(f"spikes/{population}")
        g.attrs["sorting"] = "by_time"
        g.create_dataset("node_ids", data=np.asarray(node_ids)[order].astype(np.uint64))
        ds = g.create_dataset("timestamps", data=np.asarray(timestamps)[order].astype(np.float64))
        ds.attrs["units"] = "ms"
