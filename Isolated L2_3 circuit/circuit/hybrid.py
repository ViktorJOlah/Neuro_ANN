"""Isolated L2/3 circuit with ANN pyramidal cells and GLIF interneurons (step 5b).

Every L2/3 interneuron (Pvalb, Sst, Htr3a) stays the Allen GLIF cell and runs in NEST at the
network's own 0.25 ms resolution. Every L2/3 PC (56,057 cells) is the Standard ANN cell. At each
whole millisecond the PCs read their summed synaptic current for the past millisecond and are all
advanced together, as one batch, by one 1 ms step of the Standard model. The two halves trade
spikes every millisecond:

    for k in 0 .. T-1 (ms):
        1. PC input for (k, k+1]: replayed input (precomputed) + interneuron and PC spikes that
           arrive in this millisecond, through the fixed synaptic layer (Allen alpha PSCs)
        2. one batched Standard step for all PCs -> PC spikes, stamped at k+1
        3. PC spikes are handed to NEST as spike-generator times at k+1
        4. NEST advances the interneurons from k to k+1 (4 steps of 0.25 ms); their spikes are
           routed to the PCs' synaptic layer with each edge's NEST delay

Every edge keeps its Allen weight, delay and receptor port. Causality holds because each spike
reaches its target at least 0.25 ms (PC -> PC) or 1.25 ms (interneuron -> PC) after it is
fired, so it always lands in a later PC millisecond than the one that produced it.

Inputs from outside L2/3 (L4/L5/L6/L1 replay, LGN, background) are the same spike streams as in
the GLIF isolated circuit. Onto interneurons they go through NEST as before. Onto PCs they don't
depend on the circuit, so their current is precomputed once (``prepare``).

Gain: ``--gain a`` multiplies every synapse onto a PC (excitatory and inhibitory, replayed and
recurrent) and every synapse from a PC (onto PCs and interneurons) by the same factor a, so
the E/I ratio of each cell's input is unchanged. ``tune`` searches a until the PC rate matches
the GLIF network.

    python -m circuit.hybrid prepare --config V1/config.json --extract-dir X --v1ext-spikes R/streams/v1ext_spikes.h5 --out H
    python -m circuit.hybrid run --prep H --standard-dir Standard --gain 1.0 --tstop 3000 --out H/run_g1.00
    python -m circuit.hybrid tune --prep H --standard-dir Standard --reference R/streams/l23_reference_spikes.h5 --out H/tune
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

from . import sonata_io as sio
from .extract_l23 import classify
from .synaptic_layer import TAU_SYN, nest_effective_delay

RES = 0.25           # NEST resolution of the Allen network (ms)
SUB = int(round(1.0 / RES))
RECEPTOR = {"e2e.json": 1, "i2e.json": 2, "e2i.json": 3, "i2i.json": 4}
PC_PORTS = (1, 2)    # receptor ports onto E cells (E->E, I->E)


# ============================================================================================
# prepare
# ============================================================================================
def _edge_files(net, prefix):
    return sorted(glob.glob(os.path.join(net, f"{prefix}_edges.h5")) + glob.glob(os.path.join(net, f"{prefix}_edges.part*.h5")))


def _edge_table(net, prefix, keep):
    """All edges of ``prefix`` with keep(src, trg) True, plus their delay and receptor port."""
    types = pd.read_csv(os.path.join(net, f"{prefix}_edge_types.csv"), sep=" ").set_index("edge_type_id")
    parts = [sio.filter_edges(fn, keep) for fn in _edge_files(net, prefix)]
    e = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    et = types.loc[e["edge_type_id"]]
    e["delay"] = et["delay"].to_numpy(float)
    e["port"] = et["dynamics_params"].map(RECEPTOR).to_numpy(np.int64)
    w = e["syn_weight"].astype(float)
    if "nsyns" in e:
        w = w * e["nsyns"]
    e["weight"] = w
    return e


def _read_spikes(path, population=None):
    if path.endswith(".csv"):
        df = pd.read_csv(path, sep=" ")
        return df["node_ids"].to_numpy(np.int64), df["timestamps"].to_numpy(float)
    for pop, ids, ts in sio.iter_spikes(path):
        if population is None or pop == population:
            return ids.astype(np.int64), ts.astype(float)
    raise ValueError(f"no spikes for {population} in {path}")


class CSR:
    """Outgoing edges grouped by source: target index, weight, arrival offset (0.25 ms steps), port."""

    def __init__(self, src, trg, w, off, port, n_src):
        order = np.argsort(src, kind="stable")
        self.ptr = np.zeros(n_src + 1, np.int64)
        np.add.at(self.ptr, src + 1, 1)
        self.ptr = np.cumsum(self.ptr)
        self.trg, self.w = trg[order].astype(np.int64), w[order].astype(np.float32)
        self.off, self.port = off[order].astype(np.int64), port[order].astype(np.int64)

    def expand(self, srcs):
        """Edge indices of all outgoing edges of the source list (repeats allowed)."""
        lo, hi = self.ptr[srcs], self.ptr[srcs + 1]
        n = hi - lo
        if n.sum() == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        rep = np.repeat(np.arange(len(srcs)), n)
        idx = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + lo[rep]
        return idx, rep

    def save(self, path):
        np.savez(path, ptr=self.ptr, trg=self.trg, w=self.w, off=self.off, port=self.port)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        obj = cls.__new__(cls)
        obj.ptr, obj.trg, obj.w, obj.off, obj.port = (z[k] for k in ("ptr", "trg", "w", "off", "port"))
        return obj


def _offsets(delay):
    """Arrival offset in 0.25 ms steps: a spike stamped at t starts its PSC at t + NEST's effective delay."""
    return np.round(nest_effective_delay(delay, RES) / RES).astype(np.int64)


class PSCBank:
    """Alpha-PSC state of the PCs' two receptor ports at 0.25 ms, with a ring of future arrivals.

    ``push(step, trg, w, port)`` adds arrivals whose jump happens at the end of absolute 0.25 ms
    step ``step``. ``advance_ms()`` integrates the next 4 steps and returns the mean current of
    that millisecond (trapezoid over its 5 grid points).
    """

    def __init__(self, n, ring=64):
        self.n, self.ring = n, ring
        self.taus = np.array([TAU_SYN[p] for p in PC_PORTS])[:, None]
        e1 = np.exp(-RES / self.taus)
        self.P11, self.P21, self.P22 = e1, RES * e1, e1
        self.y1 = np.zeros((2, n))
        self.I = np.zeros((2, n))
        self.buf = np.zeros((ring, 2, n))
        self.j = 0  # next 0.25 ms step to integrate

    def push(self, step, trg, w, port):
        if len(trg) == 0:
            return
        if (step < self.j).any() or (step >= self.j + self.ring).any():
            raise ValueError("arrival outside the PSC ring (causality or ring size)")
        pk = np.where(port == PC_PORTS[0], 0, 1)
        flat = ((step % self.ring) * 2 + pk) * self.n + trg
        self.buf.reshape(-1)[:] += np.bincount(flat, weights=w, minlength=self.buf.size)

    def push_sparse(self, step, trg, w, port):
        """Same as push, for few arrivals (no full-size bincount)."""
        if len(trg) == 0:
            return
        if (step < self.j).any() or (step >= self.j + self.ring).any():
            raise ValueError("arrival outside the PSC ring (causality or ring size)")
        pk = np.where(port == PC_PORTS[0], 0, 1)
        np.add.at(self.buf, (step % self.ring, pk, trg), w)

    def advance_ms(self, per_port=False):
        """Mean current of the next millisecond: summed (n,), or per port (2, n) with ``per_port``."""
        acc = 0.5 * self.I.copy()
        for q in range(SUB):
            self.I = self.P21 * self.y1 + self.P22 * self.I
            self.y1 = self.P11 * self.y1
            slot = self.j % self.ring
            self.y1 += self.buf[slot] * (np.e / self.taus)
            self.buf[slot] = 0.0
            self.j += 1
            acc += self.I if q < SUB - 1 else 0.5 * self.I
        acc /= SUB
        return acc if per_port else acc.sum(0)


def prepare(config, extract_dir, v1ext_spikes, out, tstop=3000.0, block_ms=100):
    from .simulate import resolved_config
    t_start = time.time()
    net = os.path.join(extract_dir, "network")
    os.makedirs(out, exist_ok=True)
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    cls = nodes["pop_name"].map(classify).to_numpy()
    is_pc = cls == "L2/3 E"
    pc_ids, in_ids = np.flatnonzero(is_pc), np.flatnonzero(~is_pc)
    l23_to_pc = np.full(len(nodes), -1, np.int64); l23_to_pc[pc_ids] = np.arange(len(pc_ids))
    l23_to_in = np.full(len(nodes), -1, np.int64); l23_to_in[in_ids] = np.arange(len(in_ids))
    n_pc, n_in = len(pc_ids), len(in_ids)
    print(f"[prepare] {n_pc} PCs (ANN), {n_in} interneurons (GLIF)")

    # ---- NEST part: interneurons + PC spike generators -------------------------------------
    nest_dir = os.path.join(out, "nest_network")
    os.makedirs(nest_dir, exist_ok=True)
    skip = {"node_id", "node_type_id", "node_group_id", "node_group_index", "pop_name", "model_type", "model_template",
            "dynamics_params", "ei", "location", "population", "layer", "gaba_synapse"}
    sub = nodes.iloc[in_ids]
    cols = {c: sub[c].to_numpy() for c in sub.columns if c not in skip and not c.endswith("_type")}
    sio.write_nodes(os.path.join(nest_dir, "l23i_nodes.h5"), "l23i", sub["node_type_id"].to_numpy(), cols)
    types = pd.read_csv(os.path.join(net, "l23_node_types.csv"), sep=" ")
    types[types["node_type_id"].isin(sub["node_type_id"].unique())].to_csv(
        os.path.join(nest_dir, "l23i_node_types.csv"), sep=" ", index=False)
    pc_type = int(nodes.iloc[pc_ids]["node_type_id"].iloc[0])
    sio.write_nodes(os.path.join(nest_dir, "l23pc_nodes.h5"), "l23pc", np.full(n_pc, pc_type), {})
    pd.DataFrame({"node_type_id": [pc_type], "pop_name": ["e23Cux2_ANN"], "ei": ["e"], "location": ["VisL23"],
                  "model_type": ["virtual"]}).to_csv(os.path.join(nest_dir, "l23pc_node_types.csv"), sep=" ", index=False)
    # dummy spike after tstop so BMTK makes the l23pc spike generators; real times are set during the run
    sio.write_spikes(os.path.join(nest_dir, "l23pc_dummy_spikes.h5"), "l23pc", np.array([0]), np.array([tstop + 1000.0]))

    to_in = lambda s, t: ~is_pc[t]  # noqa: E731
    rec_in = _edge_table(net, "l23_l23", to_in)
    src_pc = is_pc[rec_in["source_node_id"]]
    for name, m, sp in (("l23i_l23i", ~src_pc, "l23i"), ("l23pc_l23i", src_pc, "l23pc")):
        e = {k: rec_in[k][m] for k in ("source_node_id", "target_node_id", "edge_type_id", "syn_weight")}
        e["source_node_id"] = (l23_to_in if sp == "l23i" else l23_to_pc)[e["source_node_id"]]
        e["target_node_id"] = l23_to_in[e["target_node_id"]]
        sio.write_edges(os.path.join(nest_dir, f"{name}_edges.h5"), name, sp, "l23i", e)
        os.system(f"cp '{os.path.join(net, 'l23_l23_edge_types.csv')}' '{os.path.join(nest_dir, name + '_edge_types.csv')}'")
    for prefix, sp in (("v1ext_l23", "v1ext"), ("lgn_l23", "lgn"), ("bkg_l23", "bkg")):
        e = _edge_table(net, prefix, to_in)
        e = {k: e[k] for k in ("source_node_id", "target_node_id", "edge_type_id", "syn_weight")}
        e["target_node_id"] = l23_to_in[e["target_node_id"]]
        name = prefix.replace("l23", "l23i")
        sio.write_edges_split(os.path.join(nest_dir, f"{name}_edges.h5"), name, sp, "l23i", e, 2_000_000)
        os.system(f"cp '{os.path.join(net, prefix + '_edge_types.csv')}' '{os.path.join(nest_dir, name + '_edge_types.csv')}'")
    for p in ("v1ext", "lgn", "bkg"):
        for suf in ("_nodes.h5", "_node_types.csv"):
            os.system(f"cp '{os.path.join(net, p + suf)}' '{nest_dir}/'")

    cfg = resolved_config(config)
    for k in ("manifest", "reports", "node_sets_file"):
        cfg.pop(k, None)
    cfg["networks"] = {
        "nodes": [{"nodes_file": f"{nest_dir}/{p}_nodes.h5", "node_types_file": f"{nest_dir}/{p}_node_types.csv"}
                  for p in ("l23i", "l23pc", "v1ext", "lgn", "bkg")],
        "edges": [{"edges_file": fn, "edge_types_file": f"{nest_dir}/{p}_edge_types.csv"}
                  for p in ("l23i_l23i", "l23pc_l23i", "v1ext_l23i", "lgn_l23i", "bkg_l23i")
                  for fn in _edge_files(nest_dir, p)]}
    ins = {k: v for k, v in cfg.get("inputs", {}).items() if v.get("node_set") in ("lgn", "bkg")}
    lgn_file = [v["input_file"] for v in ins.values() if v["node_set"] == "lgn"][0]
    bkg_file = [v["input_file"] for v in ins.values() if v["node_set"] == "bkg"][0]
    ins["V1EXT_replay"] = {"input_type": "spikes", "module": "sonata", "input_file": os.path.abspath(v1ext_spikes),
                           "node_set": "v1ext"}
    ins["L23PC_ann"] = {"input_type": "spikes", "module": "sonata",
                        "input_file": os.path.join(nest_dir, "l23pc_dummy_spikes.h5"), "node_set": "l23pc"}
    cfg["inputs"] = ins
    cfg["output"] = {"output_dir": os.path.join(out, "nest_output"), "spikes_file": "spikes.h5",
                     "overwrite_output_dir": True, "log_file": "log.txt", "quiet_simulator": True}
    cfg["run"]["duration"] = cfg["run"]["tstop"] = float(tstop)
    with open(os.path.join(out, "config_nest_part.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- edges onto PCs ------------------------------------------------------------------
    to_pc = lambda s, t: is_pc[t]  # noqa: E731
    rec = _edge_table(net, "l23_l23", to_pc)
    s_pc = is_pc[rec["source_node_id"]]
    trg = l23_to_pc[rec["target_node_id"]]
    CSR(l23_to_pc[rec["source_node_id"][s_pc]], trg[s_pc], rec["weight"][s_pc], _offsets(rec["delay"][s_pc]),
        rec["port"][s_pc], n_pc).save(os.path.join(out, "pc_to_pc.npz"))
    CSR(l23_to_in[rec["source_node_id"][~s_pc]], trg[~s_pc], rec["weight"][~s_pc], _offsets(rec["delay"][~s_pc]),
        rec["port"][~s_pc], n_in).save(os.path.join(out, "in_to_pc.npz"))
    stats = {"pc_to_pc_edges": int(s_pc.sum()), "in_to_pc_edges": int((~s_pc).sum()),
             "min_offset_pc_to_pc": int(_offsets(rec["delay"][s_pc]).min()),
             "min_offset_in_to_pc": int(_offsets(rec["delay"][~s_pc]).min())}

    # ---- replayed input onto PCs: precomputed current, (T, n_pc) float32, mean over each ms ---
    T = int(round(tstop))
    I_ext = np.lib.format.open_memmap(os.path.join(out, "pc_external_current.npy"), "w+", np.float32, (T, n_pc))
    # per receptor port too (E->E port 1, I->E port 2), for the kernel layer's driving force
    I_ext_e = np.lib.format.open_memmap(os.path.join(out, "pc_external_current_e.npy"), "w+", np.float32, (T, n_pc))
    I_ext_i = np.lib.format.open_memmap(os.path.join(out, "pc_external_current_i.npy"), "w+", np.float32, (T, n_pc))
    bank = PSCBank(n_pc, ring=4 * block_ms + 64)
    streams = []
    for prefix, spikes, pop in (("v1ext_l23", v1ext_spikes, "v1ext"), ("lgn_l23", lgn_file, None), ("bkg_l23", bkg_file, None)):
        e = _edge_table(net, prefix, to_pc)
        n_src = int(e["source_node_id"].max()) + 1
        csr = CSR(e["source_node_id"], l23_to_pc[e["target_node_id"]], e["weight"], _offsets(e["delay"]), e["port"], n_src)
        ids, ts = _read_spikes(spikes, pop)
        keep = ids < n_src
        stamp = np.ceil(ts[keep] / RES - 1e-6).astype(np.int64)  # NEST step at which the spike is emitted
        o = np.argsort(stamp, kind="stable")
        streams.append((csr, ids[keep][o], stamp[o]))
        stats[f"{prefix}_edges_onto_pc"] = int(len(e["weight"]))
        print(f"[prepare] {prefix}: {len(e['weight'])} edges onto PCs, {keep.sum()} spikes")
    nb = 4 * block_ms
    for b0 in range(0, 4 * T, nb):
        for csr, ids, stamp in streams:
            lo, hi = np.searchsorted(stamp, [b0, b0 + nb])
            idx, rep = csr.expand(ids[lo:hi])
            if len(idx):
                # jump at the end of step (stamp + off - 1): the PSC starts at stamp*RES + d_eff
                bank.push(stamp[lo:hi][rep] + csr.off[idx] - 1, csr.trg[idx], csr.w[idx], csr.port[idx])
        for k in range(b0 // 4, min(T, (b0 + nb) // 4)):
            pp = bank.advance_ms(per_port=True)
            I_ext_e[k], I_ext_i[k] = pp[0], pp[1]
            I_ext[k] = pp.sum(0)
        print(f"[prepare] external current {min(T, (b0 + nb) // 4)}/{T} ms", flush=True)
    for z in (I_ext, I_ext_e, I_ext_i):
        z.flush()
    with h5py.File(os.path.join(out, "ids.h5"), "w") as f:
        f["pc_to_l23"] = pc_ids
        f["in_to_l23"] = in_ids
    stats.update(n_pc=n_pc, n_in=n_in, tstop=tstop, mean_external_current_pa=float(I_ext[500:].mean()),
                 prepare_s=time.time() - t_start)
    with open(os.path.join(out, "prepare.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


# ============================================================================================
# run
# ============================================================================================
class NestInterneurons:
    """The GLIF interneurons in NEST (built by BMTK from the prepared config), stepped 1 ms at a time."""

    def __init__(self, config_path, gain, threads=1):
        """``gain``: factor on every PC -> interneuron synapse, a float or {class: factor}
        (class as in ``extract_l23.classify``, e.g. "L2/3 Pvalb")."""
        from .simulate import patch_bmtk_gid_lookup
        patch_bmtk_gid_lookup()
        from bmtk.simulator import pointnet
        import nest
        self.nest = nest
        conf = pointnet.Config.from_json(config_path)
        conf["run"]["n_thread"] = int(threads)
        conf.build_env()
        net = pointnet.PointNetwork.from_config(conf)
        self.sim = pointnet.PointSimulator.from_config(conf, net)
        for mod in self.sim._mods:  # what PointSimulator.run does first: builds the input spike generators
            mod.initialize(self.sim)
        vmap = net._virtual_ids_map["l23pc"]
        n_pc = len(vmap)
        self.pc_gens = np.array([vmap[i] for i in range(n_pc)], np.int64)
        self.pc_nc = nest.NodeCollection(sorted(self.pc_gens.tolist()))
        nest.SetStatus(self.pc_nc, {"spike_times": []})
        in_df = net.get_nodes_df("l23i")
        in_nest = net.gid_map.get_nestids("l23i", np.arange(len(in_df)))
        self.in_nest = np.asarray(in_nest, np.int64)
        if not isinstance(gain, dict):
            gain = {None: float(gain)}
        if any(g != 1.0 for g in gain.values()):
            conns = nest.GetConnections(source=self.pc_nc)
            w = np.asarray(conns.get("weight"), float)
            if None in gain:
                fac = np.full(len(w), gain[None])
            else:
                in_cls = in_df.sort_index()["pop_name"].map(classify).to_numpy()
                lut = np.ones(int(self.in_nest.max()) + 1)
                lut[self.in_nest] = [gain.get(c, 1.0) for c in in_cls]
                fac = lut[np.asarray(conns.get("target"), np.int64)]
            conns.set(weight=w * fac)
        self.nest_to_in = np.full(int(max(self.in_nest.max(), self.pc_gens.max())) + 1, -1, np.int64)
        self.nest_to_in[self.in_nest] = np.arange(len(self.in_nest))
        self.rec = nest.Create("spike_recorder")
        nest.Connect(nest.NodeCollection(sorted(self.in_nest.tolist())), self.rec)
        nest.Prepare()

    def set_pc_spikes(self, pc_idx, t):
        """Give PC generators one spike each at time t (scalar or one per PC); they fire in the next run_ms."""
        if len(pc_idx) == 0:
            return
        t = np.broadcast_to(np.asarray(t, float), np.shape(pc_idx))
        gid = self.pc_gens[pc_idx]
        o = np.argsort(gid)
        gid, t = gid[o], t[o]
        u, first = np.unique(gid, return_index=True)
        groups = np.split(t, first[1:])
        self.nest.SetStatus(self.nest.NodeCollection(u.tolist()), [{"spike_times": sorted(g.tolist())} for g in groups])

    def run_ms(self):
        """Advance 1 ms; return (interneuron index, spike time) of the spikes in this millisecond."""
        nest = self.nest
        nest.Run(1.0)
        ev = self.rec.get("events")
        snd, ts = np.asarray(ev["senders"], np.int64), np.asarray(ev["times"], float)
        nest.Cleanup()
        self.rec.n_events = 0
        nest.Prepare()
        return self.nest_to_in[snd], ts

    def close(self):
        self.nest.Cleanup()


KERNEL = dict(E_e=0.0, E_i=-80.0, V_ref=-55.0)   # kernel-layer synapses: reversal potentials, normalisation (mV)


def run(prep, standard_dir, gain, tstop, out, threads=1, log_every=100, replay_pc=None, probe=0,
        pc_to_in_gain=None, population=None, synapses="kernel", corr=None, front_jit=True):
    """Run the hybrid circuit. ``gain`` multiplies every synapse onto a PC; ``pc_to_in_gain`` every
    PC -> interneuron synapse (float or {class: factor}; None = the same as ``gain``, the PR 9 rule). ``replay_pc``: an l23 spikes file whose PC spikes replace the ANN (a
    check of the loop: with the GLIF network's own PC spikes, the interneurons must reproduce it).
    ``probe``: record the synaptic current of the first ``probe`` PCs at 1 ms.
    ``population``: a population JSON (``population.py``) giving each PC its own fitted model and,
    optionally, its own input scale; without it every PC is the Standard.
    ``synapses``: "kernel" (default since 2026-11-26) sends the per-port alpha PSCs through the kernel layer
    (conductance with driving force, normalised at KERNEL["V_ref"], plus each model's membrane correction; the front
    end sees no current). "current" is the PR 9 path: the summed current through the 64 ms front end.
    ``corr``: calibration.json for the Standard's correction when no population is given.
    ``front_jit``: XLA-compile the CNN-LSTM front end (the default). On some GPUs the non-XLA path,
    which can use cuDNN's fused LSTM, is faster; both give the same model."""
    sys.path.insert(0, os.path.abspath(standard_dir))
    import standard_model as sm
    from .pc_layer import BatchedStandard
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    meta = json.load(open(os.path.join(prep, "prepare.json")))
    n_pc, n_in = meta["n_pc"], meta["n_in"]
    T = int(round(tstop))
    kernel = synapses == "kernel"
    I_ext = np.load(os.path.join(prep, "pc_external_current.npy"), mmap_mode="r")
    if kernel:
        I_ext_e = np.load(os.path.join(prep, "pc_external_current_e.npy"), mmap_mode="r")
        I_ext_i = np.load(os.path.join(prep, "pc_external_current_i.npy"), mmap_mode="r")
    assert T <= I_ext.shape[0], "prepare with a longer --tstop"
    pc2pc, in2pc = CSR.load(os.path.join(prep, "pc_to_pc.npz")), CSR.load(os.path.join(prep, "in_to_pc.npz"))
    with h5py.File(os.path.join(prep, "ids.h5")) as f:
        pc_to_l23, in_to_l23 = f["pc_to_l23"][()], f["in_to_l23"][()]

    if pc_to_in_gain is None:
        pc_to_in_gain = gain
    nest_part = NestInterneurons(os.path.join(prep, "config_nest_part.json"), pc_to_in_gain, threads)
    bank = PSCBank(n_pc)
    if replay_pc:
        rid, rts = _read_spikes(replay_pc, "l23")
        l23_to_pc = np.full(max(rid.max(), pc_to_l23.max()) + 1, -1, np.int64); l23_to_pc[pc_to_l23] = np.arange(n_pc)
        m = l23_to_pc[rid] >= 0
        rid, rts = l23_to_pc[rid[m]], rts[m]
        o = np.argsort(rts, kind="stable"); rid, rts = rid[o], rts[o]
    in_scale = None
    if population and not replay_pc:
        from .population import MixedStandard, read_population
        models, assign, in_scale = read_population(population)
        assert len(assign) == n_pc, "population assignment must cover every PC of the prep"
        I0 = gain * np.asarray(I_ext[0], np.float32) * (1.0 if in_scale is None else in_scale)
        cells = MixedStandard(models, assign, standard_dir, I_first=I0, jit=front_jit, kernel=KERNEL if kernel else None)
    elif not replay_pc:
        from .population import load_corr
        cells = BatchedStandard(sm.load(standard_dir), n_pc, I_first=gain * np.asarray(I_ext[0], np.float32), jit=front_jit,
                                kernel=dict(KERNEL, corr=load_corr(corr)) if kernel else None)
    probe_I = np.zeros((T, probe), np.float32)
    pc_spk, in_spk = [], []
    I_mean = np.zeros(T, np.float32)
    timers = dict(pc=0.0, nest=0.0, route=0.0)
    for k in range(T):
        t0 = time.time()
        pp = bank.advance_ms(per_port=True)
        if kernel:
            I_e = gain * (np.asarray(I_ext_e[k], np.float64) + pp[0])
            I_i = gain * (np.asarray(I_ext_i[k], np.float64) + pp[1])
            if in_scale is not None:
                I_e, I_i = I_e * in_scale, I_i * in_scale
            I_k = I_e + I_i
        else:
            I_k = gain * (np.asarray(I_ext[k], np.float64) + pp.sum(0))
            if in_scale is not None:
                I_k = I_k * in_scale
        I_mean[k] = I_k.mean()
        probe_I[k] = I_k[:probe]
        if replay_pc:
            lo, hi = np.searchsorted(rts, [k + 1e-9, k + 1 + 1e-9])  # spikes in (k, k+1]
            fired, t_spk = rid[lo:hi], rts[lo:hi]
        else:
            fired = np.flatnonzero(cells.step_kernel(I_e, I_i) if kernel else cells.step(I_k))
            t_spk = np.full(len(fired), k + 1.0)
        t1 = time.time()
        nest_part.set_pc_spikes(fired, t_spk)
        idx_in, ts_in = nest_part.run_ms()
        t2 = time.time()
        # route spikes onto PCs: PC spikes stamped at k+1 -> 0.25 ms step 4(k+1); jump at end of step stamp+off-1
        if len(fired):
            idx, rep = pc2pc.expand(fired)
            stamp = np.round(t_spk / RES).astype(np.int64)
            bank.push_sparse(stamp[rep] + pc2pc.off[idx] - 1, pc2pc.trg[idx], pc2pc.w[idx], pc2pc.port[idx])
            pc_spk.append(np.c_[fired, t_spk])
        if len(idx_in):
            stamp = np.round(ts_in / RES).astype(np.int64)
            idx, rep = in2pc.expand(idx_in)
            bank.push_sparse(stamp[rep] + in2pc.off[idx] - 1, in2pc.trg[idx], in2pc.w[idx], in2pc.port[idx])
            in_spk.append(np.c_[idx_in, ts_in])
        t3 = time.time()
        timers["pc"] += t1 - t0; timers["nest"] += t2 - t1; timers["route"] += t3 - t2
        if (k + 1) % log_every == 0:
            n_p = sum(len(x) for x in pc_spk); n_i = sum(len(x) for x in in_spk)
            print(f"[run gain={gain:.3f}] {k + 1}/{T} ms  PC {n_p / n_pc / ((k + 1) / 1000):.2f} Hz  "
                  f"IN {n_i / n_in / ((k + 1) / 1000):.2f} Hz  mean I {I_mean[k]:.1f} pA  "
                  f"{(time.time() - t_start) / (k + 1):.2f} s/ms", flush=True)
    nest_part.close()
    P = np.concatenate(pc_spk) if pc_spk else np.zeros((0, 2))
    Q = np.concatenate(in_spk) if in_spk else np.zeros((0, 2))
    ids = np.r_[pc_to_l23[P[:, 0].astype(np.int64)], in_to_l23[Q[:, 0].astype(np.int64)]]
    ts = np.r_[P[:, 1], Q[:, 1]]
    sio.write_spikes(os.path.join(out, "spikes.h5"), "l23", ids, ts)
    np.save(os.path.join(out, "pc_mean_input_current.npy"), I_mean)
    if probe:
        np.save(os.path.join(out, "probe_current.npy"), probe_I)
    info = {"gain": gain, "pc_to_in_gain": pc_to_in_gain, "tstop": tstop, "synapses": synapses,
            "kernel": KERNEL if kernel else None, "corr": corr,
            "population": os.path.abspath(population) if population else None, "front_jit": bool(front_jit),
            "wall_s": time.time() - t_start, "timers_s": timers,
            "pc_rate_hz": len(P) / n_pc / (T / 1000), "interneuron_rate_hz": len(Q) / n_in / (T / 1000),
            "pc_spikes": int(len(P)), "interneuron_spikes": int(len(Q))}
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    return info


def parse_gain(text):
    """"0.9" -> 0.9; "Pvalb=1.1,Sst=1.0,Htr3a=0.9" -> {"L2/3 Pvalb": 1.1, ...}; None -> None."""
    if text is None or isinstance(text, (int, float, dict)):
        return text
    if "=" not in text:
        return float(text)
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        k = k.strip()
        out[k if k.startswith("L2/3 ") else f"L2/3 {k}"] = float(v)
    return out


def format_gain(g):
    if g is None or not isinstance(g, dict):
        return None if g is None else str(g)
    return ",".join(f"{k.replace('L2/3 ', '')}={v}" for k, v in g.items())


def _run_subprocess(prep, standard_dir, gain, tstop, out, threads, pc_to_in_gain=None, population=None,
                    synapses="kernel", corr=None):
    """One run per process: NEST can only be built once per process."""
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ); env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-m", "circuit.hybrid", "run", "--prep", prep, "--standard-dir", standard_dir,
                    "--gain", str(gain), "--tstop", str(tstop), "--out", out, "--threads", str(threads)]
                   + (["--pc-to-in-gain", format_gain(pc_to_in_gain)] if pc_to_in_gain is not None else [])
                   + (["--population", population] if population else [])
                   + ["--synapses", synapses] + (["--corr", corr] if corr else []),
                   check=True, env=env)
    return json.load(open(os.path.join(out, "run.json")))


def reference_rate(reference, prep, t0=0.0, t1=3000.0):
    with h5py.File(os.path.join(prep, "ids.h5")) as f:
        pc_to_l23 = f["pc_to_l23"][()]
    ids, ts = _read_spikes(reference, "l23")
    is_pc = np.zeros(max(ids.max(), pc_to_l23.max()) + 1, bool); is_pc[pc_to_l23] = True
    m = (ts >= t0) & (ts < t1) & is_pc[ids]
    return m.sum() / len(pc_to_l23) / ((t1 - t0) / 1000)


def tune(prep, standard_dir, reference, out, tstop=1500.0, gains=(1.0,), max_iter=6, tol=0.1, threads=1,
         pc_to_in_gain=None, population=None, target=None, synapses="kernel", corr=None):
    """Search the gain onto PCs until the PC rate over 0..tstop is within ``tol`` of the GLIF network's.
    ``pc_to_in_gain`` stays fixed during the search (None = equal to the searched gain, the PR 9 rule).

    Brackets the target, then bisects in log(gain)."""
    os.makedirs(out, exist_ok=True)
    if target is None:  # default: the GLIF network's PC rate
        target = reference_rate(reference, prep, 0.0, tstop)
    hist = []
    path = os.path.join(out, "tune.json")
    if os.path.exists(path):
        hist = json.load(open(path))["history"]
    for fn in sorted(glob.glob(os.path.join(out, "gain_*", "run.json"))):
        r = json.load(open(fn))
        if r["tstop"] == tstop and all(abs(h["gain"] - r["gain"]) > 1e-6 for h in hist):
            hist.append({"gain": r["gain"], "pc_rate_hz": r["pc_rate_hz"], "interneuron_rate_hz": r["interneuron_rate_hz"]})
    queue = [g for g in gains if all(abs(h["gain"] - g) > 1e-6 for h in hist)]
    for _ in range(max_iter):
        done = {round(h["gain"], 4) for h in hist}
        if queue:
            g = queue.pop(0)
        else:
            lo = [h for h in hist if h["pc_rate_hz"] < target]
            hi = [h for h in hist if h["pc_rate_hz"] > target]
            if lo and hi:
                a = max(lo, key=lambda h: h["gain"]); b = min(hi, key=lambda h: h["gain"])
                # interpolate in log(rate) vs log(gain)
                ra, rb = max(a["pc_rate_hz"], 1e-3), b["pc_rate_hz"]
                f = (np.log(target) - np.log(ra)) / (np.log(rb) - np.log(ra))
                g = float(np.exp(np.log(a["gain"]) + np.clip(f, 0.2, 0.8) * (np.log(b["gain"]) - np.log(a["gain"]))))
            else:  # not bracketed yet: step by a damped power of the rate ratio
                h = min(hi, key=lambda h: h["gain"]) if hi else max(lo, key=lambda h: h["gain"])
                g = h["gain"] * float(np.clip((target / max(h["pc_rate_hz"], 1e-3)) ** 0.3, 0.7, 1.4))
        if round(g, 4) in done:
            continue
        r = _run_subprocess(prep, standard_dir, g, tstop, os.path.join(out, f"gain_{g:.4f}"), threads, pc_to_in_gain,
                            population, synapses=synapses, corr=corr)
        hist.append({"gain": g, "pc_rate_hz": r["pc_rate_hz"], "interneuron_rate_hz": r["interneuron_rate_hz"]})
        json.dump({"target_pc_rate_hz": target, "tstop": tstop, "history": hist}, open(path, "w"), indent=2)
        print(f"[tune] gain {g:.4f}: PC {r['pc_rate_hz']:.3f} Hz (target {target:.3f})", flush=True)
        if abs(r["pc_rate_hz"] / target - 1) <= tol:
            break
    best = min(hist, key=lambda h: abs(np.log(max(h["pc_rate_hz"], 1e-3) / target)))
    json.dump({"target_pc_rate_hz": target, "tstop": tstop, "history": hist, "best": best}, open(path, "w"), indent=2)
    return best


# ============================================================================================
# report
# ============================================================================================
def _gain_label(run_json):
    g, gi = run_json["gain"], run_json.get("pc_to_in_gain", run_json["gain"])
    if isinstance(gi, dict):
        gi = "/".join(f"{v:.2f}" for v in gi.values())
    elif gi is not None and abs(float(gi) - g) > 1e-9:
        gi = f"{float(gi):.2f}"
    else:
        return f"gain {g:.2f}"
    return f"gain {g:.2f}, PC→IN {gi}"


def report(prep, extract_dir, reference, hybrid_spikes, out, tune_json=None, tstop=3000.0, gray_until=500.0,
           previous_spikes=None):
    """Overview figure (rates per class, E rate distribution, raster) plus a summary figure:
    the gain search and each PC's rate in the hybrid vs the GLIF network. ``previous_spikes``: an
    earlier hybrid run (with its run.json next to it) shown as a third run for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_overview import overview, CLASSES, CLASS_COLORS
    os.makedirs(out, exist_ok=True)
    rj = json.load(open(os.path.join(os.path.dirname(hybrid_spikes), "run.json")))
    gain = rj["gain"]
    label = f"ANN PCs ({_gain_label(rj)})"
    runs = [("GLIF network", reference, "glif")]
    if previous_spikes:
        pj = json.load(open(os.path.join(os.path.dirname(previous_spikes), "run.json")))
        runs.append((f"ANN PCs, PR 9 ({_gain_label(pj)})", previous_spikes, "previous"))
    runs.append((label, hybrid_spikes, "hybrid"))
    stats = overview(extract_dir, [(lb, p) for lb, p, _ in runs], os.path.join(out, "overview.png"), tstop, gray_until,
                     raster_run=len(runs) - 1)
    net = os.path.join(extract_dir, "network")
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    cls = nodes["pop_name"].map(classify).to_numpy()
    n = len(nodes)
    rates = {}
    for _, path, key in runs:
        ids, ts = _read_spikes(path, "l23")
        m = ts < tstop
        r_all = np.bincount(ids[m], minlength=n)[:n] / (tstop / 1000)
        g = (ts >= gray_until) & m
        r_gr = np.bincount(ids[g], minlength=n)[:n] / ((tstop - gray_until) / 1000)
        rates[key] = (r_all, r_gr)
    per_class = {c: {f"{k}_hz": float(rates[k][0][cls == c].mean()) for _, _, k in runs} for c in CLASSES}
    e = cls == "L2/3 E"
    a, b = rates["glif"][1][e], rates["hybrid"][1][e]
    r_pc = float(np.corrcoef(a, b)[0, 1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(a, b).correlation)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(wspace=0.35))
    ax = axs[0]
    if tune_json and os.path.exists(tune_json):
        tj = json.load(open(tune_json))
        h = sorted(tj["history"], key=lambda x: x["gain"])
        ax.plot([x["gain"] for x in h], [x["pc_rate_hz"] for x in h], "o-", color=CLASS_COLORS["L2/3 E"], label="ANN PCs")
        ax.plot([x["gain"] for x in h], [x["interneuron_rate_hz"] for x in h], "s--", color="#eb6834", label="GLIF interneurons")
        ax.axhline(tj["target_pc_rate_hz"], color="#222222", lw=1, ls=":", label="GLIF network PCs")
        fixed = rj.get("pc_to_in_gain")
        xl = ("gain on synapses onto and from PCs" if fixed is None or fixed == gain
              else f"gain on synapses onto PCs (PC→IN fixed at {fixed})")
        ax.set(xlabel=xl, ylabel=f"mean rate 0-{tj['tstop']:.0f} ms (Hz)", title="Gain search")
        ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    x = np.arange(len(CLASSES))
    wd = 0.8 / len(runs)
    for j, (lb, _, k) in enumerate(runs):
        xs = x + (j - (len(runs) - 1) / 2) * wd
        vals = [per_class[c][f"{k}_hz"] for c in CLASSES]
        if k == "glif":
            ax.bar(xs, vals, wd, color="#222222", label=lb)
        elif k == "previous":
            ax.bar(xs, vals, wd, color="white", edgecolor=[CLASS_COLORS[c] for c in CLASSES], hatch="///", lw=0.8, label=lb)
        else:
            ax.bar(xs, vals, wd, color=[CLASS_COLORS[c] for c in CLASSES], label=lb)
        if k != "glif":
            for xi, v, c in zip(xs, vals, CLASSES):
                ax.text(xi, v, f"{100 * (v / per_class[c]['glif_hz'] - 1):+.0f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, [c.replace("L2/3 ", "") for c in CLASSES])
    ax.set(ylabel=f"mean rate 0-{tstop:.0f} ms (Hz)", title="Rates per class")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax = axs[2]
    jit = np.random.default_rng(0).uniform(-0.15, 0.15, (2, e.sum())) / ((tstop - gray_until) / 1000)
    ax.scatter(a + 0.1 + jit[0], b + 0.1 + jit[1], s=1, alpha=0.15, color=CLASS_COLORS["L2/3 E"], lw=0)
    lim = [0.08, max(a.max(), b.max()) * 1.3]
    ax.plot(lim, lim, color="0.5", lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="GLIF network rate + 0.1 (Hz)",
           ylabel="hybrid rate + 0.1 (Hz)", title=f"Each PC during the grating (r = {r_pc:.2f}, Spearman {rho:.2f})")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(10)
    fig.savefig(os.path.join(out, "hybrid_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    res = {"gain": gain, "pc_to_in_gain": rj.get("pc_to_in_gain", gain), "per_class_rates_0_tstop": per_class, "distribution_grating": stats,
           "pc_rate_correlation_pearson": r_pc, "pc_rate_correlation_spearman": rho}
    with open(os.path.join(out, "hybrid_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("prepare")
    a.add_argument("--config", required=True); a.add_argument("--extract-dir", required=True)
    a.add_argument("--v1ext-spikes", required=True); a.add_argument("--out", required=True)
    a.add_argument("--tstop", type=float, default=3000.0)
    b = sp.add_parser("run")
    b.add_argument("--prep", required=True); b.add_argument("--standard-dir", required=True)
    b.add_argument("--gain", type=float, default=1.0); b.add_argument("--tstop", type=float, default=3000.0)
    b.add_argument("--out", required=True); b.add_argument("--threads", type=int, default=1)
    b.add_argument("--replay-pc", help="l23 spikes file whose PC spikes replace the ANN (loop check)")
    b.add_argument("--probe", type=int, default=0, help="record the input current of the first N PCs")
    b.add_argument("--pc-to-in-gain", help="factor on PC -> interneuron synapses: a number or Pvalb=..,Sst=..,Htr3a=.. "
                   "(default: same as --gain)")
    b.add_argument("--population", help="population JSON: a fitted model (and input scale) per PC")
    b.add_argument("--synapses", choices=("kernel", "current"), default="kernel",
                   help="kernel: synaptic kernel layer (default); current: summed PSC current through the front end (PR 9)")
    b.add_argument("--corr", help="calibration.json with the Standard's membrane correction (kernel mode, no population)")
    b.add_argument("--no-front-jit", action="store_true", help="run the CNN-LSTM front end without XLA (try on GPU)")
    b.add_argument("--log-every", type=int, default=100, help="print progress every N ms")
    c = sp.add_parser("tune")
    c.add_argument("--prep", required=True); c.add_argument("--standard-dir", required=True)
    c.add_argument("--reference", required=True); c.add_argument("--out", required=True)
    c.add_argument("--tstop", type=float, default=1500.0)
    c.add_argument("--gains", type=float, nargs="+", default=[1.0])
    c.add_argument("--max-iter", type=int, default=6); c.add_argument("--tol", type=float, default=0.1)
    c.add_argument("--threads", type=int, default=1)
    c.add_argument("--pc-to-in-gain", help="fixed factor on PC -> interneuron synapses (default: same as the searched gain)")
    c.add_argument("--population", help="population JSON: a fitted model (and input scale) per PC")
    c.add_argument("--synapses", choices=("kernel", "current"), default="kernel",
                   help="kernel: synaptic kernel layer (default); current: summed PSC current through the front end (PR 9)")
    c.add_argument("--corr", help="calibration.json with the Standard's membrane correction (kernel mode, no population)")
    c.add_argument("--target", type=float, help="target mean PC rate (Hz) instead of the GLIF network's")
    d = sp.add_parser("report")
    d.add_argument("--prep", required=True); d.add_argument("--extract-dir", required=True)
    d.add_argument("--reference", required=True); d.add_argument("--hybrid-spikes", required=True)
    d.add_argument("--out", required=True); d.add_argument("--tune-json")
    d.add_argument("--tstop", type=float, default=3000.0)
    d.add_argument("--previous-spikes", help="an earlier hybrid run's spikes.h5 to show alongside")
    x = p.parse_args(argv)
    if x.cmd == "prepare":
        prepare(x.config, x.extract_dir, x.v1ext_spikes, x.out, x.tstop)
    elif x.cmd == "run":
        run(x.prep, x.standard_dir, x.gain, x.tstop, x.out, x.threads, replay_pc=x.replay_pc, probe=x.probe,
            pc_to_in_gain=parse_gain(x.pc_to_in_gain), population=x.population, synapses=x.synapses, corr=x.corr,
            front_jit=not x.no_front_jit, log_every=x.log_every)
    elif x.cmd == "report":
        report(x.prep, x.extract_dir, x.reference, x.hybrid_spikes, x.out, x.tune_json, x.tstop,
               previous_spikes=x.previous_spikes)
    else:
        print(json.dumps(tune(x.prep, x.standard_dir, x.reference, x.out, x.tstop, x.gains, x.max_iter, x.tol,
                              x.threads, parse_gain(x.pc_to_in_gain), x.population, x.target, x.synapses, x.corr), indent=2))


if __name__ == "__main__":
    main()
