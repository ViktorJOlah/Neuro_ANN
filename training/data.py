"""Loading, cleaning, normalizing and windowing NEURON training traces.

Design choices (each fixes a failure mode of the original scripts):

* NEURON ``Matrix.fprint`` header lines ("20001 59") are detected by their
  field count, not by the text prefix '200', and nothing is ever appended to a
  file, so re-running cannot duplicate the dataset.
* Data are kept as whole traces. Windows never cross a trace boundary, and the
  train / val / test split is by trace, so validation windows never overlap
  training windows.
* One normalization is fit on the *training traces of all pools* and reused by
  every stage, so the subthreshold and spiking stages see identically scaled
  inputs. It is saved to JSON next to the model.
* All preprocessing is vectorized numpy (the original per-row pandas loops are
  silent no-ops under pandas copy-on-write and take minutes on 4M rows).
* Everything is float32 (the original validation set was float16, which rounds
  normalized Vm to ~5e-4).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- IO

def _count_fields(line: str) -> int:
    return len(line.split())


def read_neuron_txt(path: str, max_rows: int | None = None):
    """Read a NEURON text output file.

    Returns ``(data, header_rows)`` where ``data`` is a float32 array of data
    rows only and ``header_rows`` is a list of ``(position_in_data, nrow)`` for
    every ``Matrix.fprint`` header found (empty if the file has no headers).
    """
    with open(path) as f:
        first = [f.readline() for _ in range(3)]
    first = [l for l in first if l.strip()]
    if not first:
        raise ValueError(f"{path} is empty")
    counts = [_count_fields(l) for l in first]
    has_header = len(counts) > 1 and counts[0] == 2 and counts[1] != 2
    ncols = counts[1] if has_header else counts[0]
    if has_header and int(float(first[0].split()[1])) != ncols:
        log.warning("%s: header says %s columns but rows have %d",
                    path, first[0].split()[1], ncols)

    # one extra name absorbs the trailing space NEURON writes after each row
    df = pd.read_csv(path, sep=" ", header=None, names=list(range(ncols + 1)),
                     usecols=list(range(ncols)), skipinitialspace=True,
                     dtype=np.float32, nrows=max_rows, engine="c")
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if ncols > 2:
        is_header = np.isnan(arr[:, 2])
    else:
        is_header = np.zeros(len(arr), dtype=bool)
    header_pos = np.flatnonzero(is_header)
    headers = []
    for k, pos in enumerate(header_pos):
        headers.append((int(pos - k), int(arr[pos, 0])))
    data = arr[~is_header]
    bad = ~np.isfinite(data).all(axis=1)
    if bad.any():
        raise ValueError(f"{path}: {bad.sum()} data rows contain NaN/inf "
                         f"(first at data row {np.flatnonzero(bad)[0]})")
    return np.ascontiguousarray(data), headers


def split_into_traces(data: np.ndarray, headers, trace_length: int | None, path: str = ""):
    """Cut a file's rows into equal-length traces."""
    if trace_length is None:
        if headers:
            lengths = {n for _, n in headers}
            if len(lengths) != 1:
                raise ValueError(f"{path}: traces of different lengths {sorted(lengths)}; set data.trace_length")
            trace_length = lengths.pop()
        else:
            log.warning("%s has no NEURON headers and data.trace_length is not set; "
                        "treating the whole file as one trace", path)
            trace_length = len(data)
    n = len(data) // trace_length
    if n == 0:
        raise ValueError(f"{path}: {len(data)} rows is shorter than one trace ({trace_length})")
    if len(data) % trace_length:
        log.warning("%s: dropping %d trailing rows that do not fill a trace (trace_length=%d)",
                    path, len(data) % trace_length, trace_length)
    return [data[i * trace_length:(i + 1) * trace_length] for i in range(n)], trace_length


# ------------------------------------------------------------------ preprocess

def preprocess_raw(x: np.ndarray, cfg_pre: dict, vm_col: int) -> np.ndarray:
    """Raw-unit cleaning (in place on a copy). Mirrors the original intent."""
    x = x.copy()
    for c in cfg_pre.get("negate_columns", []) or []:
        x[:, c] = -x[:, c]
    for c, (thr, div) in (cfg_pre.get("compress") or {}).items():
        c = int(c)
        col = x[:, c]
        over = col > thr
        col[over] = (col[over] - thr) / div + thr
    if cfg_pre.get("vm_clip_max") is not None:
        np.minimum(x[:, vm_col], cfg_pre["vm_clip_max"], out=x[:, vm_col])
    return x


def detect_spikes(vm: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of upward threshold crossings in a 1-D voltage trace."""
    above = vm >= threshold
    return np.flatnonzero(above[1:] & ~above[:-1]) + 1


@dataclass
class Scaler:
    """Min-max scaling of the state columns to [0, 1], fit on training traces."""
    state_columns: list
    mins: list
    maxs: list
    preprocess: dict = field(default_factory=dict)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        for c, lo, hi in zip(self.state_columns, self.mins, self.maxs):
            x[:, c] = (x[:, c] - lo) / (hi - lo)
        return x

    def inverse_state(self, s: np.ndarray) -> np.ndarray:
        """Inverse transform of an (..., n_state) array of normalized predictions."""
        lo = np.asarray(self.mins, dtype=np.float32)
        hi = np.asarray(self.maxs, dtype=np.float32)
        return s * (hi - lo) + lo

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Scaler":
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------- dataset

class TraceDataset:
    """All traces of a run, concatenated, with pool / split bookkeeping."""

    def __init__(self, cfg: dict, scaler: Scaler | None = None, split_assignment: dict | None = None):
        d = cfg["data"]
        self.cfg = cfg
        self.state_columns = list(d["state_columns"])
        self.vm_col = int(d["vm_column"])
        if self.vm_col not in self.state_columns:
            raise ValueError("data.vm_column must be one of data.state_columns")
        self.time_steps = int(cfg["model"]["time_steps"])
        self.skip = int(d["skip_trace_start"])
        rng = np.random.default_rng(cfg["seed"])

        raw_traces, pools, sources = [], [], []
        self.trace_length = d["trace_length"]
        n_cols = None
        for pool, files in d["files"].items():
            for path in files:
                arr, headers = read_neuron_txt(path, d["max_rows_per_file"])
                traces, tl = split_into_traces(arr, headers, self.trace_length, path)
                if self.trace_length is None:
                    self.trace_length = tl
                elif tl != self.trace_length:
                    raise ValueError(f"{path}: trace length {tl} != {self.trace_length}")
                if n_cols is None:
                    n_cols = arr.shape[1]
                elif arr.shape[1] != n_cols:
                    raise ValueError(f"{path}: {arr.shape[1]} columns, expected {n_cols}")
                for i, t in enumerate(traces):
                    raw_traces.append(preprocess_raw(t, d["preprocess"], self.vm_col))
                    pools.append(pool)
                    sources.append(f"{os.path.basename(path)}#{i}")
        if not raw_traces:
            raise ValueError("No data files given in data.files")
        self.n_features = n_cols
        self.n_state = len(self.state_columns)

        # spikes on raw vm, then pool assignment
        self.spikes = [detect_spikes(t[:, self.vm_col], d["spike_threshold_mv"]) for t in raw_traces]
        if set(d["files"]) == {"all"}:
            pools = ["spiking" if len(s) >= d["min_spikes_spiking"] else "subthreshold"
                     for s in self.spikes]
        self.trace_pool = np.array(pools)
        self.trace_source = sources
        self.pools = sorted(set(pools))

        # split by trace, within each pool
        n = len(raw_traces)
        self.trace_split = np.empty(n, dtype=object)
        if split_assignment is not None:
            by_src = {s: k for k, ids in split_assignment.items() for s in ids}
            for i, s in enumerate(sources):
                if s not in by_src:
                    raise ValueError(f"Trace {s} is not in the saved split; data files changed?")
                self.trace_split[i] = by_src[s]
        else:
            for pool in self.pools:
                ids = np.flatnonzero(self.trace_pool == pool)
                ids = ids[rng.permutation(len(ids))]
                n_val = int(round(d["split"]["val"] * len(ids)))
                n_test = int(round(d["split"]["test"] * len(ids)))
                if len(ids) >= 3:
                    n_val, n_test = max(n_val, 1), max(n_test, 1)
                n_train = len(ids) - n_val - n_test
                if n_train < 1:
                    raise ValueError(f"Pool '{pool}' has only {len(ids)} traces; need more for a split")
                self.trace_split[ids[:n_train]] = "train"
                self.trace_split[ids[n_train:n_train + n_val]] = "val"
                self.trace_split[ids[n_train + n_val:]] = "test"

        # normalization, fit on training traces of all pools
        if scaler is None:
            train = [t for t, s in zip(raw_traces, self.trace_split) if s == "train"]
            stacked = np.concatenate([t[:, self.state_columns] for t in train])
            mins, maxs = stacked.min(axis=0), stacked.max(axis=0)
            if np.any(maxs - mins <= 0):
                raise ValueError("A state column is constant in the training data")
            scaler = Scaler(self.state_columns, mins.tolist(), maxs.tolist(), dict(d["preprocess"]))
        self.scaler = scaler

        self.lengths = np.array([len(t) for t in raw_traces])
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.X = np.concatenate([scaler.transform(t) for t in raw_traces]).astype(np.float32)
        del raw_traces

        # per-row "near a spike" mask, used for the spike-weighted loss
        w = int(cfg["training"]["spike_window"])
        self.near_spike = np.zeros(len(self.X), dtype=np.float32)
        for start, length, sp in zip(self.starts, self.lengths, self.spikes):
            for s in sp:
                lo, hi = max(s - w, 0), min(s + w + 1, length)
                self.near_spike[start + lo:start + hi] = 1.0

        self._log_summary()

    # ------------------------------------------------------------------ helpers
    def _log_summary(self):
        for pool in self.pools:
            parts = []
            for split in SPLITS:
                m = (self.trace_pool == pool) & (self.trace_split == split)
                n_sp = sum(len(self.spikes[i]) for i in np.flatnonzero(m))
                parts.append(f"{split}={m.sum()} traces/{n_sp} spikes")
            log.info("pool %-12s %s", pool, ", ".join(parts))
        log.info("trace_length=%d, features=%d, state columns=%s",
                 self.trace_length, self.n_features, self.state_columns)

    def trace_ids(self, pool: str | None, split: str) -> np.ndarray:
        m = self.trace_split == split
        if pool is not None:
            m &= self.trace_pool == pool
        return np.flatnonzero(m)

    def window_starts(self, pool: str | None, split: str, horizon: int = 1) -> np.ndarray:
        """Global row index of every valid window start.

        A window is ``time_steps`` input rows followed by ``horizon`` target rows,
        all inside one trace.
        """
        T = self.time_steps
        out = []
        for i in self.trace_ids(pool, split):
            a, L = self.starts[i], self.lengths[i]
            last = L - T - horizon
            if last >= self.skip:
                out.append(np.arange(a + self.skip, a + last + 1, dtype=np.int64))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def split_assignment(self) -> dict:
        return {s: [src for src, sp in zip(self.trace_source, self.trace_split) if sp == s]
                for s in SPLITS}

    def save_metadata(self, run_dir: str) -> None:
        self.scaler.to_json(os.path.join(run_dir, "scaler.json"))
        with open(os.path.join(run_dir, "data_split.json"), "w") as f:
            json.dump({"trace_length": int(self.trace_length),
                       "n_features": int(self.n_features),
                       "pools": {src: str(p) for src, p in zip(self.trace_source, self.trace_pool)},
                       "splits": self.split_assignment()}, f, indent=2)


class WindowSampler:
    """Draws epochs of window starts as a mixture of the subthreshold and spiking pools."""

    def __init__(self, ds: TraceDataset, split: str, horizon: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.by_pool = {p: ds.window_starts(p, split, horizon) for p in ds.pools}

    def sample(self, n: int, spiking_fraction: float) -> np.ndarray:
        sub = self.by_pool.get("subthreshold", np.zeros(0, np.int64))
        spk = self.by_pool.get("spiking", np.zeros(0, np.int64))
        other = [v for k, v in self.by_pool.items() if k not in ("subthreshold", "spiking")]
        if other and not len(sub) and not len(spk):  # custom pool names: sample uniformly
            allw = np.concatenate(other)
            return self.rng.choice(allw, n)
        p = float(np.clip(spiking_fraction, 0.0, 1.0))
        if not len(spk):
            if p > 0:
                raise ValueError("Stage asks for spiking windows but the spiking pool is empty")
            return self.rng.choice(sub, n)
        if not len(sub):
            if p < 1:
                raise ValueError("Stage asks for subthreshold windows but that pool is empty")
            return self.rng.choice(spk, n)
        n_spk = self.rng.binomial(n, p)
        idx = np.concatenate([self.rng.choice(sub, n - n_spk), self.rng.choice(spk, n_spk)])
        return idx[self.rng.permutation(n)]
