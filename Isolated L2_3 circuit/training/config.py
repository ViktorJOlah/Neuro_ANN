"""Configuration handling for the ANN training pipeline.

A run is described by one YAML (or JSON) file. Anything not given in the file
falls back to DEFAULTS below, so a config only needs to list what differs.
See training/configs/*.yaml for complete, commented examples.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is on Colab, JSON still works
    yaml = None


DEFAULTS: dict[str, Any] = {
    "run_name": "run",
    "output_dir": "runs",
    "seed": 1234,

    "data": {
        # Each pool is a list of NEURON output files. "subthreshold" and "spiking"
        # are the names the stages refer to. If only "all" is given, traces are
        # assigned to pools automatically by spike count (see min_spikes_spiking).
        "files": {"all": []},
        # Rows per trace (single compartment: 10001, L2/3 calibrate.hoc: 20001).
        # null = infer it from the "nrow ncol" header lines NEURON writes.
        "trace_length": None,
        "max_rows_per_file": None,       # cap rows per file (memory / smoke tests)
        "column_names": None,            # optional list, used only for logging
        # Columns predicted by the ANN and fed back during closed-loop simulation,
        # by index. Single compartment: [0, 1, 2] (vm, ik, ina). L2/3: [0].
        "state_columns": [0],
        "vm_column": 0,
        # Raw-unit preprocessing, applied before normalization (vectorized; the
        # original row-by-row loops are silent no-ops on pandas >= 2 / CoW).
        "preprocess": {
            "vm_clip_max": None,          # e.g. -20.0 for L2/3 (clip spike peaks)
            "negate_columns": [],         # e.g. [2] to flip the sign of ina
            # piecewise compression of large outliers: {col: [threshold, divisor]}
            # single compartment: {1: [0.05, 75], 2: [0.1, 190]}
            "compress": {},
        },
        # Spike detection on raw vm (mV): upward crossing of this threshold.
        "spike_threshold_mv": -20.0,
        "min_spikes_spiking": 1,          # auto-split: traces with >= this many spikes are "spiking"
        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        "skip_trace_start": 0,            # drop this many initial rows of each trace (transient)
        "max_val_windows": 20000,         # fixed validation subset per pool
    },

    "model": {
        "architecture": "cnn_lstm_multi",  # cnn_lstm_multi (L2/3) | cnn_lstm_single (ball)
        "time_steps": 128,
        "dropout": "standard",             # standard | alpha | none
        "init_weights": None,              # optional .h5 of a previously trained model
    },

    "training": {
        "batch_size": 64,
        "optimizer": "adam",               # adam | nadam | adamax
        "clipnorm": 1.0,
        "loss": "mae",                     # mae | mse
        "state_loss_weights": None,        # per state column, e.g. [1, 0.7, 0.7]
        "steps_per_epoch": 2000,
        "val_steps": None,                 # None = whole fixed validation subset
        "early_stopping_patience": 20,
        "plateau_patience": 8,             # epochs without improvement before LR *= plateau_factor
        "plateau_factor": 0.5,
        "min_lr": 1e-6,
        "min_delta": 1e-6,
        # Instability guard: if an epoch ends with a non-finite loss, or val loss
        # above blowup_factor x the best so far, restore the best weights,
        # reset the optimizer, multiply the LR by lr_backoff and retry.
        "blowup_factor": 5.0,
        "lr_backoff": 0.5,
        "max_retries": 3,
        # Spike-weighted loss: samples whose target lies within +/- spike_window
        # rows of a detected spike get weight spike_loss_weight (per stage).
        "spike_window": 3,
        # ---- opt-in closed-loop robustness (change model behaviour) ----
        "state_noise_std": 0.0,            # Gaussian noise on fed-back state channels of the input window
        "rollout_steps": 0,                # >0: add a k-step closed-loop rollout loss
        "rollout_loss_weight": 0.0,
    },

    # Each stage trains on a mixture of pools. Ramped values are given as
    # {start, end, ramp_epochs}; plain numbers are constant.
    "stages": [
        {"name": "subthreshold", "epochs": 100, "lr": 1e-3, "warmup_steps": 500,
         "spiking_fraction": 0.0, "spike_loss_weight": 1.0},
        {"name": "spiking", "epochs": 200, "lr": 3e-4, "warmup_steps": 1000,
         "spiking_fraction": 1.0, "spike_loss_weight": 1.0},
    ],

    "eval": {
        "rollout_every": 5,                # epochs; 0 disables periodic closed-loop checks
        "rollout_traces": 8,
        "rollout_steps": 1000,
        "spike_tolerance": 10,             # rows
        # Spike detection threshold for scoring (mV). Keep it below
        # data.preprocess.vm_clip_max, otherwise clipped peaks are never detected.
        "spike_threshold_mv": -30.0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("files", "compress"):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path_or_dict: str | dict | None = None, overrides: dict | None = None) -> dict:
    if path_or_dict is None:
        user = {}
    elif isinstance(path_or_dict, dict):
        user = path_or_dict
    else:
        with open(path_or_dict) as f:
            text = f.read()
        if path_or_dict.endswith((".yaml", ".yml")):
            if yaml is None:
                raise ImportError("PyYAML is needed for .yaml configs (pip install pyyaml)")
            user = yaml.safe_load(text) or {}
        else:
            user = json.loads(text)
    cfg = _deep_merge(DEFAULTS, user)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    for section in ("data", "model", "training", "eval"):
        bad = set(cfg[section]) - set(DEFAULTS[section])
        if bad:
            raise ValueError(f"Unknown keys in '{section}': {sorted(bad)}")
    split = cfg["data"]["split"]
    if abs(sum(split.values()) - 1.0) > 1e-6:
        raise ValueError(f"data.split must sum to 1, got {split}")
    names = [s["name"] for s in cfg["stages"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Stage names must be unique: {names}")
    for s in cfg["stages"]:
        for key in ("name", "epochs", "lr"):
            if key not in s:
                raise ValueError(f"Stage {s} is missing '{key}'")
    if cfg["model"]["dropout"] not in ("standard", "alpha", "none"):
        raise ValueError("model.dropout must be standard | alpha | none")


def ramp_value(spec, epoch: int) -> float:
    """Value of a possibly ramped stage parameter at a 0-based epoch within the stage."""
    if isinstance(spec, dict):
        start, end = float(spec["start"]), float(spec["end"])
        n = max(int(spec.get("ramp_epochs", 1)), 1)
        # epoch 0 -> start, epoch ramp_epochs-1 (and later) -> end
        frac = min(epoch / (n - 1), 1.0) if n > 1 else 1.0
        if spec.get("shape", "linear") == "cosine":
            import math
            frac = 0.5 - 0.5 * math.cos(math.pi * frac)
        return start + (end - start) * frac
    return float(spec)


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if yaml is not None and path.endswith((".yaml", ".yml")):
            yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            json.dump(cfg, f, indent=2)
