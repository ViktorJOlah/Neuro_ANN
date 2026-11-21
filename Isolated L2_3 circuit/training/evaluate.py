"""Closed-loop (autoregressive) evaluation.

The ANN predicts the next row of the state columns from the previous
``time_steps`` rows; in simulation its own predictions are fed back while the
synaptic input columns come from the recording. This module runs exactly that
loop on whole held-out traces, many traces in parallel, and scores it.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .data import TraceDataset, detect_spikes


def make_predict_fn(model):
    @tf.function(reduce_retracing=True)
    def predict(x):
        return tf.reduce_mean(model(x, training=False), axis=1)
    return predict


def rollout(model, ds: TraceDataset, trace_ids, steps: int, predict_fn=None):
    """Simulate ``steps`` rows of each trace in closed loop.

    Returns ``(pred, true)``, each ``(n_traces, steps, n_state)`` in normalized units.
    """
    T = ds.time_steps
    trace_ids = np.asarray(trace_ids)
    max_steps = int(min(ds.lengths[trace_ids].min() - ds.skip - T, steps))
    if max_steps <= 0:
        raise ValueError("Traces are too short for a rollout")
    predict_fn = predict_fn or make_predict_fn(model)
    starts = ds.starts[trace_ids] + ds.skip
    win = np.stack([ds.X[s:s + T] for s in starts])
    state = ds.state_columns
    pred = np.zeros((len(trace_ids), max_steps, len(state)), np.float32)
    true = np.zeros_like(pred)
    for j in range(max_steps):
        p = predict_fn(tf.constant(win)).numpy()
        rows = ds.X[starts + T + j].copy()
        true[:, j] = rows[:, state]
        pred[:, j] = p
        rows[:, state] = p
        win = np.concatenate([win[:, 1:], rows[:, None]], axis=1)
    return pred, true


def _match_spikes(true_sp, pred_sp, tol):
    tp, used = 0, set()
    for t in true_sp:
        cand = [k for k, p in enumerate(pred_sp) if abs(p - t) <= tol and k not in used]
        if cand:
            used.add(min(cand, key=lambda k: abs(pred_sp[k] - t)))
            tp += 1
    return tp, len(pred_sp) - tp, len(true_sp) - tp


def score_rollout(pred, true, ds: TraceDataset, spike_threshold_mv: float, tol: int, exclude: int = 5):
    """Variance explained (subthreshold), spike precision/recall/F1, divergence."""
    vi = ds.state_columns.index(ds.vm_col)
    lo, hi = ds.scaler.mins[vi], ds.scaler.maxs[vi]
    vm_p = pred[..., vi] * (hi - lo) + lo
    vm_t = true[..., vi] * (hi - lo) + lo
    tp = fp = fn = 0
    se, var, n_true_sp, n_pred_sp = [], [], 0, 0
    for p, t in zip(vm_p, vm_t):
        sp_t = detect_spikes(t, spike_threshold_mv)
        sp_p = detect_spikes(p, spike_threshold_mv)
        n_true_sp += len(sp_t)
        n_pred_sp += len(sp_p)
        a, b, c = _match_spikes(list(sp_t), list(sp_p), tol)
        tp, fp, fn = tp + a, fp + b, fn + c
        mask = np.ones(len(t), bool)
        for s in np.concatenate([sp_t, sp_p]).astype(int):
            mask[max(s - exclude, 0):s + 3 * exclude] = False
        if mask.sum() > 10:
            se.append(np.mean((p[mask] - t[mask]) ** 2))
            var.append(np.var(t[mask]))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else (float("nan") if not n_true_sp else 0.0)
    diverged = np.mean(np.any((pred < -0.25) | (pred > 1.25) | ~np.isfinite(pred), axis=(1, 2)))
    return {
        "var_explained_sub": float(1 - np.mean(se) / np.mean(var)) if se else float("nan"),
        "rmse_sub_mv": float(np.sqrt(np.mean(se))) if se else float("nan"),
        "spike_precision": float(precision),
        "spike_recall": float(recall),
        "spike_f1": float(f1),
        "spike_count_ratio": float(n_pred_sp / n_true_sp) if n_true_sp else float("nan"),
        "diverged_fraction": float(diverged),
    }


def evaluate_closed_loop(model, ds: TraceDataset, cfg: dict, split: str = "val", n_traces=None,
                         steps=None, predict_fn=None):
    """Closed-loop scores per pool on a split. Returns {pool: metrics}."""
    e = cfg["eval"]
    n_traces = n_traces or e["rollout_traces"]
    steps = steps or e["rollout_steps"]
    out = {}
    for pool in ds.pools:
        ids = ds.trace_ids(pool, split)[:n_traces]
        if not len(ids):
            continue
        pred, true = rollout(model, ds, ids, steps, predict_fn)
        out[pool] = score_rollout(pred, true, ds, e["spike_threshold_mv"], e["spike_tolerance"])
    return out
