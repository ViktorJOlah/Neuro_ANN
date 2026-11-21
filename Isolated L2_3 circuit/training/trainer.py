"""Staged / curriculum training loop with checkpoints, resume and a blow-up guard.

Run layout (``<output_dir>/<run_name>/``)::

    config.yaml          the full resolved config
    scaler.json          normalization shared by every stage (needed for simulation)
    data_split.json      which trace went to train / val / test
    state.json           progress, used by --resume
    <stage>/best.weights.h5, last.weights.h5, best_model.keras, best_model.h5,
            history.csv, eval_test.json
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time

import keras
import numpy as np
import tensorflow as tf

from .config import ramp_value, save_config
from .data import Scaler, TraceDataset, WindowSampler
from .evaluate import evaluate_closed_loop, make_predict_fn
from .models import build_model

log = logging.getLogger(__name__)

STAGE_DEFAULTS = {"warmup_steps": 500, "spiking_fraction": 0.0, "spike_loss_weight": 1.0,
                  "reset_optimizer": True, "epochs": 1}


def _final(spec):
    return ramp_value(spec, 10 ** 9)


class Trainer:
    def __init__(self, cfg: dict, resume: bool = False):
        self.cfg = cfg
        self.t = cfg["training"]
        keras.utils.set_random_seed(cfg["seed"])
        self.run_dir = os.path.join(cfg["output_dir"], cfg["run_name"])
        os.makedirs(self.run_dir, exist_ok=True)
        self.state_path = os.path.join(self.run_dir, "state.json")
        self.state = {"stages": {}}

        if resume and os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
            scaler = Scaler.from_json(os.path.join(self.run_dir, "scaler.json"))
            with open(os.path.join(self.run_dir, "data_split.json")) as f:
                split = json.load(f)["splits"]
            self.ds = TraceDataset(cfg, scaler=scaler, split_assignment=split)
            log.info("Resuming run in %s", self.run_dir)
        else:
            if os.path.exists(self.state_path):
                raise FileExistsError(f"{self.run_dir} already has a run; use --resume or a new run_name")
            self.ds = TraceDataset(cfg)
            self.ds.save_metadata(self.run_dir)
            save_config(cfg, os.path.join(self.run_dir, "config.yaml"))

        ds = self.ds
        self.T = ds.time_steps
        self.k = int(self.t["rollout_steps"])
        self.horizon = self.k + 1
        self.model = build_model(cfg, ds.n_features, ds.n_state)
        self.X_tf = tf.constant(ds.X)
        self.near_tf = tf.constant(ds.near_spike)
        mask = np.zeros(ds.n_features, np.float32)
        mask[ds.state_columns] = 1.0
        self.state_mask = tf.constant(mask)
        sel = np.zeros((ds.n_state, ds.n_features), np.float32)
        for i, c in enumerate(ds.state_columns):
            sel[i, c] = 1.0
        self.state_sel = tf.constant(sel)
        sw = self.t["state_loss_weights"] or [1.0] * ds.n_state
        if len(sw) != ds.n_state:
            raise ValueError("training.state_loss_weights needs one value per state column")
        self.state_w = tf.constant((np.asarray(sw) / np.sum(sw)).astype(np.float32))
        self.sampler = WindowSampler(ds, "train", self.horizon, cfg["seed"] + 1)
        vrng = np.random.default_rng(cfg["seed"] + 2)
        self.val_windows = {}
        for pool in ds.pools:
            w = ds.window_starts(pool, "val", 1)
            if len(w) > cfg["data"]["max_val_windows"]:
                w = np.sort(vrng.choice(w, cfg["data"]["max_val_windows"], replace=False))
            if len(w):
                self.val_windows[pool] = w
        if not self.val_windows:
            raise ValueError("No validation windows; add traces or raise data.split.val")
        self._eval_fn = self._make_eval_fn()
        self.predict_fn = make_predict_fn(self.model)

    # ----------------------------------------------------------------- pieces
    def _gather(self, idx, length):
        rows = idx[:, None] + tf.range(length, dtype=idx.dtype)[None, :]
        return tf.gather(self.X_tf, rows)

    def _per_sample_loss(self, rep, y):
        d = rep - y[:, None, :]
        e = tf.abs(d) if self.t["loss"] == "mae" else tf.square(d)
        return tf.reduce_mean(tf.reduce_sum(e * self.state_w, axis=-1), axis=1)

    def _make_optimizer(self, lr):
        kind = self.t["optimizer"].lower()
        cls = {"adam": keras.optimizers.Adam, "nadam": keras.optimizers.Nadam,
               "adamax": keras.optimizers.Adamax}[kind]
        opt = cls(learning_rate=lr, clipnorm=self.t["clipnorm"])
        opt.build(self.model.trainable_variables)
        return opt

    def _make_train_step(self, opt):
        model, T, k = self.model, self.T, self.k
        noise_std = float(self.t["state_noise_std"])
        roll_w = float(self.t["rollout_loss_weight"])

        @tf.function(reduce_retracing=True)
        def step(idx, spike_w):
            win = self._gather(idx, T + self.horizon)
            x = win[:, :T]
            y = tf.gather(win[:, T], self.ds.state_columns, axis=1)
            w = 1.0 + (spike_w - 1.0) * tf.gather(self.near_tf, idx + T)
            x_in = x
            if noise_std > 0:
                x_in = x + tf.random.normal(tf.shape(x), stddev=noise_std) * self.state_mask
            with tf.GradientTape() as tape:
                rep = model(x_in, training=True)
                one = tf.reduce_sum(w * self._per_sample_loss(rep, y)) / tf.reduce_sum(w)
                total = one
                if k > 0 and roll_w > 0:
                    x_cur, rl = x, 0.0
                    p = tf.reduce_mean(rep, axis=1)
                    for j in range(1, k + 1):
                        row = win[:, T + j - 1]
                        row = row * (1.0 - self.state_mask) + tf.matmul(p, self.state_sel)
                        x_cur = tf.concat([x_cur[:, 1:], row[:, None]], axis=1)
                        rep_j = model(x_cur, training=True)
                        y_j = tf.gather(win[:, T + j], self.ds.state_columns, axis=1)
                        w_j = 1.0 + (spike_w - 1.0) * tf.gather(self.near_tf, idx + T + j)
                        rl += tf.reduce_sum(w_j * self._per_sample_loss(rep_j, y_j)) / tf.reduce_sum(w_j)
                        p = tf.reduce_mean(rep_j, axis=1)
                    total = total + roll_w * rl / k
                if model.losses:
                    total = total + tf.add_n(model.losses)
            grads = tape.gradient(total, model.trainable_variables)
            finite = tf.math.is_finite(total)
            for g in grads:
                finite = tf.logical_and(finite, tf.reduce_all(tf.math.is_finite(g)))
            grads = [tf.where(finite, g, tf.zeros_like(g)) for g in grads]
            opt.apply_gradients(zip(grads, model.trainable_variables))
            return one, tf.logical_not(finite)

        return step

    def _make_eval_fn(self):
        model, T = self.model, self.T

        @tf.function(reduce_retracing=True)
        def ev(idx):
            win = self._gather(idx, T + 1)
            y = tf.gather(win[:, T], self.ds.state_columns, axis=1)
            rep = model(win[:, :T], training=False)
            return self._per_sample_loss(rep, y), tf.gather(self.near_tf, idx + T)

        return ev

    def validate(self, spike_w: float) -> dict:
        out = {}
        for pool, wins in self.val_windows.items():
            ls, ns = [], []
            for i in range(0, len(wins), 1024):
                l, n = self._eval_fn(tf.constant(wins[i:i + 1024]))
                ls.append(l.numpy())
                ns.append(n.numpy())
            l, n = np.concatenate(ls), np.concatenate(ns)
            w = 1.0 + (spike_w - 1.0) * n
            out[f"val_{pool}"] = float(l.mean())
            out[f"val_{pool}_weighted"] = float((w * l).sum() / w.sum())
        return out

    def _monitor(self, val: dict, stage: dict) -> float:
        p = _final(stage["spiking_fraction"])
        sub, spk = val.get("val_subthreshold_weighted"), val.get("val_spiking_weighted")
        if sub is None and spk is None:
            return float(np.mean([v for k, v in val.items() if k.endswith("_weighted")]))
        if sub is None:
            return spk
        if spk is None:
            return sub
        return (1 - p) * sub + p * spk

    def _save_state(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.state_path)

    # -------------------------------------------------------------------- run
    def run(self, only: list[str] | None = None):
        stages = [{**STAGE_DEFAULTS, **s} for s in self.cfg["stages"]]
        names = [s["name"] for s in stages]
        if only:
            missing = set(only) - set(names)
            if missing:
                raise ValueError(f"Unknown stages {sorted(missing)}; config has {names}")
        results = {}
        for i, stage in enumerate(stages):
            name = stage["name"]
            st = self.state["stages"].get(name, {})
            if only and name not in only:
                continue
            if st.get("status") == "done":
                log.info("Stage '%s' already done, skipping", name)
                continue
            if not st:  # fresh stage: start from the best weights of the latest earlier stage
                for prev in reversed(names[:i]):
                    p = os.path.join(self.run_dir, prev, "best.weights.h5")
                    if os.path.exists(p):
                        self.model.load_weights(p)
                        log.info("Stage '%s' starts from best weights of '%s'", name, prev)
                        break
                else:
                    if i > 0:
                        log.warning("Stage '%s' has no earlier checkpoint; starting from %s", name,
                                    "init_weights" if self.cfg["model"]["init_weights"] else "random init")
            results[name] = self._run_stage(stage)
        return results

    def _run_stage(self, stage: dict) -> dict:
        name, t = stage["name"], self.t
        sdir = os.path.join(self.run_dir, name)
        os.makedirs(sdir, exist_ok=True)
        best_path, last_path = os.path.join(sdir, "best.weights.h5"), os.path.join(sdir, "last.weights.h5")
        hist_path = os.path.join(sdir, "history.csv")
        st = self.state["stages"].setdefault(name, {"status": "running", "epoch": 0, "best": None,
                                                    "lr_scale": 1.0, "bad_epochs": 0, "retries": 0})
        if st["epoch"] > 0 and os.path.exists(last_path):
            self.model.load_weights(last_path)
            log.info("Stage '%s': resuming at epoch %d (optimizer restarts with warmup)", name, st["epoch"])
        if not os.path.exists(best_path):
            self.model.save_weights(best_path)  # rollback target before the first epoch finishes
        best = st["best"] if st["best"] is not None else math.inf

        opt = self._make_optimizer(stage["lr"])
        train_step = self._make_train_step(opt)
        warm, since_reset = max(int(stage["warmup_steps"]), 1), 0
        B, steps = int(t["batch_size"]), int(t["steps_per_epoch"])
        if stage.get("steps_per_epoch"):
            steps = int(stage["steps_per_epoch"])

        epoch = st["epoch"]
        while epoch < stage["epochs"]:
            t0 = time.time()
            p = ramp_value(stage["spiking_fraction"], epoch)
            sw = ramp_value(stage["spike_loss_weight"], epoch)
            idx = self.sampler.sample(steps * B, p).reshape(steps, B)
            base_lr = max(stage["lr"] * st["lr_scale"], t["min_lr"])
            losses, skipped = [], 0
            sw_t = tf.constant(sw, tf.float32)
            for s in range(steps):
                lr = base_lr * min(1.0, (since_reset + 1) / warm)
                opt.learning_rate.assign(lr)
                l, sk = train_step(tf.constant(idx[s]), sw_t)
                losses.append(l)
                skipped += int(sk)
                since_reset += 1
            train_loss = float(np.mean([x.numpy() for x in losses]))
            val = self.validate(_final(stage["spike_loss_weight"]))
            monitor = self._monitor(val, stage)

            blew_up = (not math.isfinite(train_loss) or not math.isfinite(monitor)
                       or skipped > steps // 2
                       or (math.isfinite(best) and monitor > t["blowup_factor"] * best))
            if blew_up:
                st["retries"] += 1
                log.warning("Stage '%s' epoch %d unstable (train=%.3g, monitor=%.3g, skipped=%d); "
                            "restoring best weights, LR x%.2f (retry %d/%d)", name, epoch, train_loss,
                            monitor, skipped, t["lr_backoff"], st["retries"], t["max_retries"])
                if st["retries"] > t["max_retries"]:
                    self._save_state()
                    raise RuntimeError(f"Stage '{name}' kept diverging; see {hist_path}. "
                                       "Lower the stage lr or raise warmup_steps.")
                self.model.load_weights(best_path)
                st["lr_scale"] *= t["lr_backoff"]
                opt = self._make_optimizer(stage["lr"])
                train_step = self._make_train_step(opt)
                since_reset = 0
                self._save_state()
                continue

            improved = monitor < best - t["min_delta"]
            if improved:
                best, st["bad_epochs"] = monitor, 0
                self.model.save_weights(best_path)
            else:
                st["bad_epochs"] += 1
                if st["bad_epochs"] % t["plateau_patience"] == 0:
                    st["lr_scale"] *= t["plateau_factor"]
                    log.info("Plateau: LR scale -> %.3g", st["lr_scale"])
            self.model.save_weights(last_path)

            row = {"epoch": epoch, "spiking_fraction": p, "spike_loss_weight": sw, "lr": base_lr,
                   "train_loss": train_loss, **val, "monitor": monitor, "best": best,
                   "skipped_steps": skipped, "seconds": round(time.time() - t0, 1)}
            every = self.cfg["eval"]["rollout_every"]
            if every and (epoch + 1) % every == 0:
                for pool, m in evaluate_closed_loop(self.model, self.ds, self.cfg, "val",
                                                    predict_fn=self.predict_fn).items():
                    row.update({f"cl_{pool}_{k}": v for k, v in m.items()})
            self._append_history(hist_path, row)
            log.info("[%s] epoch %d  p_spk=%.2f  w_spk=%.1f  lr=%.2e  train=%.5f  monitor=%.5f%s",
                     name, epoch, p, sw, base_lr, train_loss, monitor, "  *" if improved else "")

            epoch += 1
            st.update(epoch=epoch, best=best)
            self._save_state()
            if st["bad_epochs"] >= t["early_stopping_patience"]:
                log.info("Early stopping stage '%s' at epoch %d", name, epoch)
                break

        # finalize: best weights, export, test-set closed-loop score
        self.model.load_weights(best_path)
        self.model.save(os.path.join(sdir, "best_model.keras"))
        try:
            self.model.save(os.path.join(sdir, "best_model.h5"))  # legacy format for the old scripts
        except Exception as e:  # pragma: no cover
            log.warning("Could not write legacy .h5: %s", e)
        test = evaluate_closed_loop(self.model, self.ds, self.cfg, "test", predict_fn=self.predict_fn)
        with open(os.path.join(sdir, "eval_test.json"), "w") as f:
            json.dump({"best_monitor": best, "closed_loop": test}, f, indent=2)
        st["status"] = "done"
        self._save_state()
        log.info("Stage '%s' done: best monitor %.5f, test closed loop %s", name, best, test)
        return {"best_monitor": best, "closed_loop": test}

    @staticmethod
    def _append_history(path, row):
        rows = []
        if os.path.exists(path):
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        rows.append(row)
        fields = []
        for r in rows:  # columns can appear later (closed-loop metrics every N epochs)
            fields += [k for k in r if k not in fields]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
