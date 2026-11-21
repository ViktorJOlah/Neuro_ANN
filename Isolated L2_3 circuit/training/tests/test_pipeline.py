"""CPU smoke tests: python -m pytest training/tests -q"""
import json
import os

import numpy as np
import pytest

from training.config import load_config, ramp_value
from training.data import TraceDataset, WindowSampler, preprocess_raw, read_neuron_txt, split_into_traces
from training.synthetic import simulate, write_neuron_txt


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    sub = simulate(8, 601, 6, rate_scale=0.1, seed=1)
    spk = simulate(8, 601, 6, rate_scale=1.5, seed=2)
    write_neuron_txt(d / "sub.txt", sub)
    write_neuron_txt(d / "spk.txt", spk)
    write_neuron_txt(d / "ball.txt", simulate(6, 401, 2, rate_scale=1.5, seed=3, currents=True), headers=False)
    return {"sub": str(d / "sub.txt"), "spk": str(d / "spk.txt"), "ball": str(d / "ball.txt"), "dir": d}


def tiny_cfg(files, tmp_path, **over):
    cfg = {
        "run_name": "t", "output_dir": str(tmp_path),
        "data": {"files": {"subthreshold": [files["sub"]], "spiking": [files["spk"]]},
                 "preprocess": {"vm_clip_max": -20.0}, "spike_threshold_mv": -30.0,
                 "split": {"train": 0.5, "val": 0.25, "test": 0.25}},
        "model": {"time_steps": 16},
        "training": {"steps_per_epoch": 4, "batch_size": 8},
        "stages": [{"name": "subthreshold", "epochs": 2, "lr": 1e-3, "warmup_steps": 2},
                   {"name": "spiking", "epochs": 2, "lr": 3e-4, "warmup_steps": 2, "spiking_fraction": 1.0}],
        "eval": {"rollout_every": 1, "rollout_traces": 2, "rollout_steps": 50},
    }
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) and k != "stages" else v
    return load_config(cfg)


# ------------------------------------------------------------------- data

def test_reader_strips_headers_and_trailing_space(files):
    data, headers = read_neuron_txt(files["sub"])
    assert data.shape == (8 * 601, 7)
    assert headers == [(i * 601, 601) for i in range(8)]
    assert np.isfinite(data).all()
    traces, tl = split_into_traces(data, headers, None)
    assert tl == 601 and len(traces) == 8


def test_reader_without_headers(files):
    data, headers = read_neuron_txt(files["ball"])
    assert headers == [] and data.shape == (6 * 401, 5)


def test_preprocess_is_vectorized_and_effective():
    x = np.array([[0.0, 0.2, -0.5], [10.0, 0.01, -200.0]], np.float32)
    y = preprocess_raw(x, {"vm_clip_max": -20.0, "negate_columns": [2],
                           "compress": {"1": [0.05, 75], 2: [0.1, 190]}}, 0)
    assert y[0, 0] == -20 and y[1, 0] == -20
    assert np.isclose(y[0, 1], (0.2 - 0.05) / 75 + 0.05)
    assert np.isclose(y[1, 2], (200 - 0.1) / 190 + 0.1)
    assert x[1, 2] == -200.0  # input untouched


def test_split_by_trace_and_windows_stay_inside_traces(files, tmp_path):
    cfg = tiny_cfg(files, tmp_path)
    ds = TraceDataset(cfg)
    for pool in ds.pools:
        ids = {s: set(ds.trace_ids(pool, s)) for s in ("train", "val", "test")}
        assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"])
        assert all(len(v) for v in ids.values())
    T = ds.time_steps
    trace_of_row = np.repeat(np.arange(len(ds.lengths)), ds.lengths)
    for horizon in (1, 4):
        w = ds.window_starts(None, "train", horizon)
        assert (trace_of_row[w] == trace_of_row[w + T + horizon - 1]).all()
        assert (ds.trace_split[trace_of_row[w]] == "train").all()


def test_scaler_fit_on_train_only(files, tmp_path):
    ds = TraceDataset(tiny_cfg(files, tmp_path))
    rows = np.concatenate([np.arange(ds.starts[i], ds.starts[i] + ds.lengths[i])
                           for i in ds.trace_ids(None, "train")])
    vm = ds.X[rows, 0]
    assert np.isclose(vm.min(), 0, atol=1e-6) and np.isclose(vm.max(), 1, atol=1e-6)
    assert ds.X.dtype == np.float32


def test_sampler_mixture(files, tmp_path):
    ds = TraceDataset(tiny_cfg(files, tmp_path))
    s = WindowSampler(ds, "train", 1, 0)
    spk = set(s.by_pool["spiking"].tolist())
    idx = s.sample(4000, 0.3)
    frac = np.mean([i in spk for i in idx])
    assert 0.25 < frac < 0.35
    assert all(i not in spk for i in s.sample(500, 0.0))


def test_auto_pool_split(files, tmp_path):
    cfg = tiny_cfg(files, tmp_path, data={"files": {"all": [files["sub"], files["spk"]]}})
    ds = TraceDataset(cfg)
    for i, p in enumerate(ds.trace_pool):
        assert (p == "spiking") == (len(ds.spikes[i]) >= 1)


def test_ramp():
    spec = {"start": 0.0, "end": 1.0, "ramp_epochs": 5}
    assert ramp_value(spec, 0) == 0 and ramp_value(spec, 4) == 1 and ramp_value(spec, 99) == 1
    assert np.isclose(ramp_value(spec, 2), 0.5)
    assert ramp_value(0.3, 7) == 0.3


def test_config_rejects_typos():
    with pytest.raises(ValueError):
        load_config({"training": {"batchsize": 3}})


# --------------------------------------------------------------- training

def test_two_stage_run_checkpoints_and_resume(files, tmp_path):
    from training.trainer import Trainer
    cfg = tiny_cfg(files, tmp_path)
    res = Trainer(cfg).run()
    run = os.path.join(str(tmp_path), "t")
    for stage in ("subthreshold", "spiking"):
        for f in ("best.weights.h5", "last.weights.h5", "best_model.keras", "best_model.h5",
                  "history.csv", "eval_test.json"):
            assert os.path.exists(os.path.join(run, stage, f)), f
        assert np.isfinite(res[stage]["best_monitor"])
    state = json.load(open(os.path.join(run, "state.json")))
    assert all(s["status"] == "done" for s in state["stages"].values())
    # a second fresh run in the same dir must refuse instead of mixing runs
    with pytest.raises(FileExistsError):
        Trainer(cfg)
    # resume with nothing left to do is a no-op
    assert Trainer(cfg, resume=True).run() == {}
    # mid-stage resume: pretend 'spiking' stopped after epoch 1 of 3
    state["stages"]["spiking"].update(status="running", epoch=1)
    json.dump(state, open(os.path.join(run, "state.json"), "w"))
    cfg["stages"][1]["epochs"] = 3
    tr = Trainer(cfg, resume=True)
    assert set(tr.run()) == {"spiking"}
    hist = open(os.path.join(run, "spiking", "history.csv")).read().strip().splitlines()
    assert len(hist) == 1 + 2 + 2  # header + first 2 epochs + epochs 1,2 after resume


def test_only_selected_stage_starts_from_previous_best(files, tmp_path, caplog):
    from training.trainer import Trainer
    cfg = tiny_cfg(files, tmp_path)
    Trainer(cfg).run(["subthreshold"])
    with caplog.at_level("INFO"):
        Trainer(cfg, resume=True).run(["spiking"])
    assert "starts from best weights of 'subthreshold'" in caplog.text


def test_curriculum_and_opt_in_rollout_loss(files, tmp_path):
    from training.trainer import Trainer
    cfg = tiny_cfg(files, tmp_path,
                   training={"rollout_steps": 3, "rollout_loss_weight": 0.5, "state_noise_std": 0.01},
                   stages=[{"name": "curriculum", "epochs": 3, "lr": 1e-3, "warmup_steps": 2,
                            "spiking_fraction": {"start": 0.0, "end": 0.8, "ramp_epochs": 3},
                            "spike_loss_weight": {"start": 1.0, "end": 4.0, "ramp_epochs": 3}}])
    Trainer(cfg).run()
    import csv
    rows = list(csv.DictReader(open(os.path.join(str(tmp_path), "t", "curriculum", "history.csv"))))
    assert [float(r["spiking_fraction"]) for r in rows] == pytest.approx([0.0, 0.4, 0.8])
    assert [float(r["spike_loss_weight"]) for r in rows] == pytest.approx([1.0, 2.5, 4.0])


def test_blowup_guard_rolls_back_then_gives_up(files, tmp_path):
    from training.trainer import Trainer
    cfg = tiny_cfg(files, tmp_path, training={"blowup_factor": 1e-9, "max_retries": 2},
                   stages=[{"name": "s", "epochs": 5, "lr": 1e-3, "warmup_steps": 1}])
    with pytest.raises(RuntimeError, match="kept diverging"):
        Trainer(cfg).run()
    state = json.load(open(os.path.join(str(tmp_path), "t", "state.json")))
    assert state["stages"]["s"]["retries"] == 3
    assert state["stages"]["s"]["lr_scale"] == pytest.approx(0.25)


def test_single_compartment_three_state_columns_and_legacy_init(files, tmp_path):
    from training.trainer import Trainer
    cfg = tiny_cfg(files, tmp_path,
                   data={"files": {"all": [files["ball"]]}, "trace_length": 401,
                         "state_columns": [0, 1, 2], "preprocess": {"negate_columns": [2]},
                         "spike_threshold_mv": -20.0},
                   model={"architecture": "cnn_lstm_single", "time_steps": 16},
                   training={"state_loss_weights": [1, 0.7, 0.7], "loss": "mse"},
                   stages=[{"name": "only", "epochs": 1, "lr": 1e-3, "warmup_steps": 1, "spiking_fraction": 1.0}],
                   eval={"rollout_every": 0, "spike_threshold_mv": -20.0})
    Trainer(cfg).run()
    h5 = os.path.join(str(tmp_path), "t", "only", "best_model.h5")
    cfg2 = load_config({**cfg, "run_name": "t2", "model": {**cfg["model"], "init_weights": h5}})
    tr = Trainer(cfg2)
    import keras
    ref = keras.models.load_model(h5, compile=False)
    for a, b in zip(tr.model.get_weights(), ref.get_weights()):
        np.testing.assert_array_equal(a, b)
