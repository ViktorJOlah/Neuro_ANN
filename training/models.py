"""The published CNN-LSTM architectures, rebuilt for Keras 3.

Layer order and sizes match the original scripts exactly, so weights saved by
the old code can be loaded with ``model.init_weights``. The only optional
change is ``dropout``: standard dropout (original), AlphaDropout (the variant
that keeps SELU layers self-normalizing) or none.

Both models output ``(batch, repeats, n_state)`` because of the RepeatVector;
like the original prediction code, the prediction is the mean over repeats.
"""
from __future__ import annotations

import logging

import keras
from keras import layers, regularizers

log = logging.getLogger(__name__)


def _dropout(kind: str, rate: float):
    if kind == "standard":
        return [layers.Dropout(rate)]
    if kind == "alpha":
        return [layers.AlphaDropout(rate)]
    return []


def cnn_lstm_multi(time_steps: int, n_features: int, n_state: int, dropout: str = "standard"):
    """L2/3 (multicompartmental) model from fit_multicompartmental.py / final_L2_3.ipynb.

    The original has no dropout, so ``dropout`` only matters if set to 'alpha'.
    """
    extra = _dropout(dropout, 0.15) if dropout == "alpha" else []
    stack = [
        keras.Input(shape=(time_steps, n_features)),
        layers.Conv1D(512, 1, activation="relu", kernel_initializer="he_uniform", padding="causal",
                      kernel_regularizer=regularizers.l1(1e-5)),
        layers.Conv1D(256, 5, activation="relu", kernel_initializer="he_uniform", padding="causal"),
        layers.Conv1D(128, 1, activation="tanh", kernel_initializer="glorot_uniform", padding="causal"),
        layers.Flatten(),
        layers.RepeatVector(5),
        layers.LSTM(128, activation="tanh", kernel_initializer="glorot_uniform", return_sequences=True),
        layers.LSTM(128, activation="tanh", kernel_initializer="glorot_uniform", return_sequences=True),
        layers.LSTM(128, activation="tanh", kernel_initializer="glorot_uniform", return_sequences=True),
        layers.Dense(128, activation="relu", kernel_initializer="he_uniform"),
    ]
    for _ in range(4):
        stack += [layers.Dense(100, activation="selu", kernel_initializer="lecun_uniform")] + extra
    stack.append(layers.Dense(n_state))
    return keras.Sequential(stack, name="cnn_lstm_multi")


def cnn_lstm_single(time_steps: int, n_features: int, n_state: int, dropout: str = "standard"):
    """Single-compartment ('ball') model from fit_CNN_LSTM_latest.py."""
    stack = [
        keras.Input(shape=(time_steps, n_features)),
        layers.Conv1D(128, 1, activation="relu", kernel_initializer="he_uniform", padding="causal"),
        layers.Conv1D(100, 5, activation="relu", kernel_initializer="he_uniform", padding="causal"),
        layers.Conv1D(50, 1, activation="tanh", kernel_initializer="glorot_uniform", padding="causal"),
        layers.Flatten(),
        layers.RepeatVector(n_features),
        layers.LSTM(128, activation="tanh", kernel_initializer="glorot_uniform", return_sequences=True),
        layers.TimeDistributed(layers.Dense(128, activation="relu", kernel_initializer="he_uniform")),
    ]
    for _ in range(4):
        stack += [layers.Dense(100, activation="selu", kernel_initializer="lecun_uniform")]
        stack += _dropout(dropout, 0.15)
    stack.append(layers.Dense(n_state))
    return keras.Sequential(stack, name="cnn_lstm_single")


ARCHITECTURES = {"cnn_lstm_multi": cnn_lstm_multi, "cnn_lstm_single": cnn_lstm_single}


def build_model(cfg: dict, n_features: int, n_state: int):
    m = cfg["model"]
    if m["architecture"] not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {m['architecture']}; choose from {list(ARCHITECTURES)}")
    model = ARCHITECTURES[m["architecture"]](int(m["time_steps"]), n_features, n_state, m["dropout"])
    if m.get("init_weights"):
        load_legacy_weights(model, m["init_weights"])
    return model


def load_legacy_weights(model, path: str) -> None:
    """Load weights from an old .h5 (weights-only or full model) or a new .weights.h5."""
    try:
        model.load_weights(path)
        log.info("Loaded weights from %s", path)
        return
    except Exception as e:  # full-model h5 saved by the old code, or layer-count mismatch
        first_error = e
    try:
        old = keras.models.load_model(path, compile=False)
        model.set_weights(old.get_weights())
        log.info("Loaded weights from full model %s", path)
    except Exception as e:
        raise ValueError(f"Could not load weights from {path}: {first_error} / {e}") from e
