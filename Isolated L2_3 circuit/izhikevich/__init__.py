"""Izhikevich (2003) top layer: batched, differentiable, checked against the published model."""
from .reference import PRESETS, V_PEAK, simulate  # noqa: F401
from .layer import IzhikevichCell, IzhikevichLayer, exact_graph_mode  # noqa: F401
