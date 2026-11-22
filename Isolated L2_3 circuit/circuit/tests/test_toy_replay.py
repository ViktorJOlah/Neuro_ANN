"""End-to-end check on a small Billeh-like network: cut out L2/3, replay its inputs, and get
the full network's L2/3 spikes back spike for spike."""
import json
import os

import h5py
import numpy as np
import pytest

pytest.importorskip("nest")
pytest.importorskip("bmtk")

from circuit import compare_rates, extract_l23, simulate, sonata_io, toy_network  # noqa: E402


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("toy"))
    cfg = toy_network.build(d, tstop_ms=500.0)
    full = simulate.run_full(cfg, os.path.join(d, "full"))
    summary = extract_l23.extract(os.path.join(d, "network"), os.path.join(d, "l23"), max_edges_per_file=500)
    streams = simulate.make_streams(full, os.path.join(d, "l23"), os.path.join(d, "streams"))
    return d, cfg, summary, streams


def test_extraction_keeps_only_l23_and_all_their_inputs(toy):
    d, _, summary, _ = toy
    assert summary["l23_cells_by_class"] == {"L2/3 E": 120, "L2/3 Htr3a": 8, "L2/3 Pvalb": 16, "L2/3 Sst": 10}
    # every edge onto an L2/3 cell in the original file is in exactly one of the new files
    nodes = sonata_io.read_nodes(os.path.join(d, "network", "v1_nodes.h5"), os.path.join(d, "network", "v1_node_types.csv"))
    l23 = set(nodes.loc[nodes.pop_name.str.match(r"^[ei]23"), "node_id"])
    with h5py.File(os.path.join(d, "network", "v1_v1_edges.h5")) as f:
        t = f["edges/v1_to_v1/target_node_id"][()]
        s = f["edges/v1_to_v1/source_node_id"][()]
    onto = np.isin(t, list(l23))
    rec = onto & np.isin(s, list(l23))
    assert summary["recurrent_l23_edges"] == rec.sum()
    assert summary["v1ext_edges_onto_l23"] == (onto & ~rec).sum()
    with h5py.File(os.path.join(d, "l23", "network", "l23_l23_edges.h5")) as f:
        assert f["edges/l23_to_l23/source_node_id"].attrs["node_population"] == "l23"
        assert f["edges/l23_to_l23/target_node_id"][()].max() < 154


def test_replay_reproduces_full_network_l23_spikes(toy):
    d, cfg, _, _ = toy
    iso = simulate.run_isolated(cfg, os.path.join(d, "l23"), os.path.join(d, "streams", "v1ext_spikes.h5"),
                                os.path.join(d, "iso"))
    res = compare_rates.compare(os.path.join(d, "l23"), os.path.join(d, "streams", "l23_reference_spikes.h5"),
                                iso, os.path.join(d, "check"), 0.0, 500.0, plot=False)
    assert res["matched_spike_fraction"] == 1.0
    assert res["by_class"][0]["ratio"] == pytest.approx(1.0)


def test_poisson_control_keeps_rates(toy):
    d, _, _, _ = toy
    info = simulate.poisson_streams(os.path.join(d, "streams", "v1ext_spikes.h5"),
                                    os.path.join(d, "streams", "v1ext_poisson_spikes.h5"), 0.0, 500.0)
    assert abs(info["spikes_poisson"] - info["spikes_original"]) < 5 * np.sqrt(info["spikes_original"])
