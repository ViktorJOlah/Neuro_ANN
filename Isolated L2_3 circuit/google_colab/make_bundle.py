"""Build hybrid_l23_bundle.zip: everything the Colab notebook needs to run the hybrid L2/3 circuit.

    python google_colab/make_bundle.py --prep H/prep --extract-dir X --streams R/streams \
        --invivo isolated_l23/invivo_comparison/npx_l23_rs_rates_all.npy --out /path/to/hybrid_l23_bundle.zip

--prep is the output of ``python -m circuit.hybrid prepare`` (see circuit/README.md), --extract-dir the
output of ``circuit.extract_l23`` (for each L2/3 cell's class and depth), --streams the folder with
``v1ext_spikes.h5`` and ``l23_reference_spikes.h5``.

Layout of the bundle (unzips to ``hybrid_l23/``):

    code/   circuit/, Standard/, experiments/population/library/, experiments/aim2_controls/ (populations), google_colab/
    data/prep/      the prepared circuit; config_nest_part.template.json has @BUNDLE@ for the bundle folder
    data/allen/     every Allen file the NEST config points at (components, LGN and background spikes)
    data/streams/   v1ext_spikes.h5, l23_reference_spikes.h5
    data/cells/     l23_cells.npz (class and depth per L2/3 id), npx_l23_rs_rates_all.npy

The NEST config is rewritten so that every file it names lies inside the bundle; the notebook replaces
@BUNDLE@ with the folder it unpacked to.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # "Isolated L2_3 circuit"
sys.path.insert(0, ROOT)
from circuit import sonata_io as sio  # noqa: E402
from circuit.extract_l23 import classify  # noqa: E402

PREP_FILES = ("prepare.json", "ids.h5", "pc_to_pc.npz", "in_to_pc.npz", "pc_external_current.npy")
CODE_DIRS = ("circuit", "Standard", os.path.join("experiments", "population", "library"), "google_colab")
AIM2_FILES = ("populations", "populations_library", "models.json", "models_library.json", "analyze.py", "README.md")


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results", "nest_output"))
    else:
        shutil.copy2(src, dst)


def localize_config(cfg, stage, prep, roots):
    """Copy every existing file/dir the config names into the bundle and point the config at it.

    roots: [(absolute source prefix, bundle subfolder)], first match wins. Paths under ``prep`` map to
    data/prep (nest_network is copied whole elsewhere; nest_output need not exist)."""
    prep = os.path.abspath(prep)

    def fix(v):
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        if not isinstance(v, str) or not os.path.isabs(v):
            return v
        p = os.path.normpath(v)
        if p == prep or p.startswith(prep + os.sep):
            return "@BUNDLE@/data/prep" + p[len(prep):]
        if not os.path.exists(p):
            return v
        for src, sub in roots:
            src = os.path.abspath(src)
            if p == src or p.startswith(src + os.sep):
                rel = os.path.join(sub, os.path.relpath(p, src))
                break
        else:
            rel = os.path.join("data", "allen", "other", os.path.basename(p))
        _copy(p, os.path.join(stage, rel))
        return "@BUNDLE@/" + rel

    return fix(cfg)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--prep", required=True); a.add_argument("--extract-dir", required=True)
    a.add_argument("--streams", required=True); a.add_argument("--invivo", required=True)
    a.add_argument("--allen-root", help="common folder of the Allen files the config names (default: guessed)")
    a.add_argument("--out", required=True); a.add_argument("--stage", help="staging folder (default: next to --out)")
    x = a.parse_args(argv)
    stage = x.stage or os.path.join(os.path.dirname(os.path.abspath(x.out)), "hybrid_l23")
    if os.path.exists(stage):
        shutil.rmtree(stage)

    # code
    for d in CODE_DIRS:
        _copy(os.path.join(ROOT, d), os.path.join(stage, "code", d))
    aim2 = os.path.join("experiments", "aim2_controls")
    for f in AIM2_FILES:
        _copy(os.path.join(ROOT, aim2, f), os.path.join(stage, "code", aim2, f))

    # every fitted model a population file points at (the 8-fit set lives outside the library)
    for pdir in ("populations", "populations_library"):
        d = os.path.join(ROOT, aim2, pdir)
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and f != "populations.json":
                for m in json.load(open(os.path.join(d, f))).get("models", []):
                    src = os.path.normpath(os.path.join(d, m["dir"]))
                    _copy(src, os.path.join(stage, "code", os.path.relpath(src, ROOT)))

    # prepared circuit
    dp = os.path.join(stage, "data", "prep")
    for f in PREP_FILES:
        _copy(os.path.join(x.prep, f), os.path.join(dp, f))
    _copy(os.path.join(x.prep, "nest_network"), os.path.join(dp, "nest_network"))
    cfg = json.load(open(os.path.join(x.prep, "config_nest_part.json")))
    for k in ("config_path", "config_dir"):  # BMTK bookkeeping of where prepare ran; not needed to run
        cfg.pop(k, None)
    allen_root = x.allen_root
    if allen_root is None:  # the folder holding the LGN / bkg spike files and components
        ins = [v["input_file"] for v in cfg["inputs"].values() if v["node_set"] in ("lgn", "bkg")]
        allen_root = os.path.commonpath([os.path.dirname(p) for p in ins] + [cfg["components"][k] for k in cfg["components"]
                                                                          if isinstance(cfg["components"][k], str)])
    roots = [(os.path.join(x.streams), os.path.join("data", "streams")), (allen_root, os.path.join("data", "allen"))]
    cfg = localize_config(cfg, stage, x.prep, roots)
    with open(os.path.join(dp, "config_nest_part.template.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # streams, cells, in vivo
    for f in ("v1ext_spikes.h5", "l23_reference_spikes.h5"):
        _copy(os.path.join(x.streams, f), os.path.join(stage, "data", "streams", f))
    net = os.path.join(x.extract_dir, "network")
    nodes = sio.read_nodes(os.path.join(net, "l23_nodes.h5"), os.path.join(net, "l23_node_types.csv"), "l23")
    nodes = nodes.sort_values("node_id").reset_index(drop=True)
    os.makedirs(os.path.join(stage, "data", "cells"), exist_ok=True)
    np.savez_compressed(os.path.join(stage, "data", "cells", "l23_cells.npz"),
                        cls=nodes["pop_name"].map(classify).to_numpy().astype("U12"),
                        depth_um=-nodes["y"].to_numpy(float), xyz=nodes[["x", "y", "z"]].to_numpy(float))
    _copy(x.invivo, os.path.join(stage, "data", "cells", "npx_l23_rs_rates_all.npy"))

    # zip (stored: the big arrays barely compress and this keeps unzipping fast)
    base = os.path.dirname(stage)
    n = 0
    with zipfile.ZipFile(x.out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for d, _, files in os.walk(stage):
            for f in files:
                p = os.path.join(d, f)
                z.write(p, os.path.relpath(p, base)); n += 1
    print(f"{n} files -> {x.out} ({os.path.getsize(x.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
