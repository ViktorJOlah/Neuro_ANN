"""Kernel-layer versions of the Aim 2 population files: each model gets its membrane correction from
experiments/kernel_calibration (calibrated from the model alone). Writes <name>_kernel.json next to each population.

    python make_kernel_populations.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "kernel_calibration", "results")


def corr_dir(name, library):
    if library or name == "487661754":
        return name
    return f"fits8_{name}"


for folder, library in (("populations", False), ("populations_library", True)):
    for fn in sorted(glob.glob(os.path.join(HERE, folder, "*.json"))):
        if fn.endswith("_kernel.json") or os.path.basename(fn) == "populations.json":
            continue
        pop = json.load(open(fn))
        missing = []
        for m in pop["models"]:
            c = os.path.join(CAL, corr_dir(m["name"], library), "calibration.json")
            if not os.path.exists(c):
                missing.append(m["name"])
            m["corr"] = os.path.relpath(c, os.path.dirname(fn))
        out = fn[:-5] + "_kernel.json"
        json.dump(pop, open(out, "w"), indent=1)
        print(out, "missing:" if missing else "ok", " ".join(missing))
