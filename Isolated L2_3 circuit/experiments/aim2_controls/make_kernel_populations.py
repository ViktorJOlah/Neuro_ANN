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


# Rheobase matching through the kernel path. In kernel mode synaptic current reaches the Izhikevich layer through each
# model's correction, whose DC gain (k0 + c1 + c2, 0.3-1.0) differs between models, so matching DC rheobase alone leaves
# a gain difference (the model's rate in the rheo_matched_kernel run follows its DC gain, Spearman ~0.77). The
# rheo_matched_dc control also divides each PC's scale by its model's DC gain, relative to the Standard's.
def dc_gain(path):
    c = json.load(open(path))["correction"]
    return c["k0"] + c["c1"] + c["c2"]


import numpy as np  # noqa: E402
for folder, library in (("populations", False), ("populations_library", True)):
    fn = os.path.join(HERE, folder, "rheo_matched_kernel.json")
    pop = json.load(open(fn))
    d = os.path.dirname(fn)
    g = np.array([dc_gain(os.path.join(d, m["corr"])) for m in pop["models"]])
    g_ref = dc_gain(os.path.join(CAL, "487661754", "calibration.json"))
    assign = np.load(os.path.join(d, pop["assignment"]))
    scale = np.load(os.path.join(d, pop["input_scale"])) * (g_ref / g[assign])
    np.save(os.path.join(d, "input_scale_rheo_dc.npy"), scale.astype(np.float32))
    pop["input_scale"] = "input_scale_rheo_dc.npy"
    json.dump(pop, open(os.path.join(d, "rheo_matched_dc_kernel.json"), "w"), indent=1)
    print(folder, "rheo_matched_dc_kernel.json: DC gains", np.round(np.percentile(g, [0, 50, 100]), 2), "ref", round(g_ref, 3))
