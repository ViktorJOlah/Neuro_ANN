"""Re-run only the SK-block test for a fitted model (e.g. when the NEURON reference finished
after the fit). Updates results/<model>/metrics.json and predictions.npz in place."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spiking_branches as sb

m = sys.argv[1]
out = os.path.join(sb.OUT, m)
r = json.load(open(os.path.join(out, "metrics.json")))
sets = sb.fs.load_split()
z = np.load(os.path.join(sb.sp.CACHE, f"front_end_{m}.npz"))
cell = sb.BranchCell(r["frozen"], r["trained"])
blk = sb.sk_block_test(cell, {pr: z[f"{pr}_test"] for pr in sb.sp.PROTOS}, sets)
arr = dict(np.load(os.path.join(out, "predictions.npz")))
for pr in blk:
    arr.update({f"skblock_{pr}_{k}": v for k, v in blk[pr].pop("_arrays").items()})
np.savez_compressed(os.path.join(out, "predictions.npz"), **arr)
r["sk_block"] = blk
json.dump(r, open(os.path.join(out, "metrics.json"), "w"), indent=2)
print(json.dumps(blk, indent=1))
