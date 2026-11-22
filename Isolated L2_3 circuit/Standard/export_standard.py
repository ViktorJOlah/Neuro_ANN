"""Writes Standard/ from the repo's fitted files (run from "Isolated L2_3 circuit/"):
front-end weights from experiments/current_window/results/cnnlstm, Izhikevich and branch parameters
from experiments/conductance_branches/results_chosen_isi/cnnlstm. Then checks standard_model.py
reproduces the saved test predictions."""
import json, os, shutil, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD = os.path.join(ROOT, "Standard")
sys.path.insert(0, os.path.join(ROOT, "experiments", "conductance_branches"))
import spiking_branches as sb  # noqa: E402
sp = sb.sp
FIT = os.path.join(ROOT, "experiments", "conductance_branches", "results_chosen_isi", "cnnlstm")
FE = os.path.join(ROOT, "experiments", "current_window", "results", "cnnlstm")

m, _ = sp.load_frozen("cnnlstm")
ws = m.net.get_weights()
os.makedirs(os.path.join(STD, "weights"), exist_ok=True)
np.savez(os.path.join(STD, "weights", "front_end_cnnlstm.npz"), **{f"w{i}": w for i, w in enumerate(ws)})
shutil.copy(os.path.join(FE, "model.weights.h5"), os.path.join(STD, "weights", "front_end_cnnlstm_full.weights.h5"))
r = json.load(open(os.path.join(FIT, "metrics.json")))
izh = {k: float(v) for k, v in r["frozen"].items()}
izh["g"], izh["I0"] = float(m.g.numpy()), float(m.I0.numpy())
json.dump({"name": "Standard (The Chosen One)", "cell": "Allen 488462783 / specimen 487661754, L2/3 PC",
           "izhikevich": izh, "branches": {k: float(v) for k, v in r["trained"].items()},
           "source": {"front_end": os.path.relpath(FE, ROOT), "spike_fit": os.path.relpath(FIT, ROOT)},
           "test_metrics": r["eval"], "spike_free_cont_rmse_mV": r["spike_free_cont_rmse_mV"]},
          open(os.path.join(STD, "params.json"), "w"), indent=2)

sys.path.insert(0, STD)
import standard_model as sm  # noqa: E402
cell = sm.load()
sets = sp.fs.load_split()
z = np.load(os.path.join(FIT, "predictions.npz"))
for pr in ("square", "noise", "ramp"):
    s = sets[(pr, "test")]
    vm, spk = cell.simulate(s["I"][:8], V_seed=s["V"][:8], seed_ms=450)
    ref_v, ref_s = z[f"{pr}_cont_vm"][:8], z[f"{pr}_cont_spikes"][:8]
    print(pr, "max |dVm| %.4f mV" % np.abs(vm - ref_v).max(), "spike mismatches", int((spk != ref_s).sum()))
