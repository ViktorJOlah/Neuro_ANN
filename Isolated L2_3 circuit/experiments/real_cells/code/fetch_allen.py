"""Download everything the real-cell comparison needs from the Allen Cell Types Database (api.brain-map.org).

Run from "Isolated L2_3 circuit/":
    python experiments/real_cells/code/fetch_allen.py --nwb-dir /mnt/project-files/real_cells_nwb [--cells ...]

Per cell, into experiments/real_cells/data/<cell>/:
  meta.json              specimen record + every published model (name, id, Allen's own EV / RMS of each run)
  sweeps.json            the EphysSweep table (stimulus name, amplitude, leak current per sweep)
  glif/model_<level>_<id>_neuron_config.json   every published GLIF level
  allactive/reconstruction.swc                 the cell's 3D reconstruction
and the raw recording into <nwb-dir>/<cell>_ephys.nwb (50 kHz, 19-91 MB, kept out of git).

The all-active parameters (allactive/fit_parameters.json) are the best hall-of-fame model of Nandi et al. 2022,
github.com/AllenInstitute/All-active-Manuscript assets/mouse_models/<cell>/hof_models/hof_param_<cell>_0.json;
--nandi tries to fetch it from raw.githubusercontent.com (see allactive/source.json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
API = "https://api.brain-map.org"
CELLS = ("513531471", "485468180", "502269786", "318808427", "521409057", "382982932")
NANDI = ("https://raw.githubusercontent.com/AllenInstitute/All-active-Manuscript/master/"
         "assets/mouse_models/{c}/hof_models/hof_param_{c}_0.json")


def query(criteria, num_rows=2000):
    u = f"{API}/api/v2/data/query.json?num_rows={num_rows}&criteria=" + urllib.parse.quote(criteria, safe=":,[]$=()'")
    return json.load(urllib.request.urlopen(u, timeout=120))["msg"]


def download(url, out):
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    urllib.request.urlretrieve(url, out + ".part")
    os.replace(out + ".part", out)
    print("  ", os.path.basename(out))


def fetch(cell, nwb_dir, nandi):
    o = os.path.join(DATA, cell)
    sp = query(f"model::Specimen,rma::criteria,[id$eq{cell}],rma::include,structure,"
               "ephys_result(well_known_files(well_known_file_type)),"
               "neuron_reconstructions(well_known_files(well_known_file_type))")[0]
    for wf in sp["ephys_result"]["well_known_files"]:
        if wf["well_known_file_type"]["name"] == "NWBDownload":
            download(API + wf["download_link"], os.path.join(nwb_dir, f"{cell}_ephys.nwb"))
    for nr in sp.get("neuron_reconstructions", []):
        for wf in nr["well_known_files"]:
            if wf["well_known_file_type"]["name"] == "3DNeuronReconstruction":
                download(API + wf["download_link"], os.path.join(o, "allactive", "reconstruction.swc"))
    sweeps = query(f"model::EphysSweep,rma::criteria,[specimen_id$eq{cell}]")
    json.dump(sorted(sweeps, key=lambda s: s["sweep_number"]), open(os.path.join(o, "sweeps.json"), "w"), indent=1)
    models = query(f"model::NeuronalModel,rma::criteria,[specimen_id$eq{cell}],rma::include,neuronal_model_template,"
                   "well_known_files(well_known_file_type),neuronal_model_runs")
    meta = []
    for m in models:
        name = m["neuronal_model_template"]["name"]
        meta.append(dict(id=m["id"], name=name, runs=[dict(id=r["id"], ev=r.get("explained_variance_ratio"),
                                                          rms=r.get("rms")) for r in m.get("neuronal_model_runs", [])]))
        lvl = re.match(r"(\d) ", name)
        if lvl:                                                   # GLIF levels 1-5
            for wf in m["well_known_files"]:
                if wf["path"].endswith("neuron_config.json"):
                    download(API + wf["download_link"],
                             os.path.join(o, "glif", f"model_{lvl.group(1)}_{m['id']}_neuron_config.json"))
    sp.pop("ephys_result", None)
    sp.pop("neuron_reconstructions", None)
    json.dump(dict(specimen=sp, models=meta), open(os.path.join(o, "meta.json"), "w"), indent=1)
    if nandi:
        download(NANDI.format(c=cell), os.path.join(o, "allactive", "fit_parameters.json"))
        json.dump(dict(specimen_id=int(cell),
                       fit_source=f"github.com/AllenInstitute/All-active-Manuscript assets/mouse_models/{cell}/hof_models/"
                                  f"hof_param_{cell}_0.json (best hall-of-fame all-active model, Nandi et al. 2022)",
                       swc_source="Allen Cell Types Database 3DNeuronReconstruction via api.brain-map.org"),
                  open(os.path.join(o, "allactive", "source.json"), "w"), indent=1)
    print(cell, [x["name"] for x in meta], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwb-dir", required=True)
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--nandi", action="store_true", help="also fetch the Nandi et al. all-active parameters")
    a = ap.parse_args()
    for c in a.cells:
        fetch(c, a.nwb_dir, a.nandi)


if __name__ == "__main__":
    main()
