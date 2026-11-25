"""List the Allen Cell Types Database VISp L2/3 spiny (pyramidal) mouse cells for the population fit.

Run from "Isolated L2_3 circuit/":
    python experiments/population/select_cells.py

Source: api.brain-map.org model::ApiCellTypesSpecimenDetail (the table behind celltypes.brain-map.org).
Keeps structure VISp2/3, dendrite type spiny, Mus musculus. Writes population/cells.csv with the specimen id,
Cre line, reporter status, soma depth (csl__normalized_depth = normalized cortical depth, 0 = pia, 1 = white
matter; csl x/y/z in CCF microns), Allen's own ephys features (rheobase, Ri, tau, Vrest) and which published
models exist (GLIF, perisomatic, all-active).
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = ("https://api.brain-map.org/api/v2/data/query.json?criteria=model::ApiCellTypesSpecimenDetail,"
       "rma::criteria,[donor__species$eq%27Mus%20musculus%27],rma::options[num_rows$eqall]")
COLS = [("cell", "specimen__id"), ("name", "specimen__name"), ("line", "line_name"),
        ("reporter", "cell_reporter_status"), ("depth_norm", "csl__normalized_depth"),
        ("ccf_x", "csl__x"), ("ccf_y", "csl__y"), ("ccf_z", "csl__z"), ("hemisphere", "specimen__hemisphere"),
        ("apical", "tag__apical"), ("reconstruction", "nr__reconstruction_type"),
        ("rheobase_pA", "ef__threshold_i_long_square"), ("ri_MOhm", "ef__ri"), ("tau_ms", "ef__tau"),
        ("vrest_mV", "ef__vrest"), ("fi_slope", "ef__f_i_curve_slope"), ("adaptation", "ef__adaptation"),
        ("glif", "m__glif"), ("biophys_perisomatic", "m__biophys_perisomatic"),
        ("biophys_all_active", "m__biophys_all_active"), ("donor_age", "donor__age")]


def main():
    rows = json.load(urllib.request.urlopen(URL, timeout=300))["msg"]
    sel = [r for r in rows if r["structure__acronym"] == "VISp2/3" and r["tag__dendrite_type"] == "spiny"]
    sel.sort(key=lambda r: r["specimen__id"])
    with open(os.path.join(HERE, "cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in COLS])
        for r in sel:
            w.writerow(["" if r.get(k) is None else r[k] for _, k in COLS])
    print(len(sel), "cells ->", os.path.join(HERE, "cells.csv"))


if __name__ == "__main__":
    main()
