#!/bin/bash
# calibrate every library cell, 2 at a time (restartable: skips cells with a calibration.json)
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3
grep -v '^$' library_cells.txt | xargs -P ${JOBS:-2} -I{} bash -c '[ -f results/{}/calibration.json ] || python calibrate.py --cell {} > logs/{}.log 2>&1'
