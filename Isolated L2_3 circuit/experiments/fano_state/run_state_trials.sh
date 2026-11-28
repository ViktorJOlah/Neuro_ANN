#!/bin/bash
# State-driven trials: same pipeline as kernel_rerun/run_trials.sh, but LGN trial k comes from the measured
# state drive (code/make_state_drive.py) instead of the Billeh trial k. Background trial k as before.
#   DRIVE=state_756029989 bash run_state_trials.sh
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"; export PYTHONPATH=$ROOT TF_CPP_MIN_LOG_LEVEL=3
V1=${V1:-/home/user/v1/v1_point}; DRIVE=${DRIVE:-state_756029989}; W=${W:-/home/user/v1/$DRIVE}; X=${X:-/home/user/v1/X}
GAIN=${GAIN:-1.18}; PCIN=${PCIN:-"Pvalb=1.06,Sst=1.04,Htr3a=1.04"}
CORR=${CORR:-experiments/kernel_calibration/results/487661754/calibration.json}
LGN=full3_production_3.0sec_SF0.04_TF2.0_ori315.0_c80.0_gs0.5
for k in ${TRIALS:-0 1 2 3 4 5 6 7 8 9}; do
  T=$W/t$k; mkdir -p $T
  [ -f $T/hyb/spikes.h5 ] && continue
  C=$V1/config_${DRIVE}_t$k.json
  sed -e "s#$LGN/spikes.trial_0.csv#$DRIVE/spikes.trial_$k.csv#" -e "s#bkg_spikes_n1_fr1000_dt0.25/spikes.trial_0.csv#bkg_spikes_n1_fr1000_dt0.25/spikes.trial_$k.csv#" $V1/config.json > $C
  [ -f $T/full/spikes.h5 ] || python -m circuit.simulate full --config $C --out $T/full --threads 4 --tstop 3000
  [ -f $T/streams/v1ext_spikes.h5 ] || python -m circuit.simulate streams --full-spikes $T/full/spikes.h5 --extract-dir $X --out $T/streams --tstop 3000
  mkdir -p $T/fullrates; python - $T/full/spikes.h5 $T/fullrates/pop_spikes.npz <<'PY' || true
import sys, h5py, numpy as np
f = h5py.File(sys.argv[1]); g = f["spikes"][list(f["spikes"].keys())[0]]
np.savez_compressed(sys.argv[2], node_ids=g["node_ids"][:], timestamps=g["timestamps"][:])
PY
  rm -rf $T/full
  [ -f $T/H/prepare.json ] || python -m circuit.hybrid prepare --config $C --extract-dir $X --v1ext-spikes $T/streams/v1ext_spikes.h5 --out $T/H
  python -m circuit.hybrid run --prep $T/H --standard-dir Standard --gain $GAIN --pc-to-in-gain "$PCIN" --corr $CORR --tstop 3000 --out $T/hyb
  rm -rf $T/H
done
echo STATE_TRIALS_DONE
