#!/bin/bash
# Grating trials 1-9 for the Fano factor (step 2): full GLIF network per trial -> replay streams -> hybrid prep ->
# hybrid run with kernel synapses at the tuned settings. Needs ~13 GB for the full network: run alone.
#   V1=/path/v1_point W=/path/workdir bash run_trials.sh
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"; export PYTHONPATH=$ROOT TF_CPP_MIN_LOG_LEVEL=3
V1=${V1:-/home/user/v1/v1_point}; W=${W:-/home/user/v1/trials}; X=${X:-/home/user/v1/X}
GAIN=${GAIN:-1.18}; PCIN=${PCIN:-"Pvalb=1.06,Sst=1.04,Htr3a=1.04"}
CORR=${CORR:-experiments/kernel_calibration/results/487661754/calibration.json}
for k in ${TRIALS:-1 2 3 4 5 6 7 8 9}; do
  T=$W/t$k; mkdir -p $T
  [ -f $T/hyb/spikes.h5 ] && continue
  sed "s/spikes.trial_0.csv/spikes.trial_$k.csv/g" $V1/config.json > $V1/config_t$k.json
  [ -f $T/full/spikes.h5 ] || python -m circuit.simulate full --config $V1/config_t$k.json --out $T/full --threads 4 --tstop 3000
  [ -f $T/streams/v1ext_spikes.h5 ] || python -m circuit.simulate streams --full-spikes $T/full/spikes.h5 --extract-dir $X --out $T/streams --tstop 3000
  rm -rf $T/full
  [ -f $T/H/prepare.json ] || python -m circuit.hybrid prepare --config $V1/config_t$k.json --extract-dir $X --v1ext-spikes $T/streams/v1ext_spikes.h5 --out $T/H
  python -m circuit.hybrid run --prep $T/H --standard-dir Standard --gain $GAIN --pc-to-in-gain "$PCIN" --corr $CORR --tstop 3000 --out $T/hyb
  rm -rf $T/H
done
