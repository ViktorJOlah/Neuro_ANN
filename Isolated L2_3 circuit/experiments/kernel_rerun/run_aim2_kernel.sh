#!/bin/bash
# Aim 2 controls in kernel mode. Usage: run_a2.sh <job>
cd "/home/user/neuro_ann/Isolated L2_3 circuit"; export PYTHONPATH=$PWD TF_CPP_MIN_LOG_LEVEL=3
H=/home/user/v1/H; HH=/home/user/v1/H_homog_in; REF=/home/user/v1/runs/l23_reference_spikes.h5; O=/home/user/v1/A2
PCIN="Pvalb=1.06,Sst=1.04,Htr3a=1.04"; CORR=experiments/kernel_calibration/results/487661754/calibration.json
P=experiments/aim2_controls/populations; PL=experiments/aim2_controls/populations_library
TGT=1.378
tune() { python -m circuit.hybrid tune --prep $1 --standard-dir Standard --reference $REF --pc-to-in-gain "$PCIN" --target $TGT --tstop 3000 --tol 0.03 --gains $3 "${@:4}" --out $O/$2; }
best() { python -c "import json;print(json.load(open('$O/$1/tune.json'))['best']['gain'])"; }
case $1 in
  shuf)  tune $H tune_shuffled 1.24 --population $P/shuffled_kernel.json ;;
  depth) python -m circuit.hybrid run --prep $H --standard-dir Standard --gain $(best tune_shuffled) --pc-to-in-gain "$PCIN" --population $P/depth_kernel.json --tstop 3000 --out $O/final_depth ;;
  rheo)  tune $H tune_rheo 1.0 --population $P/rheo_matched_kernel.json ;;
  hin)   python -m circuit.hybrid tune --prep $HH --standard-dir Standard --reference $REF --pc-to-in-gain "$PCIN" --target 1.378 --tstop 3000 --tol 0.03 --gains 1.18 --corr $CORR --out $O/tune_homog_in ;;
  shin)  python -m circuit.hybrid run --prep $HH --standard-dir Standard --gain $(best tune_shuffled) --pc-to-in-gain "$PCIN" --population $P/shuffled_kernel.json --tstop 3000 --out $O/final_shuffled_homog_in ;;
  lshuf) tune $H lib_tune_shuffled 0.9 --population $PL/shuffled_kernel.json ;;
  ldepth) python -m circuit.hybrid run --prep $H --standard-dir Standard --gain $(best lib_tune_shuffled) --pc-to-in-gain "$PCIN" --population $PL/depth_kernel.json --tstop 3000 --out $O/lib_final_depth ;;
  lrheo) tune $H lib_tune_rheo 0.9 --population $PL/rheo_matched_kernel.json ;;
esac
case $1 in
  rheodc)  tune $H tune_rheo_dc 1.3 --population $P/rheo_matched_dc_kernel.json ;;
  lrheodc) tune $H lib_tune_rheo_dc 1.4 --population $PL/rheo_matched_dc_kernel.json ;;
esac
