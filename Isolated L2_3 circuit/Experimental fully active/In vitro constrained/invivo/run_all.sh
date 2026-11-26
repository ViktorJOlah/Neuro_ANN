#!/bin/bash
# All in vivo-like runs used in the README (3 in parallel, ~1 min each with dendritic recording)
cd "$(dirname "$0")/.."
OUT=${1:-results/invivo}
run(){ python3 invivo/run_invivo.py --out "$OUT" "$@" 2>&1 | grep -v -i -E "display|graphics"; }
export -f run; export OUT
cat <<LIST | xargs -P 3 -I{} bash -c "run {}"
--cell 82141 --cond control --td-rate 0
--cell 82141 --cond HCN --td-rate 0
--cell 82141 --cond NMDA --td-rate 0
--cell 82141 --cond 4APd --td-rate 0
--cell 82141 --cond control --td-rate 2
--cell 82141 --cond control --td-rate 5
--cell 82141 --cond control --td-rate 10
--cell 82141 --cond control --td-rate 20
--cell 82141 --cond HCN --td-rate 5
--cell 82141 --cond HCN --td-rate 10
--cell 82141 --cond NMDA --td-rate 5
--cell 82141 --cond NMDA --td-rate 10
--cell 82141 --cond 4APd --td-rate 5
--cell 82141 --cond 4APd --td-rate 10
--cell 81559 --cond control --td-rate 0
--cell 80733 --cond control --td-rate 0
--cell 82141 --cond control --td-rate 0 --seed 1
--cell 82141 --cond control --td-rate 5 --seed 1
LIST
