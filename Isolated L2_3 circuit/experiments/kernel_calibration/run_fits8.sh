#!/bin/bash
# the 7 non-Standard fits of the 8-fit Aim 2 set (Standard = results/487661754), into results/fits8_<cell>
cd "$(dirname "$0")"; export TF_CPP_MIN_LOG_LEVEL=3
while read c d; do [ -f results/fits8_$c/calibration.json ] || python calibrate.py --cell $c --model-dir $d --tag fits8_$c > logs/fits8_$c.log 2>&1; done < fits8.txt
