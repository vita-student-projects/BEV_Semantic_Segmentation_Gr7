#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
CARVIEW=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/inference.py --config $CONFIG --checkpoint $CHECKPOINT --show_dir visual_results --car_view $CARVIEW
