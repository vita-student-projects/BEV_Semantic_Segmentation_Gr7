#!/usr/bin/env bash


CONFIG=$1
master_port=12366
nnodes=1
master_addr="10.233.66.255"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --master_port=${master_port} --nproc_per_node=1  \
        --nnodes=${nnodes} --node_rank=$2 --master_addr=${master_addr} \
        $(dirname "$0")/train.py \
        --config $CONFIG --launcher pytorch ${@:3} --deterministic
