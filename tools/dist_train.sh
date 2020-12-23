#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --master_port 2002 --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:3}
