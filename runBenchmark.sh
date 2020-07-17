#!/bin/bash

source ~/intel/openvino/bin/setupvars.sh

MODEL=VGG19
NOCOPY=--nocopy

BATCHSIZE=1
python3 bench.py -m $MODEL -b $BATCHSIZE $NOCOPY
BATCHSIZE=32
python3 bench.py -m $MODEL -b $BATCHSIZE $NOCOPY
BATCHSIZE=64
python3 bench.py -m $MODEL -b $BATCHSIZE $NOCOPY
