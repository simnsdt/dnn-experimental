#!/bin/bash

source ~/intel/openvino/bin/setupvars.sh
MODEL=VGG19
BATCHSIZE=1
python3 bench.py -m $MODEL -b $BATCHSIZE
BATCHSIZE=32
python3 bench.py -m $MODEL -b $BATCHSIZE
BATCHSIZE=64
python3 bench.py -m $MODEL -b $BATCHSIZE
