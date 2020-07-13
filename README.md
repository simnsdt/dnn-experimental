# Benchmarking Pipeline
Work in progress.

Experimental pipeline for converting, deploying, executing and benchmarking different DNN-Models on CPU, Intel NCS or Coral TPU Dev Board.

## Requirements Host
* *.onnx file for selected models placed next to bench.py
* ONNX file named according to model name, e.g. ResNet50.onnx
* TF Version >=2.0.0 and <=2.2.0 installed
* python >= 3.5 and <= 3.7 installed on host
* edgetpu_compiler installed according to google docs
* OpenVINO Toolkit installed in default path ~/intel/openvino*
* OpenVINO Toolkit initialized (source ~/intel/openvino/bin/setupvars.sh)
* sample.jpg next to bench.py

## Requirements Coral Dev Board
* Accessible via mdt - follow instructions here: https://coral.ai/docs/dev-board/get-started/
* TFLite runtime 2.1.0 installed on dev board (should be the case if instructions were followed correctly)


Tested on Ubuntu 18.04 LTS
## How to use
> python3 bench.py
