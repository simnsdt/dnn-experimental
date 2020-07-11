# Benchmarking Pipeline:

Pipeline for converting, deploying, executing and benchmarking different DNN-Models on CPU, Intel NCS or Coral TPU Dev Board.

## Requirements:
* *.onnx file for selected model placed next to bench.py
* ONNX file named according to model name, e.g. ResNet50.onnx
* TF Version >=2.0.0 and <=2.2.0 installed
* python >= 3.5 and <= 3.7 installed
* edgetpu_compiler installed according to google docs
* OpenVINO Toolkit installed in default path ~/intel/openvino*
* OpenVINO Toolkit initialized (source ~/intel/openvino/bin/setupvars.sh)


Tested on Ubuntu 18.04 LTS
## How to use:
> python3 bench.py
