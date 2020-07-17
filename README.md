# Benchmarking Pipeline
Experimental pipeline for converting, deploying, executing and benchmarking different DNN-Models on CPU, Intel NCS or Coral TPU Dev Board.

## Requirements Host
* Tensorflow Version >=2.0.0 and <=2.2.0 installed
  > pip3 install tensorflow==2.0.0b1
* python >= 3.5 and <= 3.7 installed on host (should be the case when using Ubuntu 18.04)
* Intel Movidius Setup according to https://software.intel.com/content/www/us/en/develop/articles/get-started-with-neural-compute-stick.html
  * Use OpenVINO Toolkit 2020R3 instead of 2019R1!
* OpenVINO Toolkit initialized (source ~/intel/openvino/bin/setupvars.sh before running)
* edgetpu_compiler installed according to https://coral.ai/docs/edgetpu/compiler/

## Requirements Coral Dev Board
* Accessible via mdt - follow instructions here: https://coral.ai/docs/dev-board/get-started/
  * Check via: mdt shell
* TFLite runtime 2.1.0 installed on dev board (should be the case if instructions were followed correctly)

## How to use
1. git clone https://github.com/simnsdt/dnn-experimental.git
2. cd dnn-experimental
3. ./runBenchmark.sh

## Options
* You can use NOCOPY=--nocopy (in runBenchmark.sh) to skip copying the prerequisites for the benchmark after the first run. Remember to reactivate it (NOCOPY="") when changing to a model not benchmarked before.


* Modify batch size and model in runBenchmark.sh

Supported models: ResNet50, VGG19


Tested using Ubuntu 18.04 LTS, Tensorflow 2.0.0b1, Python 3.6.9, OpenVINO Toolkit 2020R3 and the linked documentations.
