#!/usr/bin/env python
from __future__ import print_function
import sys
import time
import os
import argparse

import numpy as np
import cv2
from openvino.inference_engine import IECore


def build_argparser():
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    parser.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    parser.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    parser.add_argument("-b","--batchsize" ,help="Optional. Number of inference runs", default=1, type=int)

    return parser


def main():
    # Prepare arguments:
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Read model
    print("Creating Inference Engine")
    ie = IECore()
    print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    print("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    # Read and pre-process input
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.input[i])
        if image.shape[:-1] != (h, w):
            print("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
        
    # Loading model to the plugin
    print("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start inference with time measurement
    print("Starting OpenVINO inference...")
    times_txt = "results-"+args.device+"-{}.txt".format(args.batchsize)
    if os.path.exists(times_txt):
        os.remove(times_txt)
    for i in range(0, args.batchsize):
        start = time.perf_counter()
        exec_net.infer(inputs={input_blob: images})
        inference_time = (time.perf_counter() - start)*1000
        text_file = open(times_txt, "a")
        text_file.write(str(inference_time)+'\n')
        text_file.close()
        print("Inference #{} on {} done...".format(i, args.device))


if __name__ == '__main__':
    sys.exit(main() or 0)
