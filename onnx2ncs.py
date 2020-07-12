import os
import subprocess
from openvino.inference_engine import IECore

def prepare(name):
    # Calls model optimizer to convert *.onnx model to optimized *.bin/*.xml files.
    filename = name + ".onnx"
    modelOptimizer = os.path.expandvars("$HOME/intel/openvino/deployment_tools/model_optimizer/mo.py")

    subprocess.run([modelOptimizer, "--input_model", filename, "--output_dir", "./"])

def deploy(name, device):
    # Loads optimized model to selected device.

    # TODO: Implement inference/benchmarking
    ie = IECore()

    # Model paths:
    model_xml = name+".xml"
    model_bin = name+".bin"

    print("Loading optimized ONNX model...")
    net = ie.read_network(model = model_xml, weights=model_bin)

    print("Deploying optimized ONNX model to {}...".format(device))
    exec_net = ie.load_network(network=net, device_name=device)


