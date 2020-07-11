import os
import subprocess
from openvino.inference_engine import IECore

def prepare(name):
    filename = name + ".onnx"
    modelOptimizer = os.path.expandvars("$HOME/intel/openvino/deployment_tools/model_optimizer/mo.py")

    subprocess.run([modelOptimizer, "--input_model", filename])

def deploy():
    # TODO: Implement benchmarking
    ie = IECore()

    # Model paths:
    model_xml = "$HOME/intel/openvino/deployment_tools/model_optimizer/model.xml"
    model_bin = "$HOME/intel/openvino/deployment_tools/model_optimizer/model.bin"

    print("Loading model...")
    net = ie.read_network(model = model_xml, weights=model_bin)

    print("Deploying model to NCS...")
    exec_net = ie.load_network(network=net, device_name="MYRIAD")


