import os
import subprocess
from openvino.inference_engine import IECore

def prepare(modelName):
# Calls model optimizer to convert *.onnx model to optimized *.bin/*.xml files.
    def _dlModel():
    # Downloads the *.onnx models if the files are not existent in the directory.
        if not os.path.isfile(filename):
            if modelName == "ResNet50":
                subprocess.run(
                    ["curl", "https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz", "--output", "ResNet50.onnx"])
            elif modelName == "VGG19":
                subprocess.run(["curl", "https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz", "--output", "VGG19.onnx" ])
                
    filename = modelName + ".onnx"
    _dlModel()
    modelOptimizer = os.path.expandvars("$HOME/intel/openvino/deployment_tools/model_optimizer/mo.py")
    subprocess.run([modelOptimizer, "--input_model", filename, "--output_dir", "./"])

def bench(modelName, device, batch_size):
    model_xml = modelName+".xml"
    sample_jpg = "sample.jpg"
    classification =  os.path.expandvars("./ncs/classification_sample.py")
    for i in range(0,batch_size):
        subprocess.run(["python3", classification, "-m", model_xml, "-i", sample_jpg, "-d", device])


