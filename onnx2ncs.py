import os
import subprocess
from openvino.inference_engine import IECore

def prepare(modelName):
    # Calls model optimizer to convert *.onnx model to optimized *.bin/*.xml files.
    filename = modelName + ".onnx"
    modelOptimizer = os.path.expandvars("$HOME/intel/openvino/deployment_tools/model_optimizer/mo.py")

    subprocess.run([modelOptimizer, "--input_model", filename, "--output_dir", "./"])

def bench(modelName, device, batch_size):
	model_xml = modelName+".xml"
	sample_jpg = "sample.jpg"
	classification =  os.path.expandvars("$HOME/intel/openvino/inference_engine/samples/python/classification_sample/classification_sample.py")
	for i in range(0,batch_size):
		subprocess.run(["python3", classification, "-m", model_xml, "-i", sample_jpg, "-d", device])


