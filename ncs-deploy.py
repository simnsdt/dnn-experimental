from openvino.inference_engine import IECore
ie = IECore()

# Model paths:
model_xml = "/home/simonschmidt/intel/openvino/deployment_tools/model_optimizer/model.xml"
model_bin = "/home/simonschmidt/intel/openvino/deployment_tools/model_optimizer/model.bin"

print("Loading model...")
net = ie.read_network(model = model_xml, weights=model_bin)

print("Deploying model to NCS...")
exec_net = ie.load_network(network=net, device_name="MYRIAD")


