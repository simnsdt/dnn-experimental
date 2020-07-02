# load the model saved in onnx format
import onnx
from onnx_tf.backend import prepare

# Load ONNX
model_onnx = onnx.load('/home/simon/Downloads/vgg19/model.onnx')

# prepare model for exporting to tensorFlow using tensorFlow backend
model_tf = prepare(model_onnx)

# export tensorFlow backend to tensorflow tf file
model_tf.export_graph('/home/simon/Downloads/vgg19.pb')
