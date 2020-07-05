import torch
import pytorchcv
import tensorflow as tf
from pytorchcv import model_provider
import os

def representative_data_gen():
  # Get sample data:
  _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
  zip_file = tf.keras.utils.get_file(origin=_URL, 
                                    fname="flower_photos.tgz", 
                                    extract=True)

  flowers_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

  IMAGE_SIZE = 224
  dataset_list = tf.data.Dataset.list_files(flowers_dir + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

def onnx2pb(inputONNX, outputPB):
  model = model_provider._models['vgg19'](pretrained=True)
  # This is for standard image net
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  torch_out = model(x)
  torch.onnx.export(model, x, inputONNX, export_params=True, input_names=['input'], output_names=['output'])
  print("CONVERTING ONNX TO PB VIA COMMAND LINE ")
  cmd = "onnx-tf convert -i {} -o {}".format(inputONNX, outputPB)
  os.system(cmd)
  print("DONE CONVERTING ONNX TO PB ")

def pb2tflite_quant(inputPB,outputTFLite):
  # Create converter:
  converter = tf.lite.TFLiteConverter.from_frozen_graph(inputPB, input_arrays=['input'], output_arrays=['output'])
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # Define inpout/output tensor type:
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  # Define representative dataset:
  converter.representative_dataset = representative_data_gen
  tf_lite_model = converter.convert()
  open(outputTFLite, 'wb').write(tf_lite_model)

def main():
  onnx2pb('torch_vgg19.onnx', 'torch_vgg19.pb')
  pb2tflite_quant('torch_vgg19.pb','vgg19_tensorFlow.tflite')

if __name__ == "__main__":
    main()
