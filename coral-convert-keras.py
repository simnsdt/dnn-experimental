import onnx2keras
from onnx2keras import onnx_to_keras
import keras
import onnx
import torch
import pytorchcv
from pytorchcv import model_provider
import tensorflow as tf
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

def dlModel(modelName, outputONNX):
  model = model_provider._models[modelName](pretrained=True)
  # This is for standard image net
  x = torch.randn(1, 3, 224, 224, requires_grad=True)
  torch_out = model(x)
  torch.onnx.export(model, x, outputONNX, export_params=True, input_names=['input'], output_names=['output'])

def onnx2keras(inputONNX, outputKeras):
  print("CONVERTING ONNX TO KERAS...")
  model = onnx.load(inputONNX)
  k_model = onnx_to_keras(model, ['input'])
  keras.models.save_model(k_model,outputKeras ,overwrite=True,include_optimizer=True)
  print("DONE CONVERTING ONNX TO KERAS!")

def keras2tflite_quant(inputKeras, outputTFLite):
  kerasModel = tf.keras.models.load_model(inputKeras)

  converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
  # This enables quantization
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.int8]
  # This ensures that if any ops can't be quantized, the converter throws an error
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # These set the input and output tensors to uint8 (added in r2.3)
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8
  # And this sets the representative dataset so we can quantize the activations
  converter.representative_dataset = representative_data_gen
  tflite_model = converter.convert()

  with open(outputTFlite, 'wb') as f:
    f.write(tflite_model)


def main():

  # Filenames:
  onnxFilename = 'torch_vgg19.onnx'
  kerasFilename = 'keras_vgg19.h5'
  tfliteFilename = 'vgg19_tensorFLow.tflite'

  # Conversion Pipeline:
  dlModel('vgg19', onnxFilename)
  onnx2keras(onnxFilename, kerasFilename)
  keras2tflite_quant(kerasFilename,tfliteFilename)


if __name__ == "__main__":
    main()
