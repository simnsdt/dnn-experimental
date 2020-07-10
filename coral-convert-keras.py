import keras
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

def dlModel(IMG_SHAPE):
    print("Loading model...")
    model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                include_top=False, 
                                                weights='imagenet')
    model.trainable = False
    return model

def keras2tflite_quant(inputKeras, outputTFlite):
    converter = tf.lite.TFLiteConverter.from_keras_model(inputKeras)
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # These set the input and output tensors to uint8 (added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # And this sets the representative dataset so we can quantize the activations
    print("Creating representative dataset for quantizing...")
    converter.representative_dataset = representative_data_gen
    print("Quantizing model to int8....")
    tflite_model = converter.convert()

    with open(outputTFlite, 'wb') as f:
        f.write(tflite_model)


def main():
    # Filenames:
    tfliteFilename = 'vgg19_tensorFlow.tflite'

    # Conversion Pipeline:
    IMG_SHAPE = (224, 224, 3)
    kerasModel = dlModel(IMG_SHAPE)
    keras2tflite_quant(kerasModel, tfliteFilename)


if __name__ == "__main__":
    main()
