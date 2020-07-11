import subprocess
import tensorflow as tf
import os


def prepare(name):

    def _keras2tflite_quant():
        converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
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
        converter.representative_dataset = _representative_data_gen
        print("Quantizing model to int8....")
        tflite_quant_model = converter.convert()
        return tflite_quant_model

    def _dlModel():
        print("Loading keras model "+name)
        IMG_SHAPE = (224, 224, 3)
        if (name == "ResNet50"):
            model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                        include_top=False, 
                                                        weights='imagenet')
        elif (name == "VGG19"):
            model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                        include_top=False, 
                                                        weights='imagenet')
        else:
            print("Model {} not supported!".format(name))
            exit()

        model.trainable = False
        return model

    def _representative_data_gen():
        # Get sample data:
        _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        zip_file = tf.keras.utils.get_file(origin=_URL, 
                                        fname="flower_photos.tgz", 
                                        extract=True)

        flowers_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
        dataset_list = tf.data.Dataset.list_files(flowers_dir + '/*/*')
        for i in range(100):
            IMG_SIZE = 224
            image = next(iter(dataset_list))
            image = tf.io.read_file(image)
            image = tf.io.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
            image = tf.cast(image / 255., tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]
    
    # Load keras model:
    kerasModel = _dlModel()
    # Convert to quantized tflite:
    tflite_quant_model = _keras2tflite_quant()
    # Save quantized model:
    with open(name+"_quant.tflite", 'wb') as f:
        f.write(tflite_quant_model)

def deploy(name):
    # TODO: Copy to TPU
    # TODO: Implement benchmark
    subprocess.run(["edgetpu_compiler",name + "_quant.tflite"])