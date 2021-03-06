import subprocess
import tensorflow as tf
import os


def prepare(name):
    # Loads pretrained keras model and converts it to full int8 *.tflite model.
    # The quantized model is then saved as [modelname]_quant.tflite.

    def _keras2tflite_quant():
    # Handles quantization to int8
    
        converter = tf.lite.TFLiteConverter.from_keras_model(kerasModel)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("Creating representative dataset for quantizing...")
        converter.representative_dataset = _representative_data_gen
        print("Quantizing pretrained keras model to int8....")
        tflite_quant_model = converter.convert()
        return tflite_quant_model

    def _dlModel():
    # Loads pretrained keras models
    
        print("Trying to load pretrained keras model ({})...".format(name))
        IMG_SHAPE = (224, 224, 3)
        if (name == "ResNet50"):
            model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')
        elif (name == "VGG19"):
            model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                                        include_top=False,
                                                        weights='imagenet')
        model.trainable = False
        return model

    def _representative_data_gen():
    # Gets sample data necessary for quantizing the weights
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

    filename = name+"_quant.tflite"
    if not os.path.isfile(filename):
        # Load keras model:
        kerasModel = _dlModel()
        # Convert to quantized tflite:
        tflite_quant_model = _keras2tflite_quant()
        # Save quantized model:
        with open(filename, 'wb') as f:
            f.write(tflite_quant_model)


def compile(modelName):
# Compiles the model for TPU.
# Saves the compiled model according to edgetpu_compiler default settings (*_edgtpu.tflite).
    
    filename = modelName+"_quant_edgetpu.tflite"
    if not os.path.isfile(filename):
        subprocess.run(["edgetpu_compiler", modelName + "_quant.tflite"])

def copyPrerequisites(modelName):
# Copy prerequisites to TPU
    subprocess.run(["mdt", "push", "sample.jpg"])
    subprocess.run(["mdt", "push", "./tpu/classify_image.py"])
    subprocess.run(["mdt", "push", "./tpu/classify.py"])
    subprocess.run(["mdt", "push", modelName+"_quant_edgetpu.tflite"])

def bench(modelName,batchSize):
# Runs classification and time measurements
    tfliteFilename = modelName+"_quant_edgetpu.tflite"
    subprocess.run(["mdt", "exec", "python3 ~/classify_image.py --model ~/{} --input ~/sample.jpg --batch_size {}".format(tfliteFilename, batchSize)])
    
def retrieveResults(batchSize):
# Copies results from dev board back to host
    subprocess.run(["mdt", "pull", "results-TPU-{}.txt".format(batchSize), "./"])
    subprocess.run(["mdt","exec","rm", "results-TPU-{}.txt".format(batchSize)])
