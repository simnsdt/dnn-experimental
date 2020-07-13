import keras2tpu
import onnx2ncs

def main():
# Loads models, converts them and handles deployment on specific hardware.

    # TODO: Implement argument parsing for model names, e.g. call python3 bench.py --model ResNet50
    model1 = "ResNet50"
    model2 = "VGG19"
    # TPU Pipeline:
    keras2tpu.prepare(model1)
    keras2tpu.deploy(model1)

    # NCS Pipeline:
    onnx2ncs.prepare(model1)
    onnx2ncs.bench(model1,"CPU",2)
    #onnx2ncs.deploy("MYRIAD")

    print("FINISHED SUCCESSFULLY!")
if __name__ == "__main__":
    main()
