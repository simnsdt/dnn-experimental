import keras2tpu
import onnx2ncs

def main():
    # Loads models, converts them and handles deployment on specific hardware.

    # TODO: Implement argument parsing for model names, e.g. call python3 bench.py --model ResNet50
    modelName = "ResNet50"
    # TPU Pipeline:
    keras2tpu.prepare(modelName)
    keras2tpu.compile(modelName)
    keras2tpu.copy(modelName)
    keras2tpu.bench(modelName, 50)
    keras2tpu.retrieveResults()

    # NCS Pipeline:
    onnx2ncs.prepare(modelName)
    onnx2ncs.bench(modelName, "CPU", 50)
    onnx2ncs.bench(modelName,"MYRIAD",50)

    print("FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
