import keras2tpu
import onnx2ncs
import argparse

def main():
    # Loads models, converts them and handles deployment on specific hardware.

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Runs benchmark for selected model on TPU, VPU and CPU.')
    parser.add_argument('-m', '--model', help='Name of the model to benchmark.', required=True,type=str)
    parser.add_argument('-b','--batch_size', help='Number of inferences per device.', default=1, type=int)
    args = parser.parse_args()
    
    modelName = args.model
    batchSize = args.batch_size
    if modelName != "ResNet50" and modelName != "VGG19":
        print("Please select supported model!")
        exit()
        
    # TPU Pipeline:
    keras2tpu.prepare(modelName)
    keras2tpu.compile(modelName)
    keras2tpu.copy(modelName)
    keras2tpu.bench(modelName, batchSize)
    keras2tpu.retrieveResults(batchSize)

    # NCS Pipeline:
    onnx2ncs.prepare(modelName)
    onnx2ncs.bench(modelName, "CPU", batchSize)
    onnx2ncs.bench(modelName,"MYRIAD",batchSize)

    print("FINISHED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
