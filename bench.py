import keras2tpu
import onnx2ncs

def main():
    model1 = "ResNet50"
    keras2tpu.prepare(model1)
    keras2tpu.deploy(model1)
    onnx2ncs.prepare(model1)

if __name__ == "__main__":
    main()
