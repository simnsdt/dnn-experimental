import keras2tpu
import onnx2ncs

def main():
    name = "ResNet50"
    keras2tpu.prepare(name)
    onnx2ncs.prepare(name)
if __name__ == "__main__":
    main()
