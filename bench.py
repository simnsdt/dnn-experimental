from keras2tpu import prepare
from onnx2ncs import prepare
from onnx2ncs import deploy

def main():
    name = "ResNet50"
    keras2tpu.prepare(name)

if __name__ == "__main__":
    main()
