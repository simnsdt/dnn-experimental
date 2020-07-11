import keras2tpu
import onnx2ncs

def main():
    # REQUIREMENTS: 
    # *.onnx file for selected model placed next to bench.py
    # ONNX file named according to model name, e.g. ResNet50.onnx
    # TF Version >=2.0.0 and <=2.2.0 installed
    # python >= 3.5 and <= 3.7 installed
    # edgetpu_compiler installed according to google docs
    # OpenVINO Toolkit installed in default path ~/intel/openvino*
    # OpenVINO Toolkit initialized (source ~/intel/openvino/bin/setupvars.sh)


    # TODO: Implement argument parsing for model names, e.g. call python3 bench.py --model ResNet50
    model1 = "ResNet50"
    model2 = "VGG19"
    # TPU Pipeline:
    keras2tpu.prepare(model1)
    keras2tpu.deploy(model1)

    # NCS Pipeline:
    onnx2ncs.prepare(model1)
    onnx2ncs.deploy("CPU")
    #onnx2ncs.deploy("MYRIAD")

    print("FINISHED SUCCESSFULLY!")
if __name__ == "__main__":
    main()
