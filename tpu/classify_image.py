import argparse
import time
import  os

from PIL import Image

import classify
import tflite_runtime.interpreter as tflite

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate('libedgetpu.so.1')
      ])


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument(
      '-b', '--batch_size', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  size = classify.input_size(interpreter)
  image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
  classify.set_input(interpreter, image)

  times_txt = "results-TPU-{}.txt".format(args.batch_size)
  if os.path.exists(times_txt):
        os.remove(times_txt)
  for i in range(0,args.batch_size):
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start)*1000
    text_file = open(times_txt, "a")
    text_file.write(str(inference_time)+'\n')
    text_file.close()
    print("Inference #{} on TPU done...".format(i))

if __name__ == '__main__':
  main()
