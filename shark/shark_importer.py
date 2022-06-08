# Lint as: python3
"""SHARK Importer"""

import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy as np
import os
import sys
import tensorflow.compat.v2 as tf
import time
import urllib.request
from shark.shark_inference import SharkInference

targets = {
  'dylib' : 'dylib-llvm-aot',
  'vulkan' : 'vulkan-spirv',
  'cuda' : 'cuda',
}

#iree_utils.py IREE_DEVICE_MAP
# configs = {
#   'dylib' : 'dylib',
#   'vulkan' : 'vulkan',
#   'cuda' : 'cuda',
# }

class GenerateInputSharkImporter():
  def __init__(self, input_details, model_source_hub="tfhub"):
    self.input_details = input_details
    self.model_source_hub = model_source_hub

  def generate_inputs(self):
    args = []
    for input in self.input_details:
      print("\t%s, %s", str(input["shape"]), input["dtype"].__name__)
      args.append(np.zeros(shape=input["shape"], dtype=input["dtype"]))
    return args

class SharkImporter:
  def __init__(self,
               model_path,
               model_type: str="tflite",
               model_source_hub: str="tfhub",
               device: str = None,
               dynamic: bool = False,
               jit_trace: bool = False,
               benchmark_mode: bool = False):
    self.model_path = model_path
    self.model_type = model_type
    self.model_source_hub = model_source_hub
    self.device = device
    self.dynamic = dynamic
    self.jit_trace = jit_trace
    self.benchmark_mode = benchmark_mode
    self.inputs = None
    self.input_details = None
    self.output_details = None

    # create tmp model file directory
    if self.model_path is None:
      print("Error. No model_path, Please input model path.")
      return

    if self.model_source_hub == "tfhub":
      # compile and run tfhub tflite
      if self.model_type == "tflite":
        print("Setting up for TMP_DIR")
        exe_basename = os.path.basename(sys.argv[0])
        self.workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
        print(f"TMP_DIR = {self.workdir}")
        os.makedirs(self.workdir, exist_ok=True)
        self.tflite_file = '/'.join([self.workdir, 'model.tflite'])
        print("Setting up local address for tflite model file: ", self.tflite_file)
        if os.path.exists(self.model_path):
          self.tflite_file = self.model_path
        else:
          print("Download tflite model")
          urllib.request.urlretrieve(self.model_path, self.tflite_file)
        print("Setting up tflite interpreter")
        self.tflite_interpreter = tf.lite.Interpreter(model_path=self.tflite_file)
        self.tflite_interpreter.allocate_tensors()

  def get_model_details(self):
    if self.model_type == "tflite":
      print("Get tflite input output details")
      self.input_details = self.tflite_interpreter.get_input_details()
      self.output_details = self.tflite_interpreter.get_output_details()
      return self.input_details, self.output_details

  def setup_inputs(self, inputs):
    print("Setting up inputs")
    self.inputs = inputs

  def compile_and_execute(self):
    # preprocess model_path to get model_type and Model Source Hub
    print("Shark Importer Compile and Execute Model")
    if self.model_source_hub == "tfhub":
      # compile and run tfhub tflite
      print("Inference tfhub model")
      shark_module = SharkInference(self.tflite_file, self.inputs,
                                    device=self.device,
                                    dynamic=self.dynamic,
                                    jit_trace=self.jit_trace)
      shark_module.set_frontend("tflite")
      shark_module.compile()
      iree_results = shark_module.forward(self.inputs)

      # Fix type information for unsigned cases.
      # for test compare result
      iree_results = list(iree_results)
      for i in range(len(self.output_details)):
        dtype = self.output_details[i]["dtype"]
        iree_results[i] = iree_results[i].astype(dtype)
      return iree_results
    elif self.model_source_hub == "huggingface":
      print("Inference huggingface model")
    elif self.model_source_hub == "jaxhub":
      print("Inference JAX hub model")

