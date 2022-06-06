# Lint as: python3
"""Test architecture for a set of tflite tests."""

import absl
from absl.flags import FLAGS
import absl.testing as testing

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
}

configs = {
  'dylib' : 'dylib',
  'vulkan' : 'vulkan',
}

absl.flags.DEFINE_string('config', 'dylib', 'model path to execute')

class GenerateInputSharkImporter():
  def __init__(self, input_details, model_source_hub="tfhub"):
    self.input_details = input_details
    self.model_source_hub = model_source_hub

  def generate_inputs(self):
    args = []
    for input in self.input_details:
      absl.logging.info("\t%s, %s", str(input["shape"]), input["dtype"].__name__)
      args.append(np.zeros(shape=input["shape"], dtype=input["dtype"]))
    return args

class SharkImporter:
  def __init__(self,
               model_path,
               model_type="tflite",
               model_source_hub="tfhub"):
    self.model_path = model_path
    self.model_type = model_type
    self.model_source_hub = model_source_hub
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
        print("Setting up address for model file")
        self.tflite_file = '/'.join([self.workdir, 'model.tflite'])
        self.tflite_ir = '/'.join([self.workdir, 'tflite.mlir'])
        self.iree_ir = '/'.join([self.workdir, 'tosa.mlir'])
        if os.path.exists(self.model_path):
          self.tflite_file = self.model_path
        else:
          urllib.request.urlretrieve(self.model_path, self.tflite_file)
        self.binary = '/'.join([self.workdir, 'module.bytecode'])

        print("Setting up for IREE")
        iree_tflite_compile.compile_file(
          self.tflite_file, input_type="tosa",
          output_file=self.binary,
          save_temp_tfl_input=self.tflite_ir,
          save_temp_iree_input=self.iree_ir,
          target_backends=[targets[absl.flags.FLAGS.config]],
          import_only=False)
        self.setup_iree()
        self.setup_tflite()

  def setup_tflite(self):
    print("Setting up tflite interpreter")
    self.tflite_interpreter = tf.lite.Interpreter(model_path=self.tflite_file)
    self.tflite_interpreter.allocate_tensors()
    self.input_details = self.tflite_interpreter.get_input_details()
    self.output_details = self.tflite_interpreter.get_output_details()
    return self.input_details, self.output_details

  def setup_iree(self):
    print("Setting up iree runtime")
    with open(self.binary, 'rb') as f:
      config = iree_rt.Config(configs[absl.flags.FLAGS.config])
      self.iree_context = iree_rt.SystemContext(config=config)
      vm_module = iree_rt.VmModule.from_flatbuffer(f.read())
      self.iree_context.add_vm_module(vm_module)

  def setup_input(self, inputs):
    print("Setting up inputs")
    self.inputs = inputs

  def invoke_tflite(self, args):
    for i, input in enumerate(args):
      self.tflite_interpreter.set_tensor(self.input_details[i]['index'], input)
    start = time.perf_counter()
    self.tflite_interpreter.invoke()
    end = time.perf_counter()
    tflite_results = []
    print(f"Invocation time: {end - start:0.4f} seconds")
    for output_detail in self.output_details:
      tflite_results.append(np.array(self.tflite_interpreter.get_tensor(
        output_detail['index'])))

    for i in range(len(self.output_details)):
      dtype = self.output_details[i]["dtype"]
      tflite_results[i] = tflite_results[i].astype(dtype)
    return tflite_results

  def invoke_iree(self, args):
    invoke = self.iree_context.modules.module["main"]
    start = time.perf_counter()
    iree_results = invoke(*args)
    end = time.perf_counter()
    print(f"Invocation time: {end - start:0.4f} seconds")
    if not isinstance(iree_results, tuple):
      iree_results = (iree_results,)
    return iree_results

  def compile_and_execute(self):
    # preprocess model_path to get model_type and Model Source Hub

    if self.model_source_hub == "tfhub":
      # compile and run tfhub tflite
      print("Invoking TFLite")
      tflite_results = self.invoke_tflite(self.inputs)

      print("Invoke IREE")
      iree_results = self.invoke_iree(self.inputs)

      # Fix type information for unsigned cases.
      iree_results = list(iree_results)
      for i in range(len(self.output_details)):
        dtype = self.output_details[i]["dtype"]
        iree_results[i] = iree_results[i].astype(dtype)
      return iree_results, tflite_results
    elif self.model_source_hub == "huggingface":
      print("huggingface model")
    elif self.model_source_hub == "jaxhub":
      print("JAX hub model")

