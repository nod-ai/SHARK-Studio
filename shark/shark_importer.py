# Lint as: python3
"""SHARK Importer"""

import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy as np
import os
import sys
import tensorflow.compat.v2 as tf
import urllib.request
from shark.shark_inference import SharkInference


class SharkImporter:

    def __init__(self,
                 model_name: str=None,
                 model_path: str=None,
                 model_type: str = "tflite",
                 model_source_hub: str = "tfhub",
                 device: str = None,
                 dynamic: bool = False,
                 jit_trace: bool = False,
                 benchmark_mode: bool = False,
                 input_details=None,
                 output_details=None,
                 tank_url: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.model_source_hub = model_source_hub
        self.device = device
        self.dynamic = dynamic
        self.jit_trace = jit_trace
        self.benchmark_mode = benchmark_mode
        self.inputs = None
        self.input_details = input_details
        self.output_details = output_details

        # create tmp model file directory
        if self.model_path is None:
            print("Error. No model_path, Please input model path.")
            return

        if self.model_source_hub == "tfhub":
            # compile and run tfhub tflite
            if self.model_type == "tflite":
                print("Setting up for TMP_DIR")
                exe_basename = os.path.basename(sys.argv[0])
                self.workdir = os.path.join(os.path.dirname(__file__), "tmp",
                                            exe_basename)
                print(f"TMP_DIR = {self.workdir}")
                os.makedirs(self.workdir, exist_ok=True)
                self.tflite_file = '/'.join([self.workdir, 'model.tflite'])
                print("Setting up local address for tflite model file: ",
                      self.tflite_file)
                if os.path.exists(self.model_path):
                    self.tflite_file = self.model_path
                else:
                    print("Download tflite model")
                    urllib.request.urlretrieve(self.model_path,
                                               self.tflite_file)

                if (self.input_details == None) or \
                        (self.output_details == None):
                    print("Setting up tflite interpreter")
                    self.tflite_interpreter = tf.lite.Interpreter(
                        model_path=self.tflite_file)
                    self.tflite_interpreter.allocate_tensors()
                    # default input initialization
                    self.input_details, self.output_details = self.get_model_details(
                    )
                    inputs = self.generate_inputs(
                        self.input_details)  # device_inputs
                self.setup_inputs(inputs)

    def generate_inputs(self, input_details):
        args = []
        for input in input_details:
            print(str(input["shape"]), input["dtype"].__name__)
            args.append(np.zeros(shape=input["shape"], dtype=input["dtype"]))
        return args

    def get_model_details(self):
        if self.model_type == "tflite":
            print("Get tflite input output details")
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
            return self.input_details, self.output_details

    def setup_inputs(self, inputs):
        print("Setting up inputs")
        self.inputs = inputs

    def compile(self, inputs=None):
        if inputs is not None:
            self.setup_inputs(inputs)
        # preprocess model_path to get model_type and Model Source Hub
        print("Shark Importer Intialize SharkInference and Do Compile")
        if self.model_source_hub == "tfhub":
            # compile and run tfhub tflite
            print("Inference tfhub model")
            self.shark_module = SharkInference(self.tflite_file,
                                               self.inputs,
                                               device=self.device,
                                               dynamic=self.dynamic,
                                               jit_trace=self.jit_trace)
            self.shark_module.set_frontend("tflite")
            self.shark_module.compile()
        elif self.model_source_hub == "huggingface":
            print("Inference", self.model_source_hub, " not implemented yet")
        elif self.model_source_hub == "jaxhub":
            print("Inference", self.model_source_hub, " not implemented yet")

    def forward(self, inputs=None):
        if inputs is not None:
            self.setup_inputs(inputs)
        # preprocess model_path to get model_type and Model Source Hub
        print("Shark Importer forward Model")
        if self.model_source_hub == "tfhub":
            shark_results = self.shark_module.forward(self.inputs)
            # Fix type information for unsigned cases.
            # for test compare result
            shark_results = list(shark_results)
            for i in range(len(self.output_details)):
                dtype = self.output_details[i]["dtype"]
                shark_results[i] = shark_results[i].astype(dtype)
            return shark_results
        elif self.model_source_hub == "huggingface":
            print("Inference", self.model_source_hub, " not implemented yet")
        elif self.model_source_hub == "jaxhub":
            print("Inference", self.model_source_hub, " not implemented yet")
