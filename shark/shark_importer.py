# Lint as: python3
"""SHARK Importer"""

import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy as np
import os
import sys
import csv
import tensorflow as tf
import urllib.request
from shark.shark_inference import SharkInference
import iree.compiler.tflite as ireec_tflite
from shark.iree_utils import IREE_TARGET_MAP


class SharkImporter:
    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        model_type: str = "tflite",
        model_source_hub: str = "tfhub",
        device: str = None,
        dynamic: bool = False,
        jit_trace: bool = False,
        benchmark_mode: bool = False,
        input_details=None,
        output_details=None,
        tank_url: str = None,
    ):
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
        self.tflite_saving_file = None
        self.tflite_tosa_file = None
        self.tank_url = tank_url

        # create tmp model file directory
        if self.model_path is None and self.model_name is None:
            print(
                "Error. No model_path, No model name,Please input either one."
            )
            return

        if self.model_source_hub == "tfhub":
            # compile and run tfhub tflite
            if self.model_type == "tflite":
                load_model_success = self.load_tflite_model()
                if load_model_success == False:
                    print("Error, load tflite model fail")
                    return

                if (self.input_details == None) or (
                    self.output_details == None
                ):
                    print(
                        "Setting up tflite interpreter to get model input details"
                    )
                    self.tflite_interpreter = tf.lite.Interpreter(
                        model_path=self.tflite_saving_file
                    )
                    self.tflite_interpreter.allocate_tensors()
                    # default input initialization
                    (
                        self.input_details,
                        self.output_details,
                    ) = self.get_model_details()
                    inputs = self.generate_inputs(
                        self.input_details
                    )  # device_inputs
                self.setup_inputs(inputs)

    def load_tflite_model(self):
        print("Setting up for TMP_DIR")
        tflite_workdir = os.path.join(
            os.path.dirname(__file__), "./../gen_shark_tank/tflite"
        )
        os.makedirs(tflite_workdir, exist_ok=True)
        print(f"TMP_TFLITE_DIR = {tflite_workdir}")
        # use model name get dir.
        tflite_model_name_dir = os.path.join(
            tflite_workdir, str(self.model_name)
        )
        # TODO Download model from google bucket to tflite_model_name_dir by tank_url
        os.makedirs(tflite_model_name_dir, exist_ok=True)
        print(f"TMP_TFLITE_MODELNAME_DIR = {tflite_model_name_dir}")

        self.tflite_saving_file = "/".join(
            [tflite_model_name_dir, str(self.model_name) + "_tflite.tflite"]
        )
        self.tflite_tosa_file = "/".join(
            [tflite_model_name_dir, str(self.model_name) + "_tosa.mlir"]
        )

        if os.path.exists(self.tflite_saving_file):
            print(
                "Local address for tflite model file Exists: ",
                self.tflite_saving_file,
            )
        else:
            print("No local tflite file, Download tflite model")
            if self.model_path is None:
                # get model file from tflite_model_list.csv or download from gs://bucket
                print("No model_path, get from tflite_model_list.csv")
                tflite_model_list_path = os.path.join(
                    os.path.dirname(__file__),
                    "../tank/tflite/tflite_model_list.csv",
                )
                tflite_model_list = csv.reader(open(tflite_model_list_path))
                for row in tflite_model_list:
                    if str(row[0]) == self.model_name:
                        self.model_path = row[1]
            if self.model_path is None:
                print("Error, No model path find in tflite_model_list.csv")
                return False
            urllib.request.urlretrieve(self.model_path, self.tflite_saving_file)
        if os.path.exists(self.tflite_tosa_file):
            print("Exists", self.tflite_tosa_file)
        else:
            print(
                "No tflite tosa.mlir, please use python generate_sharktank.py to download tosa model"
            )
        return True

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
            if os.path.exists(self.tflite_tosa_file):
                print("Use", self.tflite_tosa_file, "as TOSA compile input")
                # compile and run tfhub tflite
                print("Inference tflite tosa model")
                tosa_model = []
                with open(self.tflite_tosa_file) as f:
                    tosa_model = f.read()
                self.shark_module = SharkInference(
                    tosa_model,
                    self.inputs,
                    device=self.device,
                    dynamic=self.dynamic,
                    jit_trace=self.jit_trace,
                )
                self.shark_module.set_frontend("tflite-tosa")
                self.shark_module.compile()
            else:
                # compile and run tfhub tflite
                print("Inference tfhub tflite model")
                self.shark_module = SharkInference(
                    self.tflite_saving_file,
                    self.inputs,
                    device=self.device,
                    dynamic=self.dynamic,
                    jit_trace=self.jit_trace,
                )
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
