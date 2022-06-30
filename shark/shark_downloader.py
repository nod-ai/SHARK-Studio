# Lint as: python3
"""SHARK Downloader"""
# Requirements : Put shark_tank in SHARK directory
#   /SHARK
#     /gen_shark_tank
#       /tflite
#         /albert_lite_base
#         /...model_name...
#       /tf
#       /pytorch
#
#
#

import numpy as np
import os
import urllib.request
import json

input_type_to_np_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
    "int32": np.int32,
    "int64": np.int64,
}


class SharkDownloader:
    def __init__(
        self,
        model_name: str,
        tank_url: str = "https://storage.googleapis.com/shark_tank",
        local_tank_dir: str = "./../gen_shark_tank/tflite",
        model_type: str = "tflite-tosa",
        input_json: str = "input.json",
        input_type: str = "int32",
    ):
        self.model_name = model_name
        self.local_tank_dir = local_tank_dir
        self.tank_url = tank_url
        self.model_type = model_type
        self.input_json = input_json  # optional if you don't have input
        self.input_type = input_type_to_np_dtype[input_type]  # optional if you don't have input
        self.mlir_file = None  # .mlir file local address.
        self.mlir_url = None
        self.inputs = None  # Input has to be (list of np.array) for sharkInference.forward use
        self.mlir_model = []

        # create tmp model file directory
        if self.tank_url is None and self.model_name is None:
            print("Error. No tank_url, No model name,Please input either one.")
            return

        self.workdir = os.path.join(
            os.path.dirname(__file__), self.local_tank_dir
        )
        os.makedirs(self.workdir, exist_ok=True)
        print(f"TMP_MODEL_DIR = {self.workdir}")
        # use model name get dir.
        self.model_name_dir = os.path.join(self.workdir, str(self.model_name))
        if not os.path.exists(self.model_name_dir):
            print(
                "Model has not been download."
                "shark_downloader will automatically download by "
                "tank_url if provided. You can also manually to "
                "download the model from shark_tank by yourself."
            )
        os.makedirs(self.model_name_dir, exist_ok=True)
        print(f"TMP_MODELNAME_DIR = {self.model_name_dir}")

        # read inputs from json file
        self.load_json_input()
        # get milr model file
        self.load_mlir_model()

    def get_mlir_file(self):
        return self.mlir_model

    def get_inputs(self):
        return self.inputs

    def load_json_input(self):
        print("load json inputs")
        if self.model_type in ["tflite-tosa"]:
            input_url = (
                self.tank_url + "/" + str(self.model_name) + "/" + "input.json"
            )
            input_file = "/".join([self.model_name_dir, str(self.input_json)])
            if os.path.exists(input_file):
                print("Input has been downloaded before.", input_file)
            else:
                print("Download input", input_url)
                urllib.request.urlretrieve(input_url, input_file)

            args = []
            with open(input_file, "r") as f:
                args = json.load(f)
            self.inputs = [np.asarray(arg, dtype=self.input_type) for arg in args]
        else:
            print(
                "No json input required for current model type. "
                "You could call setup_inputs(YOU_INPUTS)."
            )
        return self.inputs

    def load_mlir_model(self):
        if self.model_type in ["tflite-tosa"]:
            self.mlir_url = (
                self.tank_url
                + "/"
                + str(self.model_name)
                + "/"
                + str(self.model_name)
                + "_tflite.mlir"
            )
            self.mlir_file = "/".join(
                [self.model_name_dir, str(self.model_name) + "_tfite.mlir"]
            )
        elif self.model_type in ["tensorflow"]:
            self.mlir_url = (
                self.tank_url
                + "/"
                + str(self.model_name)
                + "/"
                + str(self.model_name)
                + "_tf.mlir"
            )
            self.mlir_file = "/".join(
                [self.model_name_dir, str(self.model_name) + "_tf.mlir"]
            )
        elif self.model_type in ["torch", "jax", "mhlo", "tosa"]:
            self.mlir_url = (
                self.tank_url + "/" + str(self.model_name) + "/" + str(self.model_name) + "_" + str(self.model_type) + ".mlir"
            )
            self.mlir_file = "/".join(
                [
                    self.model_name_dir,
                    str(self.model_name) + "_" + str(self.model_type) + ".mlir",
                ]
            )
        else:
            print("Unsupported mlir model")

        if os.path.exists(self.mlir_file):
            print("Model has been downloaded before.", self.mlir_file)
        else:
            print("Download mlir model", self.mlir_url)
            urllib.request.urlretrieve(self.mlir_url, self.mlir_file)

        print("Get .mlir model return")
        with open(self.mlir_file) as f:
            self.mlir_model = f.read()
        return self.mlir_model

    def setup_inputs(self, inputs):
        print("Setting up inputs. Input has to be (list of np.array)")
        self.inputs = inputs
