import tensorflow as tf
import numpy as np
import os
import csv
import urllib.request
import json


class TFLiteModelUtil:
    def __init__(self, raw_model_file):
        self.raw_model_file = str(raw_model_file)
        self.tflite_interpreter = None
        self.input_details = None
        self.output_details = None
        self.inputs = []

    def setup_tflite_interpreter(self):
        self.tflite_interpreter = tf.lite.Interpreter(
            model_path=self.raw_model_file
        )
        self.tflite_interpreter.allocate_tensors()
        # default input initialization
        return self.get_model_details()

    def get_model_details(self):
        print("Get tflite input output details")
        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()
        return self.input_details, self.output_details

    def invoke_tflite(self, inputs):
        self.inputs = inputs
        print("invoke_tflite")
        for i, input in enumerate(self.inputs):
            self.tflite_interpreter.set_tensor(
                self.input_details[i]["index"], input
            )
        self.tflite_interpreter.invoke()

        # post process tflite_result for compare with mlir_result,
        # for tflite the output is a list of numpy.tensor
        tflite_results = []
        for output_detail in self.output_details:
            tflite_results.append(
                np.array(
                    self.tflite_interpreter.get_tensor(output_detail["index"])
                )
            )

        for i in range(len(self.output_details)):
            out_dtype = self.output_details[i]["dtype"]
            tflite_results[i] = tflite_results[i].astype(out_dtype)
        return tflite_results


class TFLitePreprocessor:
    def __init__(
        self,
        model_name,
        input_details=None,
        output_details=None,
        model_path=None,
    ):
        self.model_name = model_name
        self.input_details = (
            input_details  # used for tflite, optional for tf/pytorch
        )
        self.output_details = (
            output_details  # used for tflite, optional for tf/pytorch
        )
        self.inputs = []
        self.model_path = model_path  # url to download the model
        self.raw_model_file = (
            None  # local address for raw tf/tflite/pytorch model
        )
        self.mlir_file = (
            None  # local address for .mlir file of tf/tflite/pytorch model
        )
        self.mlir_model = None  # read of .mlir file
        self.output_tensor = (
            None  # the raw tf/pytorch/tflite_output_tensor, not mlir_tensor
        )
        self.interpreter = (
            None  # could be tflite/tf/torch_interpreter in utils
        )
        self.input_file = None

        # create tmp model file directory
        if self.model_path is None and self.model_name is None:
            print(
                "Error. No model_path, No model name,Please input either one."
            )
            return

        print("Setting up for TMP_WORK_DIR")
        self.workdir = os.path.join(
            os.path.dirname(__file__), "./../gen_shark_tank"
        )
        os.makedirs(self.workdir, exist_ok=True)
        print(f"TMP_WORK_DIR = {self.workdir}")

        # compile and run tfhub tflite
        load_model_success = self.load_tflite_model()
        if not load_model_success:
            print("Error, load tflite model fail")
            return

        if (self.input_details is None) or (self.output_details is None):
            # print("Setting up tflite interpreter to get model input details")
            self.setup_interpreter()

            inputs = self.generate_inputs(self.input_details)  # device_inputs
        self.setup_inputs(inputs)

    def load_tflite_model(self):
        # use model name get dir.
        tflite_model_name_dir = os.path.join(
            self.workdir, str(self.model_name)
        )

        os.makedirs(tflite_model_name_dir, exist_ok=True)
        print(f"TMP_TFLITE_MODELNAME_DIR = {tflite_model_name_dir}")

        self.raw_model_file = "/".join(
            [tflite_model_name_dir, str(self.model_name) + "_tflite.tflite"]
        )
        self.mlir_file = "/".join(
            [tflite_model_name_dir, str(self.model_name) + "_tflite.mlir"]
        )
        self.input_file = "/".join([tflite_model_name_dir, "input.json"])

        if os.path.exists(self.raw_model_file):
            print(
                "Local address for .tflite model file Exists: ",
                self.raw_model_file,
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
                    if str(row[0]) == str(self.model_name):
                        self.model_path = row[1]
                        print("tflite_model_name", str(row[0]))
                        print("tflite_model_link", self.model_path)
            if self.model_path is None:
                print("Error, No model path find in tflite_model_list.csv")
                return False
            urllib.request.urlretrieve(self.model_path, self.raw_model_file)
        return True

    def setup_interpreter(self):
        self.interpreter = TFLiteModelUtil(self.raw_model_file)
        (
            self.input_details,
            self.output_details,
        ) = self.interpreter.setup_tflite_interpreter()

    def generate_inputs(self, input_details):
        self.inputs = []
        for tmp_input in input_details:
            # print(str(tmp_input["shape"]), tmp_input["dtype"].__name__)
            self.inputs.append(
                np.ones(shape=tmp_input["shape"], dtype=tmp_input["dtype"])
            )
        # save inputs into json file
        tmp_json = []
        for tmp_input in input_details:
            # print(str(tmp_input["shape"]), tmp_input["dtype"].__name__)
            tmp_json.append(
                np.ones(
                    shape=tmp_input["shape"], dtype=tmp_input["dtype"]
                ).tolist()
            )
        with open(self.input_file, "w") as f:
            json.dump(tmp_json, f)
        return self.inputs

    def setup_inputs(self, inputs):
        # print("Setting up inputs")
        self.inputs = inputs

    def get_mlir_model(self):
        return self.mlir_model

    def get_mlir_file(self):
        return self.mlir_file

    def get_inputs(self):
        return self.inputs

    def get_raw_model_output(self):
        self.output_tensor = self.interpreter.invoke_tflite(self.inputs)
        return self.output_tensor

    def get_model_details(self):
        return self.input_details, self.output_details

    def get_raw_model_file(self):
        return self.raw_model_file

    def get_interpreter(self):
        return self.interpreter
