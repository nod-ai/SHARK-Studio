import tensorflow as tf
import numpy as np


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
