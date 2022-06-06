# RUN: %PYTHON %s
import unittest
from absl import app
import numpy as np
import sys
print(sys.path)
print(f"Name: {__name__}")
print(f"Package: {__package__}")

from shark.shark_importer import SharkImporter, GenerateInputSharkImporter

model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"

# for different model, we have to generate different inputs.
class AlbertInput(GenerateInputSharkImporter):
    def __init__(self, input_details, model_source_hub):
        super(AlbertInput, self).__init__(input_details, model_source_hub)

    # Inputs modified to be useful albert inputs.
    def generate_inputs(self):
        for input in self.input_details:
            print("\t%s, %s", str(input["shape"]), input["dtype"].__name__)

        args = []
        args.append(
            np.random.randint(low=0, high=256, size=self.input_details[0]["shape"], dtype=self.input_details[0]["dtype"]))
        args.append(np.ones(shape=self.input_details[1]["shape"], dtype=self.input_details[1]["dtype"]))
        args.append(np.zeros(shape=self.input_details[2]["shape"], dtype=self.input_details[2]["dtype"]))
        return args

    def compare_results(self, iree_results, tflite_results, details):
        print("len(iree_results):", len(iree_results))
        print("len(tflite_results):", len(tflite_results))
        assert (len(iree_results) == len(tflite_results), f"Number of results do not match")

        for i in range(len(details)):
            iree_result = iree_results[i]
            tflite_result = tflite_results[i]
            iree_result = iree_result.astype(np.single)
            tflite_result = tflite_result.astype(np.single)
            assert (iree_result.shape == tflite_result.shape)
            maxError = np.max(np.abs(iree_result - tflite_result))
            print("Max error (%d): %f", i, maxError)

        assert(np.isclose(iree_results[0], tflite_results[0], atol=1e-4).all() == True)
        assert(np.isclose(iree_results[1], tflite_results[1], atol=1e-4).all() == True)

def main(argv):
    my_shark_importer = SharkImporter(model_path, "tflite", "tfhub")
    input_details, output_details = my_shark_importer.setup_tflite()
    albert_inputs_obj = AlbertInput(input_details, "tfhub")
    inputs = albert_inputs_obj.generate_inputs()
    my_shark_importer.setup_input(inputs)
    iree_results, tflite_results = my_shark_importer.compile_and_execute()
    albert_inputs_obj.compare_results(iree_results, tflite_results, output_details)

if __name__ == '__main__':
    app.run(main)