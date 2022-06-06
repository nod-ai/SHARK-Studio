# RUN: %PYTHON %s
import numpy
import sys
print(sys.path)
print(f"Name: {__name__}")
print(f"Package: {__package__}")

from shark.shark_importer import SharkImporter, GenerateInputSharkImporter

model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"

class AlberInput(GenerateInputSharkImporter):
    def __init__(self):
        super(AlberInput, self).__init__(input_details, )

    # Inputs modified to be useful albert inputs.
    def generate_inputs(self, input_details):
        for input in input_details:
            print("\t%s, %s", str(input["shape"]), input["dtype"].__name__)

        args = []
        args.append(
            numpy.random.randint(low=0, high=256, size=input_details[0]["shape"], dtype=input_details[0]["dtype"]))
        args.append(numpy.ones(shape=input_details[1]["shape"], dtype=input_details[1]["dtype"]))
        args.append(numpy.zeros(shape=input_details[2]["shape"], dtype=input_details[2]["dtype"]))
        return args


if __name__ == '__main__':
    my_shark_importer = SharkImporter(model_path, "tflite", "tfhub")
    input_details, output_details = my_shark_importer.setup_tflite()
    inputs = AlberInput(input_details, "tfhub").generate_inputs()
    my_shark_importer.setup_input(inputs)
    iree_results, tflite_results = my_shark_importer.compile_and_execute()
    my_shark_importer.compare_results(iree_results, tflite_results, output_details)
