# RUN: %PYTHON %s
# XFAIL: *

import numpy
from shark.shark_importer import SharkImporter

model_path = "https://storage.googleapis.com/iree-model-artifacts/mobilebert-edgetpu-s-float.tflite"

def generate_inputs(input_details):
    for input in input_details:
        print(str(input["shape"]), input["dtype"].__name__)

    args = []
    args.append(
        numpy.random.randint(low=0,
                             high=256,
                             size=input_details[0]["shape"],
                             dtype=input_details[0]["dtype"]))
    args.append(
        numpy.ones(shape=input_details[1]["shape"],
                   dtype=input_details[1]["dtype"]))
    args.append(
        numpy.zeros(shape=input_details[2]["shape"],
                    dtype=input_details[2]["dtype"]))
    return args


if __name__ == '__main__':
    my_shark_importer = SharkImporter(model_path=model_path,
                                      model_type="tflite",
                                      model_source_hub="tfhub",
                                      device="cpu",
                                      dynamic=False,
                                      jit_trace=True)
    # Case1: Use default inputs
    my_shark_importer.compile()
    shark_results = my_shark_importer.forward()
    # Case2: Use manually set inputs
    input_details, output_details = my_shark_importer.get_model_details()
    inputs = generate_inputs(input_details)  # device_inputs
    my_shark_importer.compile(inputs)
    shark_results = my_shark_importer.forward(inputs)
    # print(shark_results)