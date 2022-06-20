# RUN: %PYTHON %s
# XFAIL: *

import numpy as np
import squad_data
from shark.shark_importer import SharkImporter

model_path = "https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite"

def generate_inputs(input_details):
    for input in input_details:
        print(str(input["shape"]), input["dtype"].__name__)

    input_0 = np.asarray(squad_data._INPUT_WORD_ID,
                         dtype=input_details[0]["dtype"])
    input_1 = np.asarray(squad_data._INPUT_TYPE_ID,
                         dtype=input_details[1]["dtype"])
    input_2 = np.asarray(squad_data._INPUT_MASK,
                         dtype=input_details[2]["dtype"])
    return [
        input_0.reshape(input_details[0]["shape"]),
        input_1.reshape(input_details[1]["shape"]),
        input_2.reshape(input_details[2]["shape"])
    ]


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