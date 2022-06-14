# RUN: %PYTHON %s

import imagenet_data
from shark.shark_importer import SharkImporter
import os
import sys

model_path = "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v3-large_224_1.0_float.tflite"

def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    inputs = imagenet_data.generate_input(workdir, input_details)
    # Normalize inputs to [-1, 1].
    inputs = (inputs.astype('float32') / 127.5) - 1
    return [inputs]


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
