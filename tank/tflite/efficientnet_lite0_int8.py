# RUN: %PYTHON %s
from shark.shark_importer import SharkImporter
import imagenet_data
import os
import sys

# Source https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2
model_path = "https://storage.googleapis.com/iree-model-artifacts/efficientnet_lite0_int8_2.tflite"

def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    return [imagenet_data.generate_input(workdir, input_details)]

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