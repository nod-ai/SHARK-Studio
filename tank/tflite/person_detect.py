# RUN: %PYTHON %s

from shark.shark_importer import SharkImporter
import os
import sys
import numpy
import urllib.request

from PIL import Image

model_path = "https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite"

def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    img_path = "https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp"
    local_path = "/".join([workdir, "person.bmp"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize(
        (shape[1], shape[2]))).astype(input_details[0]["dtype"])
    args = [im.reshape(shape)]
    return args


if __name__ == '__main__':
    # Case2: Use manually set inputs
    input_details = [{
        "shape": [1, 96, 96, 1],
        "dtype": numpy.int8,
        "index": 0,
    }]
    output_details = [{
        "shape": [1, 2],
        "dtype": numpy.int8,
    }]
    my_shark_importer = SharkImporter(model_path=model_path,
                                      model_type="tflite",
                                      model_source_hub="tfhub",
                                      device="cpu",
                                      dynamic=False,
                                      jit_trace=True,
                                      input_details=input_details,
                                      output_details=output_details)
    inputs = generate_inputs(input_details)  # device_inputs
    my_shark_importer.compile(inputs)
    shark_results = my_shark_importer.forward(inputs)
    # print(shark_results)
