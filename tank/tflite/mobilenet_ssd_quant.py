# RUN: %PYTHON %s

from shark.shark_importer import SharkImporter
import os
import sys
import urllib.request
from PIL import Image
import numpy

# Model from https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess.tflite
# but trimmed the final TFLite_PostProcess op.
model_path = "https://storage.googleapis.com/iree-shared-files/models/ssd_mobilenet_v2_face_quant.tflite"

def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    img_path = "https://github.com/google-coral/test_data/raw/master/grace_hopper.bmp"
    local_path = "/".join([workdir, "grace_hopper.bmp"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
    args = [im.reshape(shape)]
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
