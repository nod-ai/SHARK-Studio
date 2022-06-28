import numpy as np
from shark.shark_importer import SharkImporter
from shark.shark_inference import SharkInference
import pytest
import unittest
from shark.parser import shark_args
import os
import sys
import urllib.request
from PIL import Image


# model_path = "https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/models/person_detect.tflite"


def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "../tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    img_path = "https://github.com/tensorflow/tflite-micro/raw/aeac6f39e5c7475cea20c54e86d41e3a38312546/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp"
    local_path = "/".join([workdir, "person.bmp"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = np.array(Image.open(local_path).resize((shape[1], shape[2]))).astype(
        input_details[0]["dtype"]
    )
    args = [im.reshape(shape)]
    return args


def compare_results(mlir_results, tflite_results, details):
    print("Compare mlir_results VS tflite_results: ")
    assert len(mlir_results) == len(
        tflite_results
    ), "Number of results do not match"
    for i in range(len(details)):
        mlir_result = mlir_results[i]
        tflite_result = tflite_results[i]
        mlir_result = mlir_result.astype(np.single)
        tflite_result = tflite_result.astype(np.single)
        assert mlir_result.shape == tflite_result.shape, "shape doesnot match"
        max_error = np.max(np.abs(mlir_result - tflite_result))
        print("Max error (%d): %f", i, max_error)


class PersonDetectionTfliteModuleTester:
    def __init__(
        self,
        dynamic=False,
        device="cpu",
        save_mlir=False,
        save_vmfb=False,
    ):
        self.dynamic = dynamic
        self.device = device
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self):
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        my_shark_importer = SharkImporter(
            model_name="person_detect", model_type="tflite"
        )

        mlir_model = my_shark_importer.get_mlir_model()
        inputs = my_shark_importer.get_inputs()
        shark_module = SharkInference(
            mlir_model, inputs, device=self.device, dynamic=self.dynamic
        )
        shark_module.set_frontend("tflite-tosa")

        # Case2: Use manually set inputs
        input_details = [
            {
                "shape": [1, 96, 96, 1],
                "dtype": np.int8,
                "index": 0,
            }
        ]
        output_details = [
            {
                "shape": [1, 2],
                "dtype": np.int8,
            }
        ]
        inputs = generate_inputs(input_details)  # device_inputs
        shark_module = SharkInference(
            mlir_model, inputs, device=self.device, dynamic=self.dynamic
        )
        shark_module.set_frontend("tflite-tosa")
        shark_module.compile()
        mlir_results = shark_module.forward(inputs)
        tflite_results = my_shark_importer.get_raw_model_output()
        compare_results(mlir_results, tflite_results, output_details)
        # print(mlir_results)


class PersonDetectionTfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = PersonDetectionTfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    # module_tester = PersonDetectionTfliteModuleTester()
    # module_tester.save_mlir = True
    # module_tester.save_vmfb = True
    # module_tester.create_and_check_module()

    unittest.main()
