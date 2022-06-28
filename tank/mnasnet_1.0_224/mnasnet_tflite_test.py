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


# model_path = "https://tfhub.dev/tensorflow/lite-model/mnasnet_1.0_224/1/metadata/1?lite-format=tflite"


def compare_results(mlir_results, tflite_results, details):
    print("Compare mlir_results VS tflite_results: ")
    assert len(mlir_results) == len(tflite_results), "Number of results do not match"
    for i in range(len(details)):
        mlir_result = mlir_results[i]
        tflite_result = tflite_results[i]
        mlir_result = mlir_result.astype(np.single)
        tflite_result = tflite_result.astype(np.single)
        assert mlir_result.shape == tflite_result.shape, "shape doesnot match"
        max_error = np.max(np.abs(mlir_result - tflite_result))
        print("Max error (%d): %f", i, max_error)


class MnasnetTfliteModuleTester:
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
        my_shark_importer = SharkImporter(model_name="mnasnet_1.0_224", model_type="tflite")

        mlir_model = my_shark_importer.get_mlir_model()
        inputs = my_shark_importer.get_inputs()
        shark_module = SharkInference(mlir_model, inputs, device=self.device, dynamic=self.dynamic)
        shark_module.set_frontend("tflite-tosa")


class MnasnetTfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = MnasnetTfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    # module_tester = MnasnetTfliteModuleTester()
    # module_tester.save_mlir = True
    # module_tester.save_vmfb = True
    # module_tester.create_and_check_module()

    unittest.main()
