import numpy as np
from shark.shark_importer import SharkImporter
from shark.shark_inference import SharkInference
import pytest
import unittest
from shark.parser import shark_args
import os
import sys
from tank.tflite import coco_data


# model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_spaghettinet_edgetpu_large.tflite"


def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "../tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)

    inputs = coco_data.generate_input(workdir, input_details)
    # Normalize inputs to [-1, 1].
    inputs = (inputs.astype("float32") / 127.5) - 1
    return [inputs]


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


class SpaghettinetTfliteModuleTester:
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
        my_shark_importer = SharkImporter(model_name="ssd_spaghettinet_edgetpu_large", model_type="tflite")

        mlir_model = my_shark_importer.get_mlir_model()
        inputs = my_shark_importer.get_inputs()
        shark_module = SharkInference(mlir_model, inputs, device=self.device, dynamic=self.dynamic)
        shark_module.set_frontend("tflite-tosa")

        # Case1: Use shark_importer default generate inputs
        shark_module.compile()
        mlir_results = shark_module.forward(inputs)
        ## post process results for compare
        input_details, output_details = my_shark_importer.get_model_details()
        mlir_results = list(mlir_results)
        for i in range(len(output_details)):
            dtype = output_details[i]["dtype"]
            mlir_results[i] = mlir_results[i].astype(dtype)
        tflite_results = my_shark_importer.get_raw_model_output()
        compare_results(mlir_results, tflite_results, output_details)

        # Case2: Use manually set inputs
        input_details, output_details = my_shark_importer.get_model_details()
        inputs = generate_inputs(input_details)  # device_inputs
        shark_module = SharkInference(mlir_model, inputs, device=self.device, dynamic=self.dynamic)
        shark_module.set_frontend("tflite-tosa")
        shark_module.compile()
        mlir_results = shark_module.forward(inputs)
        tflite_results = my_shark_importer.get_raw_model_output()
        compare_results(mlir_results, tflite_results, output_details)
        # print(mlir_results)


class SpaghettinetTfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = SpaghettinetTfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    # module_tester = SpaghettinetTfliteModuleTester()
    # module_tester.save_mlir = True
    # module_tester.save_vmfb = True
    # module_tester.create_and_check_module()

    unittest.main()
