import numpy as np
from shark.shark_downloader import SharkDownloader
from shark.shark_inference import SharkInference
import pytest
import unittest
from shark.parser import shark_args
import os
import sys
from tank.tflite import imagenet_data
from shark.tflite_utils import TFLitePreprocessor


# Source https://tfhub.dev/tensorflow/lite-model/inception_v4_quant/1/default/1
# model_path = "https://storage.googleapis.com/iree-model-artifacts/inception_v4_299_uint8.tflite"


def generate_inputs(input_details):
    exe_basename = os.path.basename(sys.argv[0])
    workdir = os.path.join(os.path.dirname(__file__), "../tmp", exe_basename)
    os.makedirs(workdir, exist_ok=True)
    return [imagenet_data.generate_input(workdir, input_details)]


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
        mlir_result = np.expand_dims(mlir_result, axis=0)
        print("mlir_result.shape", mlir_result.shape)
        print("tflite_result.shape", tflite_result.shape)
        assert mlir_result.shape == tflite_result.shape, "shape doesnot match"
        max_error = np.max(np.abs(mlir_result - tflite_result))
        print("Max error (%d): %f", i, max_error)


class Inception_v4_299_uint8TfliteModuleTester:
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

        # Preprocess to get SharkImporter input args
        tflite_preprocessor = TFLitePreprocessor(
            model_name="inception_v4_299_uint8"
        )
        tflite_preprocessor = TFLitePreprocessor(
            model_name="inception_v4_299_uint8"
        )
        # inputs = tflite_preprocessor.get_inputs()

        shark_downloader = SharkDownloader(
            model_name="inception_v4_299_uint8",
            tank_url="https://storage.googleapis.com/shark_tank",
            local_tank_dir="./../gen_shark_tank",
            model_type="tflite",
            input_json="input.json",
            input_type="uint8",
        )
        mlir_model = shark_downloader.get_mlir_file()
        inputs = shark_downloader.get_inputs()
        func_name = "main"

        # Use SharkInference to get inference result
        shark_module = SharkInference(
            mlir_module=mlir_model,
            function_name=func_name,
            device=self.device,
            mlir_dialect="tflite",
        )

        # Case1: Use shark_importer default generate inputs
        shark_module.compile()
        mlir_results = shark_module.forward(inputs)
        ## post process results for compare
        input_details, output_details = tflite_preprocessor.get_model_details()
        mlir_results = list(mlir_results)
        for i in range(len(output_details)):
            dtype = output_details[i]["dtype"]
            mlir_results[i] = mlir_results[i].astype(dtype)
        tflite_results = tflite_preprocessor.get_raw_model_output()
        compare_results(mlir_results, tflite_results, output_details)

        # Case2: Use manually set inputs
        input_details, output_details = tflite_preprocessor.get_model_details()
        inputs = generate_inputs(input_details)  # new inputs

        shark_module = SharkInference(
            mlir_module=mlir_model,
            function_name=func_name,
            device=self.device,
            mlir_dialect="tflite",
        )
        shark_module.compile()
        mlir_results = shark_module.forward(inputs)
        ## post process results for compare
        tflite_results = tflite_preprocessor.get_raw_model_output()
        compare_results(mlir_results, tflite_results, output_details)
        # print(mlir_results)


class Inception_v4_299_uint8TfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = Inception_v4_299_uint8TfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    import sys

    @pytest.mark.skip(
        reason="known macos tflite install issue & "
        "'tosa.conv2d' op attribute 'quantization_info' failed "
    )
    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    # module_tester = Inception_v4_299_uint8TfliteModuleTester()
    # module_tester.save_mlir = True
    # module_tester.save_vmfb = True
    # module_tester.create_and_check_module()

    unittest.main()
