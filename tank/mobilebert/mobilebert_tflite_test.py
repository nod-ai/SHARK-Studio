import numpy as np
from shark.shark_importer import SharkImporter
from shark.shark_inference import SharkInference
import pytest
import unittest
from shark.parser import shark_args
from tank.tflite import squad_data


# model_path = "https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite"


def generate_inputs(input_details):
    for input in input_details:
        print(str(input["shape"]), input["dtype"].__name__)

    input_0 = np.asarray(squad_data._INPUT_WORD_ID, dtype=input_details[0]["dtype"])
    input_1 = np.asarray(squad_data._INPUT_TYPE_ID, dtype=input_details[1]["dtype"])
    input_2 = np.asarray(squad_data._INPUT_MASK, dtype=input_details[2]["dtype"])
    return [
        input_0.reshape(input_details[0]["shape"]),
        input_1.reshape(input_details[1]["shape"]),
        input_2.reshape(input_details[2]["shape"]),
    ]


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


class MobilebertTfliteModuleTester:
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
        my_shark_importer = SharkImporter(model_name="mobilebert", model_type="tflite")

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


class MobilebertTfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = MobilebertTfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    # module_tester = MobilebertTfliteModuleTester()
    # module_tester.save_mlir = True
    # module_tester.save_vmfb = True
    # module_tester.create_and_check_module()

    unittest.main()
