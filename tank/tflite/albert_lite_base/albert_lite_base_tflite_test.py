import numpy as np
from shark.shark_downloader import download_tflite_model
from shark.shark_inference import SharkInference
import pytest
import unittest
from shark.parser import shark_args


# model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"
# model_path = model_path


# Inputs modified to be useful albert inputs.
def generate_inputs(input_details):
    for input in input_details:
        print(str(input["shape"]), input["dtype"].__name__)
    # [  1 384] int32
    # [  1 384] int32
    # [  1 384] int32

    args = []
    args.append(
        np.random.randint(
            low=0,
            high=256,
            size=input_details[0]["shape"],
            dtype=input_details[0]["dtype"],
        )
    )
    args.append(
        np.ones(
            shape=input_details[1]["shape"], dtype=input_details[1]["dtype"]
        )
    )
    args.append(
        np.zeros(
            shape=input_details[2]["shape"], dtype=input_details[2]["dtype"]
        )
    )
    return args


def compare_results(mlir_results, tflite_results):
    print("Compare mlir_results VS tflite_results: ")
    assert len(mlir_results) == len(
        tflite_results
    ), "Number of results do not match"
    rtol = 1e-02
    atol = 1e-03
    print(
        "numpy.allclose: ",
        np.allclose(mlir_results, tflite_results, rtol, atol),
    )
    for i in range(len(mlir_results)):
        mlir_result = mlir_results[i]
        tflite_result = tflite_results[i]
        mlir_result = mlir_result.astype(np.single)
        tflite_result = tflite_result.astype(np.single)
        assert mlir_result.shape == tflite_result.shape, "shape doesnot match"
        max_error = np.max(np.abs(mlir_result - tflite_result))
        print("Max error (%d): %f", i, max_error)


class AlbertTfliteModuleTester:
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

        (
            mlir_model,
            function_name,
            inputs,
            tflite_results,
        ) = download_tflite_model(model_name="albert_lite_base")

        shark_module = SharkInference(
            mlir_module=mlir_model,
            function_name="main",
            device=self.device,
            mlir_dialect="tflite",
        )
        shark_module.compile()
        mlir_results = shark_module.forward(inputs)
        # print(shark_results)
        compare_results(mlir_results, tflite_results)


class AlbertTfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = AlbertTfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    unittest.main()
    # module_tester = AlbertTfliteModuleTester()
    # module_tester.create_and_check_module()
