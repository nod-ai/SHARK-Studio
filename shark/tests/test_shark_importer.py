# RUN: %PYTHON %s
import numpy as np
from shark.shark_importer import SharkImporter
import pytest
from shark.parser import shark_args
from shark.shark_inference import SharkInference
from shark.tflite_utils import TFLitePreprocessor

# model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"


# Inputs modified to be useful albert inputs.
def generate_inputs(input_details):
    for input in input_details:
        print(str(input["shape"]), input["dtype"].__name__)

    args = []
    args.append(
        np.random.randint(
            low=0,
            high=256,
            size=input_details[0]["shape"],
            dtype=input_details[0]["dtype"],
        )
    )
    args.append(np.ones(shape=input_details[1]["shape"], dtype=input_details[1]["dtype"]))
    args.append(np.zeros(shape=input_details[2]["shape"], dtype=input_details[2]["dtype"]))
    return args


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
        tflite_preprocessor = TFLitePreprocessor(model_name="albert_lite_base")

        raw_model_file_path = tflite_preprocessor.get_raw_model_file()
        inputs = tflite_preprocessor.get_inputs()
        tflite_interpreter = tflite_preprocessor.get_interpreter()

        my_shark_importer = SharkImporter(
            module=tflite_interpreter,
            inputs=inputs,
            frontend="tflite",
            raw_model_file=raw_model_file_path,
        )
        mlir_model, func_name = my_shark_importer.import_mlir()

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


# A specific case can be run by commenting different cases. Runs all the test
# across cpu, gpu and vulkan according to available drivers.
pytest_param = pytest.mark.parametrize(
    ("dynamic", "device"),
    [
        pytest.param(False, "cpu"),
        # TODO: Language models are failing for dynamic case..
        pytest.param(True, "cpu", marks=pytest.mark.skip),
    ],
)


@pytest_param
def test_albert(dynamic, device):
    module_tester = AlbertTfliteModuleTester(dynamic=dynamic, device=device)
    module_tester.create_and_check_module()


if __name__ == "__main__":
    test_albert(False, "cpu")
