import numpy as np
from shark.shark_importer import SharkImporter
import pytest
import unittest
from shark.parser import shark_args

# model_path = "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1?lite-format=tflite"
# model_path = model_path

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
        my_shark_importer = SharkImporter(
            model_name="albert_lite_base",
            # model_path=model_path,
            model_type="tflite",
            model_source_hub="tfhub",
            device=self.device,
            dynamic=self.dynamic,
            jit_trace=True,
            tank_url=None,
        )
        # Case1: Use default inputs
        my_shark_importer.compile()
        shark_results = my_shark_importer.forward()
        # Case2: Use manually set inputs
        input_details, output_details = my_shark_importer.get_model_details()
        inputs = generate_inputs(input_details)  # device_inputs
        my_shark_importer.compile(inputs)
        shark_results = my_shark_importer.forward(inputs)
        # print(shark_results)


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
    # module_tester = AlbertTfliteModuleTester()
    # module_tester.create_and_check_module()
    unittest.main()
