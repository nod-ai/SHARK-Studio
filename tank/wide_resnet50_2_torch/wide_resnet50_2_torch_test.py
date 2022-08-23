from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from tank.model_utils import get_vision_model, compare_tensors
from shark.shark_downloader import download_torch_model
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import unittest
import numpy as np
import pytest


class WideResnet50ModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model_mlir, func_name, input, act_out = download_torch_model(
            "wide_resnet50_2", dynamic
        )

        # from shark.shark_importer import SharkImporter
        # mlir_importer = SharkImporter(
        #    model,
        #    (input,),
        #    frontend="torch",
        # )
        # minilm_mlir, func_name = mlir_importer.import_mlir(
        #    is_dynamic=dynamic, tracing_required=True
        # )

        shark_module = SharkInference(
            model_mlir,
            func_name,
            device=device,
            mlir_dialect="linalg",
            is_benchmark=self.benchmark,
        )
        shark_module.compile()
        results = shark_module.forward(input)
        assert True == compare_tensors(act_out, results)

        if self.benchmark == True:
            shark_module.shark_runner.benchmark_all_csv(
                (input),
                "wide_resnet50_2",
                dynamic,
                device,
                "torch",
            )


class WideResnet50ModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = WideResnet50ModuleTester(self)
        self.module_tester.save_mlir = pytestconfig.getoption("save_mlir")
        self.module_tester.save_vmfb = pytestconfig.getoption("save_vmfb")
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
