from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from tank.model_utils import compare_tensors
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag
from shark.shark_downloader import download_torch_model
from tank.test_utils import get_valid_test_params, shark_test_name_func

from parameterized import parameterized
import unittest
import numpy as np
import pytest


class AlexnetModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model_mlir, func_name, input, act_out = download_torch_model(
            "alexnet", dynamic
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
                "alexnet",
                dynamic,
                device,
                "torch",
            )


class AlexnetModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = AlexnetModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        if device in ["metal", "vulkan"]:
            if dynamic == False:
                if "m1-moltenvk-macos" in get_vulkan_triple_flag():
                    pytest.xfail(
                        reason="Assert Error:https://github.com/iree-org/iree/issues/10075"
                    )
            else:
                pytest.xfail(reason="Not supported arith.floordivsi")

        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
