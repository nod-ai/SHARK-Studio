from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from tank.model_utils import compare_tensors
from shark.shark_downloader import download_torch_model
from shark.parser import shark_args
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import unittest
import numpy as np
import pytest


class BloomModuleTester:
    def __init__(
        self,
        benchmark=False,
        onnx_bench=False,
    ):
        self.benchmark = benchmark
        self.onnx_bench = onnx_bench

    def create_and_check_module(self, dynamic, device):
        model_mlir, func_name, input, act_out = download_torch_model(
            "bigscience/bloom-560m", dynamic
        )
        shark_module = SharkInference(
            model_mlir,
            func_name,
            device=device,
            mlir_dialect="tm_tensor",
            is_benchmark=self.benchmark,
        )
        if self.benchmark == True:
            shark_args.enable_tf32 = True
            shark_module.compile()
            shark_args.onnx_bench = self.onnx_bench
            shark_module.shark_runner.benchmark_all_csv(
                (input),
                "bigscience/bloom-560m",
                dynamic,
                device,
                "torch",
            )
            shark_args.enable_tf32 = False
            rtol = 1e-01
            atol = 1e-02
        else:
            shark_module.compile()
            rtol = 1e-02
            atol = 1e-03

        results = shark_module.forward(input)
        assert True == compare_tensors(act_out, results, rtol, atol)


class BloomModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = BloomModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")
        self.module_tester.onnx_bench = pytestconfig.getoption("onnx_bench")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    dynamic = False
    device = "cpu"
    module_tester = BloomModuleTester()
    module_tester.create_and_check_module(dynamic, device)
    # unittest.main()
