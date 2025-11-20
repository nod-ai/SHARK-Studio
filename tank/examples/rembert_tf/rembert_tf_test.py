from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_downloader import download_model
from tank.test_utils import get_valid_test_params, amdshark_test_name_func
from parameterized import parameterized

import iree.compiler as ireec
import unittest
import pytest
import numpy as np


class RemBertModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_model(
            "google/rembert", frontend="tf"
        )

        amdshark_module = AMDSharkInference(
            model, func_name, device=device, mlir_dialect="mhlo"
        )
        amdshark_module.compile()
        result = amdshark_module.forward(inputs)
        np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)


class RemBertModuleTest(unittest.TestCase):
    @pytest.skip(reason="Model too large to convert.", allow_module_level=True)
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = RemBertModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=amdshark_test_name_func)
    def test_module(self, dynamic, device):
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
