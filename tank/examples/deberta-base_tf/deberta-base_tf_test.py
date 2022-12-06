from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_model
from shark.parser import shark_args
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import iree.compiler as ireec
import unittest
import pytest
import numpy as np
import tempfile
import os


class DebertaBaseModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_model(
            "microsoft/deberta-base", frontend="tf"
        )

        shark_module = SharkInference(
            model, func_name, device=device, mlir_dialect="mhlo"
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)


class DebertaBaseModuleTest(unittest.TestCase):
    @pytest.skip(reason="Model can't be imported.", allow_module_level=True)
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = DebertaBaseModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
