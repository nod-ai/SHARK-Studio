from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_tf_model
from tank.test_utils import get_valid_test_params, shark_test_name_func

import iree.compiler as ireec
import unittest
import pytest
import numpy as np
from parameterized import parameterized


class AlbertBaseModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_tf_model(
            "albert-base-v2"
        )

        shark_module = SharkInference(
            model, func_name, device=device, mlir_dialect="mhlo"
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)


class AlbertBaseModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = AlbertBaseModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()
    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        if device == "gpu":
            if (check_device_drivers("gpu")):
                pytest.skip(reason=device_driver_info("gpu"))
        elif device == "vulkan":
            if (check_device_drivers("vulkan")):
                pytest.skip(reason=device_driver_info("vulkan"))
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
