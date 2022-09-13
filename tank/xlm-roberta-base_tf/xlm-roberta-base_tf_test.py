from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_tf_model
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import iree.compiler as ireec
import unittest
import pytest
import numpy as np


class XLMRobertaModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_tf_model(
            "xlm-roberta-base"
        )

        shark_module = SharkInference(
            model, func_name, device=device, mlir_dialect="mhlo"
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        np.testing.assert_allclose(
            result, golden_out, rtol=1e-02, atol=1e-01, verbose=True
        )


class XLMRobertaModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = XLMRobertaModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        if device == "cuda":
            pytest.xfail(reason="https://github.com/nod-ai/SHARK/issues/274")
        elif device in ["metal", "vulkan"]:
            if dynamic == False:
                if "m1-moltenvk-macos" in get_vulkan_triple_flag():
                    pytest.xfail(reason="M1: CompilerToolError | M2: Pass")
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
