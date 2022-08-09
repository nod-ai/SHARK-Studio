from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_tf_model
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag

import iree.compiler as ireec
import unittest
import pytest
import numpy as np


class DistilBertModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_tf_model(
            "distilbert-base-uncased"
        )

        shark_module = SharkInference(
            model, func_name, device=device, mlir_dialect="mhlo"
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)


class DistilBertModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = DistilBertModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    @pytest.mark.xfail(reason="shark_tank hash issues -- awaiting triage")
    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="shark_tank hash issues -- awaiting triage")
    @pytest.mark.skipif(
        check_device_drivers("gpu"), reason=device_driver_info("gpu")
    )
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="shark_tank hash issues -- awaiting triage")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    @pytest.mark.xfail(
        "m1-moltenvk-macos" in get_vulkan_triple_flag(),
        reason="Checking: Error invoking IREE compiler tool (no repro on M2)",
    )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
