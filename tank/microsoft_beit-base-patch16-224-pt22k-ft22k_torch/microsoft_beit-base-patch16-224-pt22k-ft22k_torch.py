from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_torch_model

import unittest
import pytest
import numpy as np


class BeitModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_torch_model(
            "microsoft/beit-base-patch16-224-pt22k-ft22k", dynamic
        )

        shark_module = SharkInference(
            model,
            func_name,
            device=device,
            mlir_dialect="linalg",
            is_benchmark=self.benchmark,
        )
        shark_module.compile()
        result = shark_module.forward(inputs)

        print(np.allclose(golden_out[0], result[0], rtol=1e-02, atol=1e-03))


class BeitModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = BeitModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skipif(
        check_device_drivers("gpu"), reason=device_driver_info("gpu")
    )
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    # dynamic = False
    # device = "cpu"
    # module_tester = BeitModuleTester()
    # module_tester.create_and_check_module(dynamic, device)
    unittest.main()
