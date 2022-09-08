from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_downloader import download_tf_model
from shark.parser import shark_args
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import unittest
import numpy as np
import pytest
import numpy as np


class Resnet50ModuleTester:
    def __init__(
        self,
        benchmark=False,
        onnx_bench=False,
    ):
        self.benchmark = benchmark
        self.onnx_bench = onnx_bench

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_tf_model("resnet50")

        shark_module = SharkInference(
            model,
            func_name,
            device=device,
            mlir_dialect="mhlo",
            is_benchmark=self.benchmark,
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        np.testing.assert_allclose(golden_out, result, rtol=1e-02, atol=1e-03)

        if self.benchmark == True:
            shark_args.enable_tf32 = True
            shark_args.onnx_bench = self.onnx_bench
            shark_module.shark_runner.benchmark_all_csv(
                (inputs), "resnet50", dynamic, device, "tensorflow"
            )


class Resnet50ModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = Resnet50ModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")
        self.module_tester.onnx_bench = pytestconfig.getoption("onnx_bench")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        if device in ["metal", "vulkan"]:
            if get_vulkan_triple_flag() is not None:
                if "m1-moltenvk-macos" in get_vulkan_triple_flag():
                    pytest.xfail(reason="M2: Assert error & M1: CompilerToolError")
        
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
