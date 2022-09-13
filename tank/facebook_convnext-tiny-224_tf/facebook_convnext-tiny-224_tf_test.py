from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_tf_model
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import unittest
import pytest
import numpy as np


class ConvNextTinyModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, func_name, inputs, golden_out = download_tf_model(
            "facebook/convnext-tiny-224"
        )

        shark_module = SharkInference(
            model, func_name, device=device, mlir_dialect="mhlo"
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        #  result: array([['logits',
        #         <IREE DeviceArray: shape=[1, 1000], dtype=<class 'numpy.float32'>>]],
        #       dtype=object)

        # post process of img output
        ir_device_array = result[0][1]
        logits = ir_device_array.astype(ir_device_array.dtype)
        logits = np.squeeze(logits, axis=0)
        print("logits: ", logits.shape)
        print("golden_out: ", golden_out[0].shape)
        print(np.allclose(golden_out[0], logits, rtol=1e-02, atol=1e-03))


class ConvNextTinyModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = ConvNextTinyModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):

        if device in ["cuda"]:
            pytest.xfail(reason="https://github.com/nod-ai/SHARK/issues/311")

        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    # dynamic = False
    # device = "cpu"
    # module_tester = ConvNextTinyModuleTester()
    # module_tester.create_and_check_module(dynamic, device)
    unittest.main()
