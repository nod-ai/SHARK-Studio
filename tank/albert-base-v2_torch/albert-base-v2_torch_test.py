from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info, IREE_DEVICE_MAP
from tank.model_utils import compare_tensors
from shark.shark_downloader import download_torch_model

import unittest
import numpy as np
import pytest
from parameterized import parameterized


class AlbertModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model_mlir, func_name, input, act_out = download_torch_model(
            "albert-base-v2", dynamic
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
                "albert-base-v2",
                dynamic,
                device,
                "torch",
            )

    
class AlbertModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = AlbertModuleTester(self)
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    device_list = tuple(IREE_DEVICE_MAP.keys())

    @parameterized.expand(device_list)
    def test_module_static(self, device):
        dynamic = False
        if(check_device_drivers(device)):
            pytest.skip(device_driver_info(device))
        self.module_tester.create_and_check_module(dynamic, device)

    @parameterized.expand(device_list)
    def test_module_dynamic(self, device):
        dynamic = True
        if(check_device_drivers(device)):
            pytest.skip(device_driver_info(device))
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
