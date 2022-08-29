from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from tank.model_utils import compare_tensors
from shark.parser import shark_args
from shark.shark_downloader import download_torch_model
from tank.test_utils import get_valid_test_params, shark_test_name_func
from parameterized import parameterized

import unittest
import numpy as np
import pytest


class DistilBertModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model_mlir, func_name, input, act_out = download_torch_model(
            "distilbert-base-uncased", dynamic
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
                "distilbert-base-uncased",
                dynamic,
                device,
                "torch",
            )


class DistilBertModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = DistilBertModuleTester(self)
        self.module_tester.save_mlir = pytestconfig.getoption("save_mlir")
        self.module_tester.save_vmfb = pytestconfig.getoption("save_vmfb")
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device):
        if device == "cpu":
            pytest.skip(
                reason="Fails to lower in torch-mlir. See https://github.com/nod-ai/SHARK/issues/222"
            )
        elif device == "gpu":
            if dynamic == False:
                pytest.skip(
                    reason="Fails to lower in torch-mlir. See https://github.com/nod-ai/SHARK/issues/222"
                )
            elif dynamic == True:
                pytest.skip(reason="DistilBert needs to be uploaded to cloud.")
        elif device in ["vulkan", "metal"]:
            pytest.skip(reason="DistilBert needs to be uploaded to cloud.")

        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
