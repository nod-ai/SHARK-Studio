from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers, device_driver_info
from tank.model_utils import get_vision_model, compare_tensors
from shark.parser import shark_args
from shark.shark_downloader import download_torch_model

import unittest
import numpy as np
import pytest


class AlexnetModuleTester:
    def __init__(
        self,
        save_mlir=False,
        save_vmfb=False,
        benchmark=False,
    ):
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model_mlir, func_name, input, act_out = download_torch_model(
            "alexnet", dynamic
        )
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb

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
            import torch
            import torchvision.models as models

            torch.manual_seed(0)
            model, input, act_out = get_vision_model(
                models.alexnet(pretrained=True)
            )
            shark_module.shark_runner.frontend_model = model
            shark_module.shark_runner.benchmark_all_csv(
                (input,),
                "alexnet",
                dynamic,
                device,
                "torch",
            )


class AlexnetModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = AlexnetModuleTester(self)
        self.module_tester.save_mlir = pytestconfig.getoption("save_mlir")
        self.module_tester.save_vmfb = pytestconfig.getoption("save_vmfb")
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    def test_module_dynamic_cpu(self):
        dynamic = True
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
        check_device_drivers("gpu"), reason=device_driver_info("gpu")
    )
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_module_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
