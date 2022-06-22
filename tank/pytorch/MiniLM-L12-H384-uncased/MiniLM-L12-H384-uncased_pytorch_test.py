from shark.shark_inference import SharkInference
from shark.iree_utils._common import check_device_drivers
from tank.model_utils import get_hf_model, compare_tensors
from shark.parser import shark_args

import torch
import unittest
import numpy as np
import pytest

torch.manual_seed(0)


class MiniLMModuleTester:
    def __init__(
        self,
        dynamic=False,
        device="cpu",
        save_mlir=False,
        save_vmfb=False,
    ):
        self.dynamic = dynamic
        self.device = device
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self):
        model, input, act_out = get_hf_model(
            "microsoft/MiniLM-L12-H384-uncased"
        )
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        shark_module = SharkInference(
            model,
            (input,),
            device=self.device,
            dynamic=self.dynamic,
            jit_trace=True,
        )
        shark_module.compile()
        results = shark_module.forward((input,))
        assert True == compare_tensors(act_out, results)


class MiniLMModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = MiniLMModuleTester(self)

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()

    @pytest.mark.xfail(reason="language models failing for dynamic case")
    def test_module_dynamic_cpu(self):
        self.module_tester.dynamic = True
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()

    @pytest.mark.skipif(
        check_device_drivers("gpu"), reason="nvidia-smi not found"
    )
    def test_module_static_gpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "gpu"
        self.module_tester.create_and_check_module()

    @pytest.mark.skipif(
        check_device_drivers("gpu"), reason="nvidia-smi not found"
    )
    def test_module_dynamic_gpu(self):
        self.module_tester.dynamic = True
        self.module_tester.device = "gpu"
        self.module_tester.create_and_check_module()

    @pytest.mark.xfail(reason="https://github.com/google/iree/issues/9554")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"),
        reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases",
    )
    def test_module_static_vulkan(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "vulkan"
        self.module_tester.create_and_check_module()

    @pytest.mark.xfail(reason="https://github.com/google/iree/issues/9554")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"),
        reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases",
    )
    def test_module_dynamic_vulkan(self):
        self.module_tester.dynamic = True
        self.module_tester.device = "vulkan"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    unittest.main()
