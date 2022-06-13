from shark.shark_inference import SharkInference
from shark.iree_utils import check_device_drivers
from tank.pytorch.tests.test_utils import get_hf_model, compare_tensors

import torch
import unittest
import numpy as np
import pytest

#torch.manual_seed(0)

class BertModuleTester:

    def __init__(
        self,
        dynamic=False,
        device="cpu",
    ):
        self.dynamic = dynamic
        self.device = device

    def create_and_check_module(self):
        model, input, act_out = get_hf_model("bert-base-uncased")
        shark_module = SharkInference(model, (input,),
                                      device=self.device,
                                      dynamic=self.dynamic,
                                      jit_trace=True)
        shark_module.compile()
        results = shark_module.forward((input,))
        assert True == compare_tensors(act_out, results)

class BertModuleTest(unittest.TestCase):

    def setUp(self):
        self.module_tester = BertModuleTester(self)
        
    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()
    
    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    def test_module_dynamic_cpu(self):
        self.module_tester.dynamic = True
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()
    
    @pytest.mark.xfail(reason="BERT model on GPU currently fails to produce torch numbers")
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_static_gpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "gpu"
        self.module_tester.create_and_check_module()

    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_dynamic_gpu(self):
        self.module_tester.dynamic = True
        self.module_tester.device = "gpu"
        self.module_tester.create_and_check_module()

    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_static_vulkan(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "vulkan"
        self.module_tester.create_and_check_module()

    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_dynamic_vulkan(self):
        self.module_tester.dynamic = True
        self.module_tester.device = "vulkan"
        self.module_tester.create_and_check_module()


if __name__ == '__main__':
    unittest.main()
