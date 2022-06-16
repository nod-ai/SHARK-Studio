from masked_lm import get_causal_lm_model
from tank.model_utils_tf import compare_tensors_tf
from shark.iree_utils import check_device_drivers
from shark.shark_inference import SharkInference

import unittest
import pytest


class RemBertModuleTester:

    def create_and_check_module(self, dynamic, device):
        model, input, act_out = get_causal_lm_model("google/rembert")
        shark_module = SharkInference(model, (input,),
                                      device=device,
                                      dynamic=dynamic,
                                      jit_trace=True)
        shark_module.set_frontend("tensorflow")
        shark_module.compile()
        results = shark_module.forward((input))
        assert True == compare_tensors_tf(act_out, results)


class RemBertModuleTest(unittest.TestCase):

    def setUp(self):
        self.module_tester = RemBertModuleTester()

    @pytest.mark.skip(reason="rembert currently failing in the lowering passes."
                     )
    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(
        reason="Language models currently failing for dynamic case")
    def test_module_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="rembert currently failing in the lowering passes."
                     )
    @pytest.mark.skipif(check_device_drivers("gpu"),
                        reason="nvidia-smi not found")
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="rembert currently failing in the lowering passes."
                     )
    @pytest.mark.xfail(
        reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(check_device_drivers("gpu"),
                        reason="nvidia-smi not found")
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="rembert currently failing in the lowering passes."
                     )
    @pytest.mark.skipif(
        check_device_drivers("vulkan"),
        reason=
        "vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
    )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="rembert currently failing in the lowering passes."
                     )
    @pytest.mark.xfail(
        reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"),
        reason=
        "vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
    )
    def test_module_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == '__main__':
    unittest.main()
