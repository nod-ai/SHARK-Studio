from shark.shark_inference import SharkInference
from shark.iree_utils import check_device_drivers
from tank.tf.tests.test_utils_tf import get_TFhf_model, compare_tensors_tf

import tensorflow as tf
import unittest
import numpy as np
import pytest

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

#Create a set of 2-dimensional inputs
tf_bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32)
]

class MiniLMTFModuleTester:

    def create_and_check_module(self, dynamic, device):
        model, input, act_out = get_TFhf_model("microsoft/MiniLM-L12-H384-uncased")
        shark_module = SharkInference(model, (input,),
                                      device=device,
                                      dynamic=dynamic,
                                      jit_trace=True)
        shark_module.set_frontend("tensorflow")
        shark_module.compile()
        results = shark_module.forward((input))
        assert True == compare_tensors_tf(act_out, results)

class MiniLMTFModuleTest(unittest.TestCase):

    def setUp(self):
        self.module_tester = MiniLMTFModuleTester()

    @pytest.mark.skip(reason="TF testing temporarily unavailable.")
    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)
    
    @pytest.mark.skip(reason="TF testing temporarily unavailable.")
    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    def test_module_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)
    
    @pytest.mark.skip(reason="TF testing temporarily unavailable.")
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="TF testing temporarily unavailable.")
    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="TF testing temporarily unavailable.")
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="TF testing temporarily unavailable.")
    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == '__main__':
    unittest.main()
