from shark.shark_inference import SharkInference
from shark.iree_utils import check_device_drivers

import torch
import unittest
import numpy as np
import torchvision.models as models
from transformers import AutoModelForSequenceClassification
import pytest

torch.manual_seed(0)

##################### Hugging Face LM Models ###################################


class HuggingFaceLanguage(torch.nn.Module):

    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_hf_model(name):
    model = HuggingFaceLanguage(name)
    # TODO: Currently the test input is set to (1,128)
    test_input = torch.randint(2, (1, 128))
    actual_out = model(test_input)
    return model, test_input, actual_out

###############################################################################

# Utility function for comparing two tensors.
def compare_tensors(torch_tensor, numpy_tensor):
    # setting the absolute and relative tolerance
    rtol = 1e-02
    atol = 1e-03
    torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_to_numpy, numpy_tensor, rtol, atol)


###############################################################################

class MiniLMModuleTester:

    def create_and_check_minilm_module(self, dynamic, device):
        model, input, act_out = get_hf_model("microsoft/MiniLM-L12-H384-uncased")
        shark_module = SharkInference(model, (input,),
                                      device=device,
                                      dynamic=dynamic,
                                      jit_trace=True)
        shark_module.compile()
        results = shark_module.forward((input,))
        assert True == compare_tensors(act_out, results)

class MiniLMModuleTest(unittest.TestCase):

    def setUp(self):
        self.module_tester = MiniLMModuleTester()

    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_minilm_module(dynamic, device)
    
    @pytest.mark.xfail("language models failing for dynamic case")
    def test_module_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_minilm_module(dynamic, device)
    
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_minilm_module(dynamic, device)

    @pytest.mark.xfail("language models failing for dynamic case")
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_minilm_module(dynamic, device)

    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_minilm_module(dynamic, device)

    @pytest.mark.xfail("language models failing for dynamic case")
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_minilm_module(dynamic, device)
