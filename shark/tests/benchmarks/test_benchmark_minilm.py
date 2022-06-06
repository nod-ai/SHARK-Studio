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

    def create_minilm_module(self, dynamic, device):
        model, input, act_out = get_hf_model("microsoft/MiniLM-L12-H384-uncased")
        shark_module = SharkInference(model, (input,),
                                      device=device,
                                      dynamic=dynamic,
                                      jit_trace=True,
                                      benchmark_mode=True)
        shark_module.compile()
    
        return shark_module, input
    
    def bench_frontend(self, shark_module, input):
        shark_module.benchmark_frontend((input,))

    def bench_python(self, shark_module, input):
        shark_module.benchmark_python((input,))

    def bench_c(self, shark_module):
        shark_module.benchmark_c()

class MiniLMModuleTest(unittest.TestCase):

    def setUp(self):
        self.module_tester = MiniLMModuleTester()

    @pytest.fixture(autouse=True)
    def setup_benchmark(self, benchmark):
        self.benchmark = benchmark

    def test_benchmark_static_cpu_frontend(self):
        dynamic = False
        device = "cpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_frontend, shark_module, input)

    def test_benchmark_static_cpu_python(self):
        dynamic = False
        device = "cpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_python, shark_module, input)
    
    def test_benchmark_static_cpu_c(self):
        dynamic = False
        device = "cpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_c, shark_module)
    
    @pytest.mark.skip
    def test_benchmark_dynamic_cpu_frontend(self):
        dynamic = True
        device = "cpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_frontend, shark_module, input)

    @pytest.mark.skip("language models failing for dynamic case")
    def test_benchmark_dynamic_cpu_python(self):
        dynamic = True
        device = "cpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_python, shark_module, input)
    
    @pytest.mark.skip("language models failing for dynamic case")
    def test_benchmark_dynamic_cpu_c(self):
        dynamic = True
        device = "cpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_c, shark_module)
    
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_benchmark_static_gpu_frontend(self):
        dynamic = False
        device = "gpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_frontend, shark_module, input)
        
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_benchmark_static_gpu_python(self):
        dynamic = False
        device = "gpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_python, shark_module, input)

    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_benchmark_static_gpu_c(self):
        dynamic = False
        device = "gpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_c, shark_module)
    
    @pytest.mark.skip
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_benchmark_dynamic_gpu_frontend(self):
        dynamic = True
        device = "gpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_frontend, shark_module, input)

    @pytest.mark.skip
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_benchmark_dynamic_gpu_python(self):
        dynamic = True
        device = "gpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_python, shark_module, input)

    @pytest.mark.skip
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_benchmark_dynamic_gpu_c(self):
        dynamic = True
        device = "gpu"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_c, shark_module)
    
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_benchmark_static_vulkan_frontend(self):
        dynamic = False
        device = "vulkan"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_frontend, shark_module, input)

    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_benchmark_static_vulkan_python(self):
        dynamic = False
        device = "vulkan"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_python, shark_module, input)
    
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_benchmark_static_vulkan_c(self):
        dynamic = False
        device = "vulkan"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_c, shark_module)

    @pytest.mark.skip
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_benchmark_dynamic_vulkan_frontend(self):
        dynamic = True
        device = "vulkan"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_frontend, shark_module, input)

    @pytest.mark.skip
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_benchmark_dynamic_vulkan_python(self):
        dynamic = True
        device = "vulkan"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_python, shark_module, input)

    @pytest.mark.skip
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_benchmark_dynamic_vulkan_c(self):
        dynamic = True
        device = "vulkan"
        shark_module, input = self.module_tester.create_minilm_module(dynamic, device)
        self.benchmark(self.module_tester.bench_c, shark_module)
