from shark.shark_inference import SharkInference
from shark.iree_utils import check_device_drivers
from tank.model_utils import get_vision_model, compare_tensors
from shark.parser import shark_args

import iree.compiler as ireec
import torch
import torchvision.models as models
import unittest
import numpy as np
import pytest
import tempfile

torch.manual_seed(0)

class Resnet50ModuleTester:

    def __init__(
        self,
        save_temps=False,
        save_mlir=False,
        save_vmfb=False,
        benchmark=False
    ):
        self.save_temps = save_temps
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, input, act_out = get_vision_model(models.resnet50(pretrained=True))
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        if self.save_temps == True:
            if dynamic == True:
                repro_dir = f"resnet50_dynamic_{device}"
            else:
                repro_dir = f"resnet50_static_{device}"
            temp_dir = tempfile.mkdtemp(prefix=repro_dir)
            np.set_printoptions(threshold=np.inf)
            np.save(f"{temp_dir}/input.npy", input[0])
            exp_out = act_out.detach().numpy()
            with open(f"{temp_dir}/expected_out.txt", "w") as out_file:
                out_file.write(np.array2string(exp_out))
            with ireec.tools.TempFileSaver(temp_dir):
                shark_module = SharkInference(model, (input,),
                                              device=device,
                                              dynamic=dynamic,
                                              benchmark_mode=self.benchmark)
                shark_module.compile()
                results = shark_module.forward((input,))
            assert True == compare_tensors(act_out, results)

        else:
            shark_module = SharkInference(model, (input,),
                                          device=device,
                                          dynamic=dynamic,
                                          benchmark_mode=self.benchmark)
            shark_module.compile()
            results = shark_module.forward((input,))
            assert True == compare_tensors(act_out, results)

        if self.benchmark == True:
            shark_module.benchmark_all_csv((input,),
                                           "resnet50",
                                           dynamic,
                                           device)
                                           

class Resnet50ModuleTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig): 
        self.module_tester = Resnet50ModuleTester(self)
        self.module_tester.save_temps = pytestconfig.getoption("save_temps")
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
    
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)
 
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="https://github.com/google/iree/issues/9554")
    @pytest.mark.skipif(
            check_device_drivers("vulkan"),
            reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="https://github.com/google/iree/issues/9554")
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
