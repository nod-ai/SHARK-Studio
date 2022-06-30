from masked_lm import get_causal_lm_model
from tank.model_utils_tf import compare_tensors_tf
from shark.iree_utils._common import check_device_drivers
from shark.shark_inference import SharkInference
from shark.parser import shark_args

import iree.compiler as ireec
import unittest
import pytest
import numpy as np
import tempfile
import os


class FlauBertModuleTester:
    def __init__(
        self,
        save_temps=False,
        save_mlir=False,
        save_vmfb=False,
        benchmark=False,
    ):
        self.save_temps = save_temps
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self, dynamic, device):
        model, input, act_out = get_causal_lm_model("hf-internal-testing/tiny-random-flaubert")
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb

        if shark_args.save_mlir == True or shark_args.save_vmfb == True or self.save_temps == True:
            repro_path = f"./shark_tmp/tiny_flaubert_tf_{dynamic}_{device}"
            if not os.path.isdir(repro_path):
                os.mkdir(repro_path)
            shark_args.repro_dir = repro_path

        if self.save_temps == True:
            temp_dir = tempfile.mkdtemp(prefix="iree_tfs", dir=shark_args.repro_dir)
            np.set_printoptions(threshold=np.inf)
            np.save(f"{temp_dir}/input1.npy", input[0])
            np.save(f"{temp_dir}/input2.npy", input[1])
            exp_out = act_out.numpy()
            with open(f"{temp_dir}/expected_out.txt", "w") as out_file:
                out_file.write(np.array2string(exp_out))
            with ireec.tools.TempFileSaver(temp_dir):
                shark_module = SharkInference(
                    model,
                    (input,),
                    device=device,
                    dynamic=dynamic,
                    jit_trace=True,
                    benchmark_mode=self.benchmark,
                )
                shark_module.set_frontend("tensorflow")
                shark_module.compile()
                results = shark_module.forward((input))
            assert True == compare_tensors_tf(act_out, results)

        else:
            shark_module = SharkInference(
                model,
                (input,),
                device=device,
                dynamic=dynamic,
                jit_trace=True,
                benchmark_mode=self.benchmark,
            )
            shark_module.set_frontend("tensorflow")
            shark_module.compile()
            results = shark_module.forward((input))
            assert True == compare_tensors_tf(act_out, results)

        if self.benchmark == True:
            shark_module.benchmark_all_csv((input), "tiny-random-flaubert", dynamic, device)


class FlauBertModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = FlauBertModuleTester(self)
        self.module_tester.save_temps = pytestconfig.getoption("save_temps")
        self.module_tester.save_mlir = pytestconfig.getoption("save_mlir")
        self.module_tester.save_vmfb = pytestconfig.getoption("save_vmfb")
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="Language models currently failing for dynamic case")
    def test_module_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(check_device_drivers("gpu"), reason="nvidia-smi not found")
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail
    @pytest.mark.skipif(
        check_device_drivers("vulkan"),
        reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases",
    )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="Language models currently failing for dynamic case")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"),
        reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases",
    )
    def test_module_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
