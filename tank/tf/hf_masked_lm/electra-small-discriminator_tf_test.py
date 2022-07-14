from tank.masked_lm_tf import get_causal_lm_model
from tank.model_utils_tf import compare_tensors_tf
from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
from shark.parser import shark_args

import iree.compiler as ireec
import unittest
import pytest
import numpy as np
import tempfile
import os


class ElectraModuleTester:
    def __init__(
        self,
        save_mlir=False,
        save_vmfb=False,
        save_temps=False,
        #       benchmark=False,
    ):
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb
        self.save_temps = save_temps

    #       self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device):
        model, input, act_out = get_causal_lm_model(
            "google/electra-small-discriminator"
        )
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb

        if (
            shark_args.save_mlir == True
            or shark_args.save_vmfb == True
            or self.save_temps == True
        ):
            repro_path = f"./shark_tmp/electra_small_discriminator_tf_{dynamic}_{device}"
            if not os.path.isdir(repro_path):
                os.mkdir(repro_path)
            shark_args.repro_dir = repro_path

        if self.save_temps == True:
            temp_dir = tempfile.mkdtemp(
                prefix="iree_tfs", dir=shark_args.repro_dir
            )
            np.set_printoptions(threshold=np.inf)
            np.save(f"{temp_dir}/input1.npy", input[0])
            np.save(f"{temp_dir}/input2.npy", input[1])
            exp_out = act_out.numpy()
            with open(f"{temp_dir}/expected_out.txt", "w") as out_file:
                out_file.write(np.array2string(exp_out))
            with ireec.tools.TempFileSaver(temp_dir):
                mlir_importer = SharkImporter(
                    model, (input,), frontend="tensorflow"
                )
                mlir_module, func_name = mlir_importer.import_mlir(
                    is_dynamic=dynamic, tracing_required=False
                )
                shark_module = SharkInference(
                    mlir_module, func_name, device=device, mlir_dialect="mhlo"
                )
                shark_module.compile()

        else:
            mlir_importer = SharkImporter(
                model,
                (input,),
                frontend="tensorflow",
            )
            mlir_module, func_name = mlir_importer.import_mlir(
                is_dynamic=dynamic, tracing_required=False
            )
            shark_module = SharkInference(
                mlir_module, func_name, device=device, mlir_dialect="mhlo"
            )
            shark_module.compile()

        results = shark_module.forward((input))
        assert True == compare_tensors_tf(act_out, results)


class ElectraModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = ElectraModuleTester(self)
        self.module_tester.save_temps = pytestconfig.getoption("save_temps")
        self.module_tester.save_mlir = pytestconfig.getoption("save_mlir")
        self.module_tester.save_vmfb = pytestconfig.getoption("save_vmfb")

    #        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    @pytest.mark.xfail(reason="https://github.com/iree-org/iree/issues/9536")
    def test_module_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="https://github.com/iree-org/iree/issues/9536")
    def test_module_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="https://github.com/google/iree/issues/9553")
    @pytest.mark.skipif(
        check_device_drivers("gpu"), reason=device_driver_info("gpu")
    )
    def test_module_static_gpu(self):
        dynamic = False
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.skip(reason="https://github.com/google/iree/issues/9553")
    @pytest.mark.skipif(
        check_device_drivers("gpu"), reason=device_driver_info("gpu")
    )
    def test_module_dynamic_gpu(self):
        dynamic = True
        device = "gpu"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="https://github.com/iree-org/iree/issues/9524")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_module_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)

    @pytest.mark.xfail(reason="https://github.com/iree-org/iree/issues/9524")
    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_module_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device)


if __name__ == "__main__":
    unittest.main()
