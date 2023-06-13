import unittest
import os
import pytest
import torch
import numpy as np
from shark_opt_wrapper import OPTForCausalLMModel
from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM

OPT_MODEL = "facebook/opt-1.3b"
OPT_FS_NAME = "opt-1_3b"
OPT_MODEL_66B = "facebook/opt-66b"


class OPTModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        opt_model = OPTForCausalLM.from_pretrained(
            model_name, return_dict=False
        )
        opt_model.eval()

        model_inputs = tokenizer(
            "The meaning of life is",
            padding="max_length",
            max_length=30,
            truncation=True,
            return_tensors="pt",
        )
        inputs = (
            model_inputs.data["input_ids"],
            model_inputs.data["attention_mask"],
        )
        act_out = opt_model(
            inputs[0], attention_mask=inputs[1], return_dict=False
        )[0]
        (
            mlir_module,
            func_name,
        ) = import_with_fx(
            model=opt_model,
            inputs=inputs,
            is_f16=False,
            model_name=OPT_FS_NAME,
        )
        del opt_model
        opt_filename = f"./{OPT_FS_NAME}_causallm_30_torch_{device}"
        mlir_path = os.path.join(opt_filename, ".mlir")
        with open(mlir_path, "w") as f:
            f.write(mlir_module)
        print(f"Saved mlir at {mlir_path}")

        shark_module = SharkInference(
            mlir_module,
            device=device,
            mlir_dialect="tm_tensor",
            is_benchmark=self.benchmark,
        )

        shark_module.compile()
        results = shark_module("forward", inputs)
        print(
            "SHARK logits have shape: ",
            str(results[0].shape) + " : " + str(results[0]),
        )
        print(
            "PyTorch logits have shape: "
            + str(act_out[0].shape)
            + " : "
            + str(act_out[0])
        )
        # exp_out = tokenizer.decode(act_out[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # shark_out = tokenizer.decode(results[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        np.testing.assert_allclose(act_out[0].detach(), results[0])

        if self.benchmark:
            shark_module.shark_runner.benchmark_all_csv(
                inputs,
                "opt",
                dynamic,
                device,
                "torch",
            )


class OPTModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.module_tester = OPTModuleTester(self)
        self.module_tester.save_mlir = False
        self.module_tester.save_vmfb = False
        self.module_tester.benchmark = pytestconfig.getoption("benchmark")

    def test_1_3b_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    def test_1_3b_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("cuda"), reason=device_driver_info("cuda")
    )
    def test_1_3b_static_cuda(self):
        dynamic = False
        device = "cuda"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("cuda"), reason=device_driver_info("cuda")
    )
    def test_1_3b_dynamic_cuda(self):
        dynamic = True
        device = "cuda"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_1_3b_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_1_3b_dynamic_vulkan(self):
        dynamic = True
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    # def test_66b_static_cpu(self):
    #    dynamic = False
    #    device = "cpu"
    #    self.module_tester.create_and_check_module(
    #        dynamic, device, OPT_MODEL_66B
    #    )

    # def test_66b_dynamic_cpu(self):
    #    dynamic = True
    #    device = "cpu"
    #    self.module_tester.create_and_check_module(
    #        dynamic, device, OPT_MODEL_66B
    #    )

    # @pytest.mark.skipif(
    #    check_device_drivers("cuda"), reason=device_driver_info("cuda")
    # )
    # def test_66b_static_cuda(self):
    #    dynamic = False
    #    device = "cuda"
    #    self.module_tester.create_and_check_module(
    #        dynamic, device, OPT_MODEL_66B
    #    )

    # @pytest.mark.skipif(
    #    check_device_drivers("cuda"), reason=device_driver_info("cuda")
    # )
    # def test_66b_dynamic_cuda(self):
    #    dynamic = True
    #    device = "cuda"
    #    self.module_tester.create_and_check_module(
    #        dynamic, device, OPT_MODEL_66B
    #    )

    # @pytest.mark.skipif(
    #    check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    # )
    # def test_66b_static_vulkan(self):
    #    dynamic = False
    #    device = "vulkan"
    #    self.module_tester.create_and_check_module(
    #        dynamic, device, OPT_MODEL_66B
    #    )

    # @pytest.mark.skipif(
    #    check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    # )
    # def test_66b_dynamic_vulkan(self):
    #    dynamic = True
    #    device = "vulkan"
    #    self.module_tester.create_and_check_module(
    #        dynamic, device, OPT_MODEL_66B
    #    )


if __name__ == "__main__":
    unittest.main()
