import unittest

import pytest
import torch_mlir
from hacked_hf_opt import OPTModel
from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from tank.model_utils import compare_tensors
from transformers import GPT2Tokenizer

OPT_MODEL = "facebook/opt-350m"
OPT_MODEL_66B = "facebook/opt-66b"


class OPTModuleTester:
    def __init__(
        self,
        benchmark=False,
    ):
        self.benchmark = benchmark

    def create_and_check_module(self, dynamic, device, model_name):
        # model_mlir, func_name, input, act_out = download_torch_model(
        #     "opt", dynamic
        # )

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # config = OPTConfig()
        # opt_model = OPTModel(config)
        opt_model = OPTModel.from_pretrained(model_name)
        opt_model.eval()

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        input_ids, attention_mask = (
            inputs.data["input_ids"],
            inputs.data["attention_mask"],
        )

        module = torch_mlir.compile(
            opt_model,
            (input_ids, attention_mask),
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=True,
        )

        model_mlir = module.operation.get_asm(
            large_elements_limit=None, enable_debug_info=True
        )
        func_name = "forward"
        act_out = opt_model(input_ids, attention_mask).detach()

        # mlir_importer = SharkImporter(
        #    model,
        #    (input,),
        #    frontend="torch",
        # )
        # minilm_mlir, func_name = mlir_importer.import_mlir(
        #    is_dynamic=dynamic, tracing_required=True
        # )

        shark_module = SharkInference(
            model_mlir,
            func_name,
            device=device,
            mlir_dialect="tm_tensor",
            is_benchmark=self.benchmark,
        )
        shark_module.compile()
        results = shark_module.forward((input_ids, attention_mask))
        assert compare_tensors(act_out, results)

        if self.benchmark:
            shark_module.shark_runner.benchmark_all_csv(
                (input_ids, attention_mask),
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

    def test_350m_static_cpu(self):
        dynamic = False
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    def test_350m_dynamic_cpu(self):
        dynamic = True
        device = "cpu"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("cuda"), reason=device_driver_info("cuda")
    )
    def test_350m_static_cuda(self):
        dynamic = False
        device = "cuda"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("cuda"), reason=device_driver_info("cuda")
    )
    def test_350m_dynamic_cuda(self):
        dynamic = True
        device = "cuda"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_350m_static_vulkan(self):
        dynamic = False
        device = "vulkan"
        self.module_tester.create_and_check_module(dynamic, device, OPT_MODEL)

    @pytest.mark.skipif(
        check_device_drivers("vulkan"), reason=device_driver_info("vulkan")
    )
    def test_350m_dynamic_vulkan(self):
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
