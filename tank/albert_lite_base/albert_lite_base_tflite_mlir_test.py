from shark.shark_downloader import SharkDownloader
from shark.shark_inference import SharkInference
import pytest
import unittest
from shark.parser import shark_args


class AlbertTfliteModuleTester:
    def __init__(
        self,
        dynamic=False,
        device="cpu",
        save_mlir=False,
        save_vmfb=False,
    ):
        self.dynamic = dynamic
        self.device = device
        self.save_mlir = save_mlir
        self.save_vmfb = save_vmfb

    def create_and_check_module(self):
        shark_args.save_mlir = self.save_mlir
        shark_args.save_vmfb = self.save_vmfb
        shark_downloader = SharkDownloader(
            model_name="albert_lite_base",
            tank_url="https://storage.googleapis.com/shark_tank",
            local_tank_dir="./../gen_shark_tank",
            model_type="tflite-tosa",
            input_json="input.json",
            input_type="int32",
        )
        tflite_tosa_model = shark_downloader.get_mlir_file()
        inputs = shark_downloader.get_inputs()

        shark_module = SharkInference(
            mlir_module=tflite_tosa_model,
            function_name="main",
            device=self.device,
            mlir_dialect="tflite",
        )
        shark_module.compile()
        shark_module.forward(inputs)
        # print(shark_results)


class AlbertTfliteModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):
        self.save_mlir = pytestconfig.getoption("save_mlir")
        self.save_vmfb = pytestconfig.getoption("save_vmfb")

    def setUp(self):
        self.module_tester = AlbertTfliteModuleTester(self)
        self.module_tester.save_mlir = self.save_mlir

    def test_module_static_cpu(self):
        self.module_tester.dynamic = False
        self.module_tester.device = "cpu"
        self.module_tester.create_and_check_module()


if __name__ == "__main__":
    # unittest.main()
    module_tester = AlbertTfliteModuleTester()
    module_tester.create_and_check_module()

# TEST RESULT:
# (shark.venv) nod% python albert_lite_base_tflite_mlir_test.py
# load json inputs
# TMP_MODEL_DIR = shark/SHARK/shark/./../gen_shark_tank/tflite
# Model has not been download.shark_downloader will automatically download by tank_url if provided. You can also manually to download the model from shark_tank by yourself.
# TMP_MODELNAME_DIR = shark/SHARK/shark/./../gen_shark_tank/tflite/albert_lite_base
# Download mlir model https://storage.googleapis.com/shark_tank/tflite/albert_lite_base/albert_lite_base_tosa.mlir
# Get tosa.mlir model return
# Target triple found:x86_64-linux-gnu
# (shark.venv) nod% python albert_lite_base_tflite_mlir_test.py
# load json inputs
# TMP_MODEL_DIR = shark/SHARK/shark/./../gen_shark_tank/tflite
# TMP_MODELNAME_DIR = shark/SHARK/shark/./../gen_shark_tank/tflite/albert_lite_base
# Model has been downloaded before. shark/SHARK/shark/./../gen_shark_tank/tflite/albert_lite_base/albert_lite_base_tosa.mlir
# Get tosa.mlir model return
# Target triple found:x86_64-linux-gnu
#
