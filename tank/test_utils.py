from shark.iree_utils._common import (
    check_device_drivers,
    device_driver_info,
    IREE_DEVICE_MAP,
)
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag
from parameterized import parameterized
from shark.shark_downloader import download_tf_model
from shark.shark_inference import SharkInference
from shark.parser import shark_args
import pytest
import unittest
import numpy as np
import csv


def load_csv_and_convert(filename, gen=False):
    """
    takes in a csv filename and generates a dict for consumption by get_valid_test_params
    """
    model_configs = []
    with open(filename, "r+") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) < 5:
                print("invalid model: "+row)
                continue
            model_configs.append({"model_name":row[0], "dialect":row[1], "framework":row[2],  "rtol":float(row[3]),"atol":float(row[4])})
    #This is a pytest workaround
    if gen:
        with open("tank/dict_configs.py", "w+") as out:
            out.write("ALL = [\n")
            for c in model_configs:
                out.write(str(c) + ",\n")
            out.write("]")
    return model_configs


def get_valid_test_params():
    """
    Generate a list of all combinations of available devices and static/dynamic flag.
    """
    device_list = [
        device
        for device in IREE_DEVICE_MAP.keys()
        if not check_device_drivers(device)
    ]
    dynamic_list = (True, False)
    #TODO: This is soooo ugly, but for some reason creating the dict at runtime
    # results in strange pytest failures. 
    load_csv_and_convert("tank/all_models.csv", True)
    from tank.dict_configs import ALL
    config_list = ALL
    param_list = [
        (dynamic, device, config) for dynamic in dynamic_list for device in device_list for config in config_list
    ]
    return param_list


def shark_test_name_func(testcase_func, param_num, param):
    """
    Generate function name string which shows dynamic/static and device name.
    this will be ingested by 'parameterized' package to rename the pytest.
    """
    param_names = []
    for x in param.args:
        if x == True:
            param_names.append("dynamic")
        elif x == False:
            param_names.append("static")
        elif "model" in str(x):
            as_list = str(x).split(" ")
            as_list = [parameterized.to_safe_name(x).strip("_") for x in as_list]
            param_names.insert(0, as_list[as_list.index("model_name")+1])
            #param_names.append(as_list[3])

        else:
            param_names.append(x)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param_names)),
    )

class SharkModuleTester():
    def __init__(self, config): 
        """config should be a dict containing minimally:
                dialect: (str) name of input dialect
                framework: (str) one of tf, tflite, pytorch
                model_name: (str) name of the model in the tank ("resnet50")
                rtol/atol: (float) tolerances for golden values
        """ 
        self.config = config
    
    def create_and_check_module(self, dynamic, device):
        if self.config["framework"] is "tf":
            model, func_name, inputs, golden_out = download_tf_model(self.config["model_name"])
        else:
            model, func_name, inputs, golden_out = None, None, None, None

        shark_module = SharkInference(
            model,
            func_name,
            device=device,
            mlir_dialect=self.config["dialect"],
            is_benchmark=self.benchmark,
        )
        shark_module.compile()
        result = shark_module.forward(inputs)
        golden_out, result = self.postprocess_outputs(golden_out, result)

        np.testing.assert_allclose(golden_out, result, rtol=self.config["rtol"], atol=self.config["atol"])

        if self.benchmark == True:
            shark_args.enable_tf32 = True
            shark_args.onnx_bench = self.onnx_bench
            shark_module.shark_runner.benchmark_all_csv(
                (inputs), self.config["model_name"], dynamic, device, "tensorflow"
            )

    def postprocess_outputs(self, golden_out, result):
        # Prepares result tensors of forward pass and golden values for comparison, when needed.
        if self.config["model_name"] in ["google_vit-base-patch16-224", "facebook_convnext-tiny-224"]:
            ir_device_array = result[0][1]
            logits = ir_device_array.astype(ir_device_array.dtype)
            logits = np.squeeze(logits, axis=0)
            expected = golden_out[0]
        elif self.config["model_name"] == "microsoft_MiniLM-L12-H384-uncased":
            logits = result[0][1].to_host()
            expected = golden_out
        else:
            logits = result
            expected = golden_out
    
        return expected, logits

class SharkModuleTest(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def configure(self, pytestconfig):    
        self.pytestconfig = pytestconfig
    param_list = get_valid_test_params()

    @parameterized.expand(param_list, name_func=shark_test_name_func)
    def test_module(self, dynamic, device, config):
        self.module_tester = SharkModuleTester(config)
        self.module_tester.benchmark = self.pytestconfig.getoption("benchmark")
        self.module_tester.onnx_bench = self.pytestconfig.getoption("onnx_bench")
        
        if config["model_name"] == "facebook_convnext-tiny-224" and device == "cuda":
            pytest.xfail(reason = "https://github.com/nod-ai/SHARK/issues/311")
        if config["model_name"] == "google_vit-base-patch16-224" and device == "cuda":
            pytest.xfail(reason = "https://github.com/nod-ai/SHARK/issues/311")
        if config["model_name"] == "resnet50" and device in ["metal", "vulkan"]:
            if "m1-moltenvk-macos" in get_vulkan_triple_flag():
                pytest.xfail(reason = "M2: Assert Error & M1: CompilerToolError")
        if config["model_name"] == "roberta-base" and device == "cuda":
            pytest.xfail(reason="https://github.com/nod-ai/SHARK/issues/274")
        if config["model_name"] == "google_rembert":
            pytest.skip(reason="Model too large to convert.")
        if config["model_name"] == "dbmdz_convbert-base-turkish-cased" and device in ["metal", "vulkan"]:
            pytest.xfail(
                reason="Issue: https://github.com/iree-org/iree/issues/9971"
            )
        if config["model_name"] == "facebook/convnext-tiny-224" and device in ["cuda", "metal", "vulkan"]:
            pytest.xfail(reason="https://github.com/nod-ai/SHARK/issues/311, https://github.com/nod-ai/SHARK/issues/342")
        if config["model_name"] == "funnel-transformer_small" and device in ["cuda", "metal", "vulkan"]:
            pytest.xfail(reason="failing in the iree-compiler passes, see https://github.com/nod-ai/SHARK/issues/201")
        if config["model_name"] == "google_vit-base-patch16-224" and device == "cuda":
            pytest.xfail(reason="https://github.com/nod-ai/SHARK/issues/311")
        if config["model_name"] == "microsoft_mpnet-base":
            pytest.xfail(reason="https://github.com/nod-ai/SHARK/issues/203")

        self.module_tester.create_and_check_module(dynamic, device)
