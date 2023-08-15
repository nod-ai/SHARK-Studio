from shark.iree_utils._common import (
    check_device_drivers,
    device_driver_info,
    get_supported_device_list,
)
from shark.iree_utils.vulkan_utils import get_vulkan_triple_flag
from shark.parser import shark_args
from parameterized import parameterized
import iree.compiler as ireec
import pytest
import unittest
import numpy as np
import csv
import tempfile
import os
import sys
import shutil


def load_csv_and_convert(filename, gen=False):
    """
    takes in a csv filename and generates a dict for consumption by get_valid_test_params
    """
    model_configs = []
    with open(filename, "r+") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) < 5:
                print("invalid model: " + row)
                continue
            model_configs.append(
                {
                    "model_name": row[0],
                    "dialect": row[1],
                    "framework": row[2],
                    "rtol": float(row[3]),
                    "atol": float(row[4]),
                    "out_type": row[5],
                    "flags": row[6],
                    "xfail_cpu": row[7],
                    "xfail_cuda": row[8],
                    "xfail_vkm": row[9],
                    "xfail_reason": row[10],
                    "xfail_other": row[11],
                }
            )
    # This is a pytest workaround
    if gen:
        with open(
            os.path.join(os.path.dirname(__file__), "dict_configs.py"), "w+"
        ) as out:
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
        for device in get_supported_device_list()
        if not check_device_drivers(device)
        and device not in ["cpu-sync", "cpu-task"]
    ]
    dynamic_list = (True, False)
    # TODO: This is soooo ugly, but for some reason creating the dict at runtime
    # results in strange pytest failures.
    load_csv_and_convert(
        os.path.join(os.path.dirname(__file__), "all_models.csv"), True
    )
    from tank.dict_configs import ALL

    config_list = ALL

    param_list = [
        (dynamic, device, config)
        for dynamic in dynamic_list
        for device in device_list
        for config in config_list
    ]

    filtered_param_list = [
        params for params in param_list if is_valid_case(params)
    ]

    return filtered_param_list


def is_valid_case(test_params):
    if test_params[0] == True and test_params[2]["framework"] == "tf":
        return False
    if test_params[2]["framework"] == "tf":
        return False
    elif "fp16" in test_params[2]["model_name"] and test_params[1] != "cuda":
        return False
    else:
        return True


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
            as_list = [
                parameterized.to_safe_name(x).strip("_") for x in as_list
            ]
            param_names.insert(0, as_list[as_list.index("model_name") + 1])
            param_names.insert(1, as_list[as_list.index("framework") + 1])
            # param_names.append(as_list[3])

        else:
            param_names.append(x)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param_names)),
    )


class SharkModuleTester:
    def __init__(self, config):
        """config should be a dict containing minimally:
        dialect: (str) name of input dialect
        framework: (str) one of tf, tflite, pytorch
        model_name: (str) name of the model in the tank ("resnet50")
        rtol/atol: (float) tolerances for golden values
        """
        self.config = config

    def create_and_check_module(self, dynamic, device):
        shark_args.update_tank = self.update_tank
        shark_args.force_update_tank = self.force_update_tank
        shark_args.shark_prefix = self.shark_tank_prefix
        shark_args.local_tank_cache = self.local_tank_cache
        shark_args.dispatch_benchmarks = self.benchmark_dispatches
        shark_args.enable_tf32 = self.tf32

        if self.benchmark_dispatches is not None:
            _m = self.config["model_name"].split("/")
            _m.extend([self.config["framework"], str(dynamic), device])
            _m = "_".join(_m)
            shark_args.dispatch_benchmarks_dir = os.path.join(
                self.dispatch_benchmarks_dir,
                _m,
            )
            if not os.path.exists(self.dispatch_benchmarks_dir):
                os.mkdir(self.dispatch_benchmarks_dir)
            if not os.path.exists(shark_args.dispatch_benchmarks_dir):
                os.mkdir(shark_args.dispatch_benchmarks_dir)
        if "nhcw-nhwc" in self.config["flags"] and not os.path.isfile(
            ".use-iree"
        ):
            shark_args.enable_conv_transform = True
        else:
            shark_args.enable_conv_transform = False
        if "img2col" in self.config["flags"]:
            shark_args.enable_img2col_transform = True
        if "winograd" in self.config["flags"]:
            shark_args.use_winograd = True

        import_config = {
            "batch_size": self.batch_size,
        }

        from shark.shark_downloader import download_model
        from shark.shark_inference import SharkInference
        from tank.generate_sharktank import NoImportException

        dl_gen_attempts = 2
        for i in range(dl_gen_attempts):
            try:
                model, func_name, inputs, golden_out = download_model(
                    self.config["model_name"],
                    frontend=self.config["framework"],
                    import_args=import_config,
                )
            except NoImportException as err:
                pytest.xfail(
                    reason=f"Artifacts for this model/config must be generated locally. Please make sure {self.config['framework']} is installed."
                )
            except AssertionError as err:
                if i < dl_gen_attempts - 1:
                    continue
                else:
                    pytest.xfail(
                        "Generating OTF may require exiting the subprocess for files to be available."
                    )
            break
        is_bench = True if self.benchmark is not None else False
        shark_module = SharkInference(
            model,
            device=device,
            mlir_dialect=self.config["dialect"],
            is_benchmark=is_bench,
        )

        try:
            shark_module.compile()
        except:
            if any([self.ci, self.save_repro, self.save_fails]) == True:
                self.save_reproducers()
            if self.ci == True:
                self.upload_repro()
            raise

        result = shark_module(func_name, inputs)
        golden_out, result = self.postprocess_outputs(golden_out, result)
        if self.tf32 == True:
            print(
                "Validating with relaxed tolerances for TensorFloat32 calculations."
            )
            self.config["atol"] = 1e-01
            self.config["rtol"] = 1e-02
        try:
            np.testing.assert_allclose(
                golden_out,
                result,
                rtol=self.config["rtol"],
                atol=self.config["atol"],
            )
        except AssertionError as msg:
            if any([self.ci, self.save_repro, self.save_fails]) == True:
                self.save_reproducers()
            if self.ci == True:
                self.upload_repro()
            if self.benchmark is not None:
                self.benchmark_module(
                    shark_module, inputs, dynamic, device, mode=self.benchmark
                )
                print(msg)
                pytest.xfail(
                    reason=f"Numerics Mismatch: Use -s flag to print stderr during pytests."
                )
        if self.benchmark is not None:
            self.benchmark_module(
                shark_module, inputs, dynamic, device, mode=self.benchmark
            )

        if self.save_repro == True:
            self.save_reproducers()

    def benchmark_module(
        self, shark_module, inputs, dynamic, device, mode="native"
    ):
        model_config = {
            "batch_size": self.batch_size,
        }

        shark_args.onnx_bench = self.onnx_bench
        shark_module.shark_runner.benchmark_all_csv(
            (inputs),
            self.config["model_name"],
            dynamic,
            device,
            self.config["framework"],
            import_args=model_config,
            mode=mode,
        )

    def save_reproducers(self):
        # Saves contents of IREE TempFileSaver temporary directory to ./{temp_dir}/saved/<test_case>.
        src = self.temp_dir
        trg = os.path.join("reproducers", self.tmp_prefix)
        if not os.path.isdir("reproducers"):
            os.mkdir("reproducers")
        if not os.path.isdir(trg):
            os.mkdir(trg)
        files = os.listdir(src)
        for fname in files:
            shutil.copy2(os.path.join(src, fname), trg)

    def upload_repro(self):
        import subprocess

        repro_path = os.path.join("reproducers", self.tmp_prefix, "*")

        bashCommand = f"gsutil cp -r {repro_path} gs://shark-public/builder/repro_artifacts/{self.ci_sha}/{self.tmp_prefix}/"
        process = subprocess.run(bashCommand.split())

    def postprocess_outputs(self, golden_out, result):
        # Prepares result tensors of forward pass and golden values for comparison, when needed.
        if self.config["out_type"] == "tf_vit":
            ir_device_array = result[0][1]
            logits = ir_device_array.astype(ir_device_array.dtype)
            logits = np.squeeze(logits, axis=0)
            expected = golden_out[0]
        elif self.config["out_type"] == "tf_hf":
            logits = result[0][1].to_host()
            expected = golden_out
        elif self.config["out_type"] == "default":
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
        self.module_tester.batch_size = self.pytestconfig.getoption(
            "batchsize"
        )
        self.module_tester.benchmark = self.pytestconfig.getoption("benchmark")
        self.module_tester.save_repro = self.pytestconfig.getoption(
            "save_repro"
        )
        self.module_tester.save_fails = self.pytestconfig.getoption(
            "save_fails"
        )
        self.module_tester.onnx_bench = self.pytestconfig.getoption(
            "onnx_bench"
        )
        self.module_tester.tf32 = self.pytestconfig.getoption("tf32")
        self.module_tester.ci = self.pytestconfig.getoption("ci")
        self.module_tester.ci_sha = self.pytestconfig.getoption("ci_sha")
        self.module_tester.local_tank_cache = self.pytestconfig.getoption(
            "local_tank_cache"
        )
        self.module_tester.update_tank = self.pytestconfig.getoption(
            "update_tank"
        )
        self.module_tester.force_update_tank = self.pytestconfig.getoption(
            "force_update_tank"
        )
        self.module_tester.shark_tank_prefix = self.pytestconfig.getoption(
            "tank_prefix"
        )
        self.module_tester.benchmark_dispatches = self.pytestconfig.getoption(
            "benchmark_dispatches"
        )
        self.module_tester.dispatch_benchmarks_dir = (
            self.pytestconfig.getoption("dispatch_benchmarks_dir")
        )

        if config["xfail_cpu"] == "True" and device in [
            "cpu",
            "cpu-sync",
            "cpu-task",
        ]:
            pytest.xfail(reason=config["xfail_reason"])

        if config["xfail_cuda"] == "True" and device == "cuda":
            pytest.xfail(reason=config["xfail_reason"])

        if config["xfail_vkm"] == "True" and device in ["metal", "vulkan"]:
            pytest.xfail(reason=config["xfail_reason"])

        if (
            self.pytestconfig.getoption("ci") == True
            and os.name == "nt"
            and "enabled_windows" not in config["xfail_other"]
        ):
            pytest.xfail(reason="this model skipped on windows")

        # Special cases that need to be marked.
        if (
            "macos" in config["xfail_other"]
            and device
            in [
                "metal",
                "vulkan",
            ]
            and sys.platform == "darwin"
        ):
            pytest.skip(
                reason="conv-related issue on MacStudio, returns VK_ERROR_DEVICE_LOST."
            )
        if (
            config["model_name"]
            in [
                "facebook/convnext-tiny-224",
                "squeezenet1_0",
            ]
            and device == "rocm"
        ):
            pytest.xfail(
                reason="iree-compile buffer limit issue: https://github.com/nod-ai/SHARK/issues/475"
            )
        if (
            config["model_name"]
            in [
                "funnel-transformer/small",
                "mobilenet_v3_small",
            ]
            and device == "rocm"
        ):
            pytest.xfail(
                reason="Numerics issues: https://github.com/nod-ai/SHARK/issues/476"
            )
        if config["framework"] == "tf" and self.module_tester.batch_size != 1:
            pytest.xfail(
                reason="Configurable batch sizes temp. unavailable for tensorflow models."
            )
        safe_name = (
            f"{config['model_name']}_{config['framework']}_{dynamic}_{device}"
        )
        self.module_tester.tmp_prefix = safe_name.replace("/", "_")

        tempdir = tempfile.TemporaryDirectory(
            prefix=self.module_tester.tmp_prefix, dir="."
        )
        self.module_tester.temp_dir = tempdir.name

        with ireec.tools.TempFileSaver(tempdir.name):
            self.module_tester.create_and_check_module(dynamic, device)
