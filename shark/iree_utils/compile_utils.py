# Copyright 2020 The Nod Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import iree.runtime as ireert
import iree.compiler as ireec
from shark.iree_utils._common import iree_device_map, iree_target_map
from shark.iree_utils.benchmark_utils import *
from shark.parser import shark_args
import numpy as np
import os
import re


# Get the iree-compile arguments given device.
def get_iree_device_args(device, extra_args=[]):
    device_uri = device.split("://")
    if len(device_uri) > 1:
        if device_uri[0] not in ["vulkan"]:
            print(
                f"Specific device selection only supported for vulkan now."
                f"Proceeding with {device} as device."
            )

    if device_uri[0] == "cpu":
        from shark.iree_utils.cpu_utils import get_iree_cpu_args

        return get_iree_cpu_args()
    if device_uri[0] == "cuda":
        from shark.iree_utils.gpu_utils import get_iree_gpu_args

        return get_iree_gpu_args()
    if device_uri[0] in ["metal", "vulkan"]:
        from shark.iree_utils.vulkan_utils import get_iree_vulkan_args

        return get_iree_vulkan_args(extra_args=extra_args)
    if device_uri[0] == "rocm":
        from shark.iree_utils.gpu_utils import get_iree_rocm_args

        return get_iree_rocm_args()
    return []


# Get the iree-compiler arguments given frontend.
def get_iree_frontend_args(frontend):
    if frontend in ["torch", "pytorch", "linalg"]:
        return ["--iree-llvm-target-cpu-features=host"]
    elif frontend in ["tensorflow", "tf", "mhlo"]:
        return [
            "--iree-llvm-target-cpu-features=host",
            "--iree-mhlo-demote-i64-to-i32=false",
            "--iree-flow-demote-i64-to-i32",
        ]
    else:
        # Frontend not found.
        return []


# Common args to be used given any frontend or device.
def get_iree_common_args():
    return [
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        "--iree-util-zero-fill-elided-attrs",
    ]


# Args that are suitable only for certain models or groups of models.
# shark_args are passed down from pytests to control which models compile with these flags,
# but they can also be set in shark/parser.py
def get_model_specific_args():
    ms_args = []
    if shark_args.enable_conv_transform == True:
        ms_args += ["--iree-flow-enable-conv-nchw-to-nhwc-transform"]
    return ms_args


def create_dispatch_dirs(bench_dir, device):
    protected_files = ["ordered-dispatches.txt"]
    bench_dir_path = bench_dir.split("/")
    bench_dir_path[-1] = "temp_" + bench_dir_path[-1]
    tmp_bench_dir = "/".join(bench_dir_path)
    for f_ in os.listdir(bench_dir):
        if os.path.isfile(f"{bench_dir}/{f_}") and f_ not in protected_files:
            dir_name = re.sub("\.\S*$", "", f_)
            if os.path.exists(f"{bench_dir}/{dir_name}"):
                os.system(f"rm -rf {bench_dir}/{dir_name}")
            os.system(f"mkdir {bench_dir}/{dir_name}")
            os.system(f"mv {bench_dir}/{f_} {bench_dir}/{dir_name}/{f_}")
    for f_ in os.listdir(tmp_bench_dir):
        if os.path.isfile(f"{tmp_bench_dir}/{f_}"):
            dir_name = ""
            for d_ in os.listdir(bench_dir):
                if re.search(f"{d_}(?=\D)", f_):
                    dir_name = d_
            if dir_name != "":
                os.system(
                    f"mv {tmp_bench_dir}/{f_} {bench_dir}/{dir_name}/{dir_name}_benchmark.mlir"
                )


def dump_isas(bench_dir):
    for d_ in os.listdir(bench_dir):
        if os.path.isdir(f"{bench_dir}/{d_}"):
            for f_ in os.listdir(f"{bench_dir}/{d_}"):
                if f_.endswith(".spv"):
                    os.system(
                        f"amdllpc -gfxip 11.0 {bench_dir}/{d_}/{f_} -v > \
                         {bench_dir}/{d_}/isa.txt"
                    )


def compile_benchmark_dirs(bench_dir, device, dispatch_benchmarks):
    benchmark_runtimes = {}
    dispatch_list = []
    all_dispatches = False

    if dispatch_benchmarks.lower().strip() == "all":
        all_dispatches = True
    else:
        try:
            dispatch_list = [
                int(dispatch_index)
                for dispatch_index in dispatch_benchmarks.split(" ")
            ]
        except:
            print("ERROR: Invalid dispatch benchmarks")
            return None
    for d_ in os.listdir(bench_dir):
        if os.path.isdir(f"{bench_dir}/{d_}"):
            in_dispatches = False
            for dispatch in dispatch_list:
                if str(dispatch) in d_:
                    in_dispatches = True
            if all_dispatches or in_dispatches:
                for f_ in os.listdir(f"{bench_dir}/{d_}"):
                    if "benchmark.mlir" in f_:
                        dispatch_file = open(f"{bench_dir}/{d_}/{f_}", "r")
                        module = dispatch_file.read()
                        dispatch_file.close()

                        flatbuffer_blob = ireec.compile_str(
                            module, target_backends=[iree_target_map(device)]
                        )

                        vmfb_file = open(
                            f"{bench_dir}/{d_}/{d_}_benchmark.vmfb", "wb"
                        )
                        vmfb_file.write(flatbuffer_blob)
                        vmfb_file.close()

                        config = get_iree_runtime_config(device)
                        vm_module = ireert.VmModule.from_flatbuffer(
                            config.vm_instance, flatbuffer_blob
                        )

                        benchmark_cl = build_benchmark_args_non_tensor_input(
                            input_file=f"{bench_dir}/{d_}/{d_}_benchmark.vmfb",
                            device=device,
                            inputs=(0,),
                            mlir_dialect="linalg",
                            function_name="",
                        )

                        benchmark_bash = open(
                            f"{bench_dir}/{d_}/{d_}_benchmark.sh", "w+"
                        )
                        benchmark_bash.write("#!/bin/bash\n")
                        benchmark_bash.write(" ".join(benchmark_cl))
                        benchmark_bash.close()

                        benchmark_data = run_benchmark_module(benchmark_cl)

                        benchmark_file = open(
                            f"{bench_dir}/{d_}/{d_}_data.txt", "w+"
                        )
                        benchmark_file.write(f"DISPATCH: {d_}\n")
                        benchmark_file.write(str(benchmark_data) + "\n")
                        benchmark_file.write(
                            "SHARK BENCHMARK RESULT: "
                            + str(1 / (benchmark_data * 0.001))
                            + "\n"
                        )
                        benchmark_file.close()

                        benchmark_runtimes[d_] = 1 / (benchmark_data * 0.001)

                    elif ".mlir" in f_ and "benchmark" not in f_:
                        dispatch_file = open(f"{bench_dir}/{d_}/{f_}", "r")
                        module = dispatch_file.read()
                        dispatch_file.close()

                        module = re.sub(
                            "hal.executable private",
                            "hal.executable public",
                            module,
                        )

                        flatbuffer_blob = ireec.compile_str(
                            module,
                            target_backends=[iree_target_map(device)],
                            extra_args=["--compile-mode=hal-executable"],
                        )

                        spirv_file = open(
                            f"{bench_dir}/{d_}/{d_}_spirv.vmfb", "wb"
                        )
                        spirv_file.write(flatbuffer_blob)
                        spirv_file.close()

    ordered_dispatches = [
        (k, v)
        for k, v in sorted(
            benchmark_runtimes.items(), key=lambda item: item[1]
        )
    ][::-1]
    f_ = open(f"{bench_dir}/ordered-dispatches.txt", "w+")
    for dispatch in ordered_dispatches:
        f_.write(f"{dispatch[0]}: {dispatch[1]}ms\n")
    f_.close()


def compile_module_to_flatbuffer(
    module,
    device,
    frontend,
    model_config_path,
    extra_args,
    model_name="None",
):
    # Setup Compile arguments wrt to frontends.
    input_type = ""
    args = get_iree_frontend_args(frontend)
    args += get_iree_device_args(device, extra_args)
    args += get_iree_common_args()
    args += get_model_specific_args()
    args += extra_args

    if frontend in ["tensorflow", "tf"]:
        input_type = "mhlo"
    elif frontend in ["mhlo", "tosa"]:
        input_type = frontend
    elif frontend in ["tflite", "tflite-tosa"]:
        input_type = "tosa"
    elif frontend in ["tm_tensor"]:
        input_type = ireec.InputType.TM_TENSOR

    # TODO: make it simpler.
    # Compile according to the input type, else just try compiling.
    if input_type != "":
        # Currently for MHLO/TOSA.
        flatbuffer_blob = ireec.compile_str(
            module,
            target_backends=[iree_target_map(device)],
            extra_args=args,
            input_type=input_type,
        )
    else:
        # Currently for Torch.
        flatbuffer_blob = ireec.compile_str(
            module,
            target_backends=[iree_target_map(device)],
            extra_args=args,
        )

    return flatbuffer_blob


def get_iree_module(flatbuffer_blob, device, device_idx=None):
    # Returns the compiled module and the configs.
    if device_idx is not None:
        print("registering device id: ", device_idx)
        haldriver = ireert.get_driver(device)

        haldevice = haldriver.create_device(
            haldriver.query_available_devices()[device_idx]["device_id"]
        )
        # haldevice = haldriver.create_default_device()
        config = ireert.Config(device=haldevice)
    else:
        config = get_iree_runtime_config(device)
    vm_module = ireert.VmModule.from_flatbuffer(
        config.vm_instance, flatbuffer_blob
    )
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module
    return ModuleCompiled, config


def get_iree_compiled_module(
    module,
    device: str,
    frontend: str = "torch",
    model_config_path: str = None,
    extra_args: list = [],
    device_idx: int = None,
):
    """Given a module returns the compiled .vmfb and configs"""
    flatbuffer_blob = compile_module_to_flatbuffer(
        module, device, frontend, model_config_path, extra_args
    )
    return get_iree_module(flatbuffer_blob, device, device_idx=device_idx)


def load_flatbuffer(flatbuffer_path: str, device: str, device_idx: int = None):
    with open(os.path.join(flatbuffer_path), "rb") as f:
        flatbuffer_blob = f.read()

    return get_iree_module(flatbuffer_blob, device, device_idx=device_idx)


def export_iree_module_to_vmfb(
    module,
    device: str,
    directory: str,
    mlir_dialect: str = "linalg",
    model_config_path: str = None,
    module_name: str = None,
    extra_args: list = [],
):
    # Compiles the module given specs and saves it as .vmfb file.
    flatbuffer_blob = compile_module_to_flatbuffer(
        module, device, mlir_dialect, model_config_path, extra_args
    )
    if module_name is None:
        device_name = (
            device if "://" not in device else "-".join(device.split("://"))
        )
        module_name = f"{mlir_dialect}_{device_name}"
    filename = os.path.join(directory, module_name + ".vmfb")
    print(f"Saved vmfb in {filename}.")
    with open(filename, "wb") as f:
        f.write(flatbuffer_blob)
    return filename


def export_module_to_mlir_file(module, frontend, directory: str):
    # TODO: write proper documentation.
    mlir_str = module
    if frontend in ["tensorflow", "tf", "mhlo", "tflite"]:
        mlir_str = module.decode("utf-8")
    elif frontend in ["pytorch", "torch"]:
        mlir_str = module.operation.get_asm()
    filename = os.path.join(directory, "model.mlir")
    with open(filename, "w") as f:
        f.write(mlir_str)
    print(f"Saved mlir in {filename}.")
    return filename


def get_results(
    compiled_vm,
    function_name,
    input,
    config,
    frontend="torch",
    send_to_host=True,
):
    """Runs a .vmfb file given inputs and config and returns output."""
    device_inputs = [ireert.asdevicearray(config.device, a) for a in input]
    result = compiled_vm[function_name](*device_inputs)
    result_tensors = []
    if isinstance(result, tuple):
        if send_to_host:
            for val in result:
                result_tensors.append(np.asarray(val, val.dtype))
        else:
            for val in result:
                result_tensors.append(val)
        return result_tensors
    elif isinstance(result, dict):
        data = list(result.items())
        if send_to_host:
            res = np.array(data, dtype=object)
            return np.copy(res)
        return data
    else:
        if send_to_host and result is not None:
            return result.to_host()
        return result


def get_iree_runtime_config(device):
    device = iree_device_map(device)
    config = ireert.Config(device=ireert.get_device(device))
    return config
