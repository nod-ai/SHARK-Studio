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
import iree.runtime.scripts.iree_benchmark_module as benchmark_module
import iree.compiler as ireec
from iree.compiler import tf as tfc
from shark.torch_mlir_utils import get_module_name_for_asm_dump
from shark.cuda_utils import get_cuda_sm_cc
import subprocess
import numpy as np
import os
import re
import sys

IREE_DEVICE_MAP = {
    "cpu": "dylib",
    "gpu": "cuda",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan"
}

UNIT_TO_SECOND_MAP = {"ms": 0.001, "s": 1}


def check_device_drivers(device):
    """Checks necessary drivers present for gpu and vulkan devices"""
    if (device in ["gpu", "cuda"]):
        try:
            subprocess.check_output('nvidia-smi')
        except Exception:
            return True
    elif (device in ["metal", "vulkan"]):
        try:
            subprocess.check_output('vulkaninfo')
        except Exception:
            return True
    elif (device == "cpu"):
        return False
    # Unknown device.
    else:
        return True

    return False


def get_iree_cpu_args():
    find_triple_cmd = "uname -s -m"
    os_name, proc_name = subprocess.run(
        find_triple_cmd, shell=True, stdout=subprocess.PIPE,
        check=True).stdout.decode('utf-8').split()
    if os_name == "Darwin":
        find_kernel_version_cmd = "uname -r"
        kernel_version = subprocess.run(find_kernel_version_cmd,
                                        shell=True,
                                        stdout=subprocess.PIPE,
                                        check=True).stdout.decode('utf-8')
        target_triple = f"{proc_name}-apple-darwin{kernel_version}"
    elif os_name == "Linux":
        target_triple = f"{proc_name}-linux-gnu"
    else:
        error_message = f"OS Type f{os_name} not supported and triple can't be determined, open issue to dSHARK team please :)"
        raise Exception(error_message)
    print(f"Target triple found:{target_triple}")
    return [f"-iree-llvm-target-triple={target_triple}"]


def get_iree_gpu_args():
    ireert.flags.FUNCTION_INPUT_VALIDATION = False
    ireert.flags.parse_flags("--cuda_allow_inline_execution")
    sm_arch = get_cuda_sm_cc()
    if sm_arch in ['sm_70', 'sm_72', 'sm_75', 'sm_80', 'sm_84', 'sm_86']:
        return [
            "--iree-hal-cuda-disable-loop-nounroll-wa",
            f"--iree-hal-cuda-llvm-target-arch={sm_arch}"
        ]
    else:
        return ["--iree-hal-cuda-disable-loop-nounroll-wa"]

def get_vulkan_triple_flag():
    vulkan_device_cmd = "vulkaninfo | grep deviceName | awk \'END{{print $3}}\'"
    vulkan_device = run_cmd(vulkan_device_cmd).strip()
    if vulkan_device == "apple":
        return "-iree-vulkan-target-triple=m1-moltenvk-macos"
    elif vulkan_device == "A100-SXM4-40GB":
        return "-iree-vulkan-target-triple=ampere-rtx3080-linux"
    else:
        print("Optimized kernel for your target device is not added yet. Contact SHARK Admin on discord or pull up an issue.")
        return None

def get_iree_vulkan_args():
    vulkan_flag = ["--iree-flow-demote-i64-to-i32"]
    vulkan_triple_flag = get_vulkan_triple_flag()
    if vulkan_triple_flag is not None:
        vulkan_flag.append(vulkan_triple_flag)
    return vulkan_flag

def get_iree_device_args(device):
    if device == "cpu":
        return get_iree_cpu_args()
    if device in ["gpu", "cuda"]:
        return get_iree_gpu_args()
    if device in ["metal", "vulkan"]:
        return get_iree_vulkan_args()
    return []


def get_iree_frontend_args(frontend):
    if frontend in ["torch", "pytorch", "linalg"]:
        return ["--iree-llvm-target-cpu-features=host"]
    elif frontend in ["tensorflow", "tf", "mhlo"]:
        return [
            "--iree-llvm-target-cpu-features=host",
            "--iree-mhlo-demote-i64-to-i32=false",
            "--iree-flow-demote-i64-to-i32"
        ]
    else:
        # Frontend not found.
        return []


# input_type should be "mhlo", "tosa" for linalg no need to mention the frontend.
def get_iree_module(module, device, input_type, args, func_name):
    flatbuffer_blob = None
    # Compile according to the input type, else just try compiling.
    if input_type in ["mhlo", "tosa"]:
        flatbuffer_blob = ireec.compile_str(
            str(module),
            target_backends=[IREE_DEVICE_MAP[device]],
            extra_args=args,
            input_type=input_type)
    else:
        flatbuffer_blob = ireec.compile_str(
            str(module),
            target_backends=[IREE_DEVICE_MAP[device]],
            extra_args=args)

    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    config = ireert.Config(IREE_DEVICE_MAP[device])
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module[func_name]
    return ModuleCompiled, config


def get_iree_compiled_module(module,
                             device: str,
                             frontend: str = "torch",
                             func_name: str = "forward"):
    """Given a module returns the compiled .vmfb and configs"""
    input_type = ""
    args = get_iree_frontend_args(frontend)
    args += get_iree_device_args(device)

    if frontend in ["tensorflow", "tf"]:
        module = tfc.compile_module(module,
                                    exported_names=[func_name],
                                    import_only=True)
        input_type = "mhlo"
    elif frontend in ["mhlo"]:
        input_type = "mhlo"
    elif frontend in ["tosa"]:
        input_type = "tosa"

    return get_iree_module(module, device, input_type, args, func_name)


def export_iree_module_to_vmfb(module, device: str, directory: str):
    module_name = get_module_name_for_asm_dump(module)
    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]])
    filename = os.path.join(directory, module_name + ".vmfb")
    with open(filename, 'wb') as f:
        f.write(flatbuffer_blob)
    return filename


def get_results(compiled_vm, input, config, frontend="torch"):
    """Runs a .vmfb file given inputs and config and returns output."""
    device_inputs = input
    if frontend in ["torch", "pytorch"]:
        device_inputs = [ireert.asdevicearray(config.device, a) for a in input]
    if frontend in ["tensorflow", "tf"]:
        device_inputs = []
        for a in input:
            if (isinstance(a, list)):
                device_inputs.append([ireert.asdevicearray(config.device, val, dtype=np.int32) for val in a])
            else:
                device_inputs.append(ireert.asdevicearray(config.device, a))
    result = compiled_vm(*device_inputs)
    result_tensors = []
    if (isinstance(result, tuple)):
        for val in result:
            result_tensors.append(np.copy(np.asarray(val, val.dtype)))
        return result_tensors
    elif (isinstance(result, dict)):
        data = list(result.items())
        res = np.array(data, dtype=object)
        return np.copy(res)
    else:
        return np.copy(np.asarray(result, dtype=result.dtype))


######### Benchmark Related Tools ###########


def tensor_to_type_str(input_tensors: tuple):
    """
    Input: A tuple of input tensors i.e tuple(torch.tensor)
    Output: list of string that represent mlir types (i.e 1x24xf64)
    # TODO: Support more than floats, and ints
    """
    list_of_type = []
    for input_tensor in input_tensors:
        type_string = "x".join([str(dim) for dim in input_tensor.shape])
        dtype_string = str(input_tensor.dtype).replace("torch.", "")
        regex_split = re.compile("([a-zA-Z]+)([0-9]+)")
        match = regex_split.match(dtype_string)
        mlir_type_string = str(match.group(1)[0]) + str(match.group(2))
        type_string += f"x{mlir_type_string}"
        list_of_type.append(type_string)
    return list_of_type


def build_benchmark_args(input_file: str,
                         device: str,
                         input_tensors: tuple,
                         training=False):
    """
    Inputs: input_file leading to vmfb, input_tensor to function, target device, and whether it is training or not.
    Outputs: string that execute benchmark-module on target model.
    """
    path = benchmark_module.__path__[0]
    benchmarker_path = os.path.join(path, "..", "..", "iree-benchmark-module")
    benchmark_cl = [benchmarker_path, f"--module_file={input_file}"]
    fn_name = "forward"
    if training == True:
        # TODO: Replace name of train with actual train fn name.
        fn_name = "train"
    benchmark_cl.append(f"--entry_function={fn_name}")
    benchmark_cl.append(f"--driver={IREE_DEVICE_MAP[device]}")
    mlir_input_types = tensor_to_type_str(input_tensors)
    for mlir_input in mlir_input_types:
        benchmark_cl.append(f"--function_input={mlir_input}")
    time_extractor = "| awk \'END{{print $2 $3}}\'"
    benchmark_cl.append(time_extractor)
    return benchmark_cl


def run_cmd(cmd):
    """
    Inputs: cli command string.
    """
    try:
        result = subprocess.run(cmd,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True)
        result_str = result.stdout.decode()
        return result_str
    except Exception:
        sys.exit("Exiting program due to error running:", cmd)


def run_benchmark(benchmark_cl):
    """
    Run benchmark command, extract result and return iteration/seconds.

    Input: benchmark command.
    """
    benchmark_path = benchmark_cl[0]
    assert os.path.exists(
        benchmark_path
    ), "Cannot find benchmark_module, Please contact SHARK maintainer on discord."
    bench_result = run_cmd(' '.join(benchmark_cl))
    regex_split = re.compile("([0-9]+[.]*[0-9]*)([a-zA-Z]+)")
    match = regex_split.match(bench_result)
    time = float(match.group(1))
    unit = match.group(2)
    return 1.0 / (time * UNIT_TO_SECOND_MAP[unit])
