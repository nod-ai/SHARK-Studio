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
from iree.compiler import tf as tfc
import subprocess
import numpy as np
import os

IREE_DEVICE_MAP = {
    "cpu": "dylib",
    "gpu": "cuda",
    "cuda": "cuda",
    "vulkan": "vulkan",
    "metal": "vulkan"
}


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
    return ["--iree-hal-cuda-disable-loop-nounroll-wa"]


def get_iree_vulkan_args():
    return [
        "--iree-flow-demote-i64-to-i32=false",
        "--iree-flow-demote-f64-to-f32=true"
    ]


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


def get_results(compiled_vm, input, config, frontend="torch"):
    """Runs a .vmfb file given inputs and config and returns output."""
    device_inputs = input
    if frontend in ["torch", "pytorch"]:
        device_inputs = [ireert.asdevicearray(config.device, a) for a in input]

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
