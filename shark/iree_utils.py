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
import subprocess
import numpy as np
import os
from shark.torch_mlir_utils import get_module_name_for_asm_dump

IREE_DEVICE_MAP = {"cpu": "dylib", "gpu": "cuda", "vulkan": "vulkan"}


def get_iree_compiled_module(module, device: str):
    """TODO: Documentation"""
    args = ["--iree-llvm-target-cpu-features=host"]
    if (device == "cpu"):
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
        args.append(f"-iree-llvm-target-triple={target_triple}")
    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]], extra_args=args)
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    config = ireert.Config(IREE_DEVICE_MAP[device])
    ctx = ireert.SystemContext(config=config)
    # TODO add optimisation args.
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]
    return ModuleCompiled, config


def export_iree_module_to_vmfb(module, device: str, directory: str):
    module_name = get_module_name_for_asm_dump(module)
    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]])
    filename = os.path.join(directory, module_name + ".vmfb")
    with open(filename, 'wb') as f:
        f.write(flatbuffer_blob)


def get_results(compiled_vm, input, config):
    """TODO: Documentation"""
    device_inputs = [ireert.asdevicearray(config.device, a) for a in input]
    result = compiled_vm(*device_inputs)
    result_tensors = []
    if (isinstance(result, tuple)):
        for val in result:
            result_tensors.append(np.copy(np.asarray(val, val.dtype)))
        return result_tensors
    else:
        return np.copy(np.asarray(result, dtype=result.dtype))
