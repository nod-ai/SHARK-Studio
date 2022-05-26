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
    
def get_iree_compiled_module_tf(module, device: str):
    """Given a tf.Module returns the compiled .vmfb"""
    #Generate module for compiler using IREE
    compiler_module = tfc.compile_module(module, exported_names = ["forward"], import_only=True)

    #Compile the module using IREE
    args = ["--iree-llvm-target-cpu-features=host", "--iree-mhlo-demote-i64-to-i32=false", "--iree-flow-demote-i64-to-i32"]
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
            error_message = f"OS Typle f{os_name} not supported and triple can't be determined, open issue to dSHARK team please :)"
            raise Exception(error_message)
        print(f"Target triple found:{target_triple}")
        args.append(f"-iree-llvm-target-triple={target_triple}")
    flatbuffer_blob = ireec.compile_str(
        compiler_module, target_backends=[IREE_DEVICE_MAP[device]], extra_args=args, input_type="mhlo")
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    #tracer = ireert.Tracer(os.getcwd())
    #config = ireert.Config(IREE_DEVICE_MAP[device], tracer)
    config = ireert.Config(IREE_DEVICE_MAP[device])
    ctx = ireert.SystemContext(config=config)
    #TODO add optimization args.
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]
    return ModuleCompiled, config
    
def export_tf_iree_module_to_vmfb(module, device:str, directory: str):
    flatbuffer_blob = ireec.compile_str(
        compiler_module, target_backends=[IREE_DEVICE_MAP[device]], extra_args=args, input_type="mhlo")
    filename = os.path.join(directory, "tf_iree_module.vmfb")
    ##TODO:get module name for assembly dump like in torch.
    with open(filename, 'wb') as output_file:
        output_file.write(flatbuffer_blob)
        print(f"Wrote vmfb to path '{filename}'")
        
def get_results_tf(compiled_vm, input, config):
    """Runs a .vmfb file given inputs and config and returns output."""
    device_inputs = input
    #device_inputs = [ireert.asdevicearray(config.device, a) for a in input]

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
