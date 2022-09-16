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
from shark.iree_utils._common import IREE_DEVICE_MAP, IREE_TARGET_MAP
import numpy as np
import os

# Get the iree-compile arguments given device.
def get_iree_device_args(device):
    if device == "cpu":
        from shark.iree_utils.cpu_utils import get_iree_cpu_args

        return get_iree_cpu_args()
    if device == "cuda":
        from shark.iree_utils.gpu_utils import get_iree_gpu_args

        return get_iree_gpu_args()
    if device in ["metal", "vulkan"]:
        from shark.iree_utils.vulkan_utils import get_iree_vulkan_args

        return get_iree_vulkan_args()
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
    ]


def compile_module_to_flatbuffer(
    module, device, frontend, func_name, model_config_path
):
    # Setup Compile arguments wrt to frontends.
    input_type = ""
    args = get_iree_frontend_args(frontend)
    args += get_iree_device_args(device)
    args += get_iree_common_args()

    if frontend in ["tensorflow", "tf"]:
        input_type = "mhlo"
    elif frontend in ["mhlo", "tosa"]:
        input_type = frontend
    elif frontend in ["tflite", "tflite-tosa"]:
        input_type = "tosa"
    elif frontend in ["tm_tensor"]:
        input_type = frontend

    # TODO: make it simpler.
    # Compile according to the input type, else just try compiling.
    if input_type not in ["mhlo", "tosa"]:
        module = str(module)
    if input_type != "":
        # Currently for MHLO/TOSA.
        flatbuffer_blob = ireec.compile_str(
            module,
            target_backends=[IREE_TARGET_MAP[device]],
            extra_args=args,
            input_type=input_type,
        )
    else:
        # Currently for Torch.
        flatbuffer_blob = ireec.compile_str(
            str(module),
            target_backends=[IREE_TARGET_MAP[device]],
            extra_args=args,
        )

    return flatbuffer_blob


def get_iree_module(flatbuffer_blob, device, func_name):
    # Returns the compiled module and the configs.
    config = ireert.Config(IREE_DEVICE_MAP[device])
    vm_module = ireert.VmModule.from_flatbuffer(
        config.vm_instance, flatbuffer_blob
    )
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module[func_name]
    return ModuleCompiled, config


def get_iree_compiled_module(
    module,
    device: str,
    frontend: str = "torch",
    func_name: str = "forward",
    model_config_path: str = None,
):
    """Given a module returns the compiled .vmfb and configs"""
    flatbuffer_blob = compile_module_to_flatbuffer(
        module, device, frontend, func_name, model_config_path
    )
    return get_iree_module(flatbuffer_blob, device, func_name)


def export_iree_module_to_vmfb(
    module,
    device: str,
    directory: str,
    mlir_dialect: str = "linalg",
    func_name: str = "forward",
    model_config_path: str = None,
):
    # Compiles the module given specs and saves it as .vmfb file.
    flatbuffer_blob = compile_module_to_flatbuffer(
        module, device, mlir_dialect, func_name, model_config_path
    )
    module_name = f"{mlir_dialect}_{func_name}_{device}"
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


def get_results(compiled_vm, input, config, frontend="torch"):
    """Runs a .vmfb file given inputs and config and returns output."""
    device_inputs = [ireert.asdevicearray(config.device, a) for a in input]
    result = compiled_vm(*device_inputs)
    result_tensors = []
    if isinstance(result, tuple):
        for val in result:
            result_tensors.append(np.copy(np.asarray(val, val.dtype)))
        return result_tensors
    elif isinstance(result, dict):
        data = list(result.items())
        res = np.array(data, dtype=object)
        return np.copy(res)
    else:
        return np.copy(np.asarray(result, dtype=result.dtype))
