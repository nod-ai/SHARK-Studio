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
import numpy as np

IREE_DEVICE_MAP = {"cpu": "dylib", "gpu": "cuda", "vulkan": "vulkan"}


def get_iree_compiled_module(module, device: str):
    """TODO: Documentation"""
    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]]
    )
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    config = ireert.Config(IREE_DEVICE_MAP[device])
    ctx = ireert.SystemContext(config=config)
    # TODO add optimisation args.
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]
    return ModuleCompiled, config


def get_results(compiled_vm, input, config):
    """TODO: Documentation"""

    # TODO: Support returning multiple outputs.
    device_inputs = [ireert.asdevicearray(config.device, a) for a in input]
    result = compiled_vm(*device_inputs)
    result_numpy = np.asarray(result, dtype=result.dtype)
    # TODO: Segfault if the copy of numpy array is not returned.
    result_copy = np.copy(result_numpy)
    # {k:v.to_host() for k, v in device_outputs.items(
    return result_copy

