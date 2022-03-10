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
import os

IREE_DEVICE_MAP = {"cpu": "dylib", "gpu": "cuda", "vulkan": "vulkan"}


def get_iree_compiled_module(module, device: str):
    """TODO: Documentation"""
    flatbuffer_blob = ireec.compile_str(
        str(module), target_backends=[IREE_DEVICE_MAP[device]]
    )
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config(IREE_DEVICE_MAP[device], tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    ModuleCompiled = ctx.modules.module["forward"]
    return ModuleCompiled


def get_results(compiled_vm, input):
    """TODO: Documentation"""

    # TODO: Currently only one output and input is supported.
    # Extend it to support multiple inputs and outputs.
    result = compiled_vm(input)
    result_numpy = np.asarray(result, dtype=result.dtype)

    # TODO: Segfault if the copy of numpy array is not returned.
    result_copy = np.copy(result_numpy)
    return result_copy
