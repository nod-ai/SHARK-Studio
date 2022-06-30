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

import torch
import io
import pickle
from typing import Sequence, Union, List

from torch_mlir.dialects.torch.importer.jit_ir import (
    ClassAnnotator,
    ModuleBuilder,
)
from torch_mlir_e2e_test.torchscript.serialization import (
    extract_serializable_annotations,
    apply_serializable_annotations,
    SerializableTest,
)

from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from torch_mlir.ir import StringAttr
import torch_mlir
import sys


def get_module_name_for_asm_dump(module):
    """Gets a name suitable for an assembly dump.
    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(
        module.operation.attributes["torch.debug_module_name"]
    ).value


def get_input_annotations(inputs: tuple, dynamic: bool) -> list:
    """TODO: Include necessary documentation"""

    annotations_list = [None]
    for i in inputs:
        temp_list = []
        if dynamic:
            temp_list.append([-1 for i in range(len(i.shape))])
        else:
            temp_list.append(list(i.shape))
        temp_list.append(i.dtype)
        temp_list.append(True)
        annotations_list.append(tuple(temp_list))
    return annotations_list


def run_on_refbackend(torch_module, inputs):
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(torch_module)
    jit_module = backend.load(compiled)
    np_inputs = [x.numpy() for x in inputs]
    return jit_module.forward(np_inputs[0])


def get_torch_mlir_module(
    module,
    input: tuple,
    use_tracing: bool,
    dynamic_axis: Union[None, Sequence[List]] = None,
):
    """TODO: Include necessary documentation."""
    if dynamic_axis != None and len(dynamic_axis) != len(input):
        sys.stderr.write("Please mention the dynamic axis for all the inputs.")
        sys.exit(1)

    dyn_inputs = input
    if dynamic_axis != None:
        dyn_inputs = []
        for i, j in zip(input, dynamic_axis):
            dyn_inputs.append(
                torch_mlir.TensorPlaceholder.like(i, dynamic_axis=j)
            )

    module = torch_mlir.compile(
        module,
        input,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=use_tracing,
    )
    return module
