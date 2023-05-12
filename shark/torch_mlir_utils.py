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

from torch_mlir.ir import StringAttr
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
import tempfile
from shark.parser import shark_args
import io

mlir_type_mapping_dict = {
    "linalg": torch_mlir.OutputType.LINALG_ON_TENSORS,
    "stablehlo": torch_mlir.OutputType.STABLEHLO,
    "tosa": torch_mlir.OutputType.TOSA,
}


def get_module_name_for_asm_dump(module):
    """Gets a name suitable for an assembly dump.
    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(
        module.operation.attributes["torch.debug_module_name"]
    ).value


def run_on_refbackend(torch_module, inputs):
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(torch_module)
    jit_module = backend.load(compiled)
    np_inputs = [x.numpy() for x in inputs]
    return jit_module.forward(np_inputs[0])


# Creates dynamic dims for all dims.
# TODO: Pass user specified dynamic dims.
def create_dynamic_placeholders(inputs):
    placeholders = []
    for inp in inputs:
        placeholder = torch_mlir.TensorPlaceholder.like(
            inp, dynamic_axes=[i for i in range(len(inp.shape))]
        )
        placeholders.append(placeholder)
    return tuple(placeholders)


def get_torch_mlir_module(
    module,
    input: tuple,
    dynamic: bool,
    jit_trace: bool,
    return_str: bool = False,
    mlir_type: str = "linalg",
):
    """Get the MLIR's linalg-on-tensors module from the torchscipt module."""
    ignore_traced_shapes = False
    if dynamic:
        input = create_dynamic_placeholders(input)
    if jit_trace:
        ignore_traced_shapes = True

    tempfile.tempdir = "."

    mlir_module = torch_mlir.compile(
        module,
        input,
        output_type=mlir_type_mapping_dict[mlir_type],
        use_tracing=jit_trace,
        ignore_traced_shapes=ignore_traced_shapes,
    )

    if return_str:
        return mlir_module.operation.get_asm()
    bytecode_stream = io.BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    return bytecode
