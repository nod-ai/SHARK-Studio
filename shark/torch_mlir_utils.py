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
import sys
import os

from io import StringIO
from torch_mlir.dialects.torch.importer.jit_ir import (
    ClassAnnotator,
    ModuleBuilder,
)
from torch_mlir_e2e_test.torchscript.serialization import (
    extract_serializable_annotations,
    apply_serializable_annotations,
    SerializableTest
)

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export
from torch_mlir.ir import StringAttr

def get_module_name_for_asm_dump(module):
    """Gets a name suitable for an assembly dump.
    The name is not guaranteed to be unique.
    """
    if not "torch.debug_module_name" in module.operation.attributes:
        return "UnnammedModule"
    return StringAttr(module.operation.attributes["torch.debug_module_name"]).value
    
def export_module_to_mlir_file(module, directory: str):
    """Writes MLIR module to /tmp/module.mlir for debugging or performance use."""
    module_name = get_module_name_for_asm_dump(module)
    asm = module.operation.get_asm()
    filename = os.path.join(directory, module_name + ".mlir")
    with open(filename, 'w') as f:
        f.write(asm)

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


def shark_jit_trace(
    module, input: tuple, dynamic: bool, tracing_required: bool
):
    """TODO: Include necessary documentation."""

    if not tracing_required:
        return torch.jit.script(module)

    traced_module = torch.jit.trace_module(module, {"forward": input})
    actual_script = traced_module._actual_script_module
    export(actual_script.forward)
    annotate_args_decorator = annotate_args(
        get_input_annotations(input, dynamic)
    )
    annotate_args_decorator(actual_script.forward)
    module = torch.jit.script(actual_script)

    # TODO: remove saved annotations.pickle
    torchscript_module_bytes = module.save_to_buffer(
        {
            "annotations.pkl": pickle.dumps(
                extract_serializable_annotations(module)
            )
        }
    )
    serializable_test = SerializableTest(
        unique_name="", program=torchscript_module_bytes, trace=None
    )
    _extra_files = {"annotations.pkl": ""}
    module = torch.jit.load(
        io.BytesIO(serializable_test.program), _extra_files=_extra_files
    )
    # Load the pickled annotations.
    annotations = pickle.loads(_extra_files["annotations.pkl"])
    apply_serializable_annotations(module, annotations)
    return module


def get_torch_mlir_module(
    module,
    input: tuple,
    dynamic: bool,
    tracing_required: bool,
    from_aot: bool,
):
    """TODO: Include necessary documentation."""

    # Tracing is not required from the aot_module.
    if not from_aot:
        module = shark_jit_trace(module, input, dynamic, tracing_required)

    mb = ModuleBuilder()
    class_annotator = ClassAnnotator()
    class_annotator.exportNone(module._c._type())
    class_annotator.exportPath(module._c._type(), ["forward"])
    class_annotator.annotateArgs(
        module._c._type(),
        ["forward"],
        get_input_annotations(input, dynamic),
    )
    mb.import_module(module._c, class_annotator)

    passes = "torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline"
    # Set `SHARK_DEBUG` environment variable to log the intermediate mlir output.
    if 'SHARK_DEBUG' in os.environ and os.environ['SHARK_DEBUG']:
        mb.module.dump()

    with mb.module.context:
        pm = PassManager.parse(passes)
        pm.run(mb.module)

    return mb.module
