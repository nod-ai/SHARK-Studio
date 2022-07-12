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

import sys
import json
import os
from typing import List, Dict

from iree.compiler import ir
from iree.compiler.transforms import ireec as ireec_trans

MATMUL_OP_NAMES = set(
    ["linalg.matmul", "linalg.batch_matmul", "mhlo.dot", "mhlo.dot_general"]
)
idx = 0


def model_annotation(
    ctx: ir.Context, *, input_contents: str, config_path: str
):
    if os.path.isfile(input_contents):
        with open(input_contents, "rb") as f:
            input_contents = f.read()

    module = ir.Module.parse(input_contents)

    with open(config_path, "r") as f:
        data = json.load(f)
        configs = data["options"]

    # The Python API does not expose a general walk() function, so we just
    # do it ourselves.
    walk_children(module.operation, configs)

    if not module.operation.verify():
        raise RuntimeError("Modified program does not verify!")

    # More efficient than: print(module)
    #   - Disables verification (already done above)
    #   - Writes as binary, avoiding costly unicode conversions
    sys.stdout.buffer.write(
        module.operation.get_asm(assume_verified=True, binary=True)
    )
    return module


def walk_children(op: ir.Operation, configs: List[Dict]):
    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                # TODO: This is dumb. Both Operation and OpView should expose
                # 'operation' and 'name' attributes.
                if isinstance(child_op, ir.OpView):
                    child_op = child_op.operation
                if child_op.name in MATMUL_OP_NAMES:
                    global idx
                    (
                        tile_sizes,
                        pipeline,
                        workgroup_size,
                        split_k,
                        pipeline_depth,
                    ) = parse_config(configs[idx])

                    add_compilation_info(
                        child_op,
                        tile_sizes=tile_sizes,
                        pipeline=pipeline,
                        workgroup_size=workgroup_size,
                        pipeline_depth=pipeline_depth,
                    )

                    if split_k:
                        add_split_k(child_op, split_k)

                    idx = idx + 1
                    print(f"Updated op {child_op}", file=sys.stderr)
                walk_children(child_op, configs)


def parse_config(config: Dict):
    if config["pipeline"] == "GPU" or config["pipeline"] == "GPU_TENSORCORE":
        pipeline = (
            "LLVMGPUMatmulSimt"
            if config["pipeline"] == "GPU"
            else "LLVMGPUMatmulTensorCore"
        )
        tile_sizes = [config["work_group_tile_sizes"]]
        workgroup_size = config["work_group_sizes"]
        try:
            pipeline_depth = config["pipeline_depth"]
        except:
            pipeline_depth = None
        try:
            split_k = config["split_k"]
        except:
            split_k = None
    else:
        pipeline = config["pipeline"]
        tile_sizes = [
            config["work_group_tile_sizes"],
            config["l1_tile_sizes"],
            config["vector_tile_sizes"],
        ]
        workgroup_size = []
        split_k = None
        pipeline_depth = None
    return tile_sizes, pipeline, workgroup_size, split_k, pipeline_depth


def add_compilation_info(
    op: ir.Operation,
    tile_sizes: List[List[int]],
    pipeline: str,
    workgroup_size: List[int],
    pipeline_depth: int,
):
    # We don't have a Python binding for CompilationInfo, so we just parse
    # its string form.
    if pipeline_depth:
        attr = ir.Attribute.parse(
            f"#iree_codegen.compilation_info<"
            f"lowering_config = <tile_sizes = {repr(tile_sizes)}>, "
            f"translation_info = <{pipeline} pipeline_depth = {pipeline_depth}>, "
            f"workgroup_size = {repr(workgroup_size)}>"
        )
    else:
        attr = ir.Attribute.parse(
            f"#iree_codegen.compilation_info<"
            f"lowering_config = <tile_sizes = {repr(tile_sizes)}>, "
            f"translation_info = <{pipeline}>, "
            f"workgroup_size = {repr(workgroup_size)}>"
        )
    op.attributes["compilation_info"] = attr


def add_split_k(op: ir.Operation, k: int):
    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), k)
    op.attributes["iree_flow_split_k"] = attr


def create_context() -> ir.Context:
    context = ir.Context()
    ireec_trans.register_all_dialects(context)
    context.allow_unregistered_dialects = True
    return context


if __name__ == "__main__":
    with create_context() as ctx:
        model_annotation(
            ctx, input_contents=sys.argv[1], config_path=sys.argv[2]
        )
