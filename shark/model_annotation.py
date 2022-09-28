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

import json
import os
import sys
from typing import Dict, List

from iree.compiler import ir
from iree.compiler.transforms import ireec as ireec_trans


def model_annotation(
    ctx: ir.Context,
    *,
    input_contents: str,
    config_path: str,
    search_op: str = "matmul",
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
    walk_children(module.operation, configs, 0, search_op)

    if not module.operation.verify():
        raise RuntimeError("Modified program does not verify!")

    return module


def walk_children(
    op: ir.Operation, configs: List[Dict], idx: int, search_op: str
):
    if search_op == "matmul":
        op_names = ["linalg.matmul", "mhlo.dot"]
    elif search_op == "bmm":
        op_names = ["linalg.batch_matmul", "mhlo.dot_general"]
    elif search_op == "conv":
        op_names = ["mhlo.convolution", "linalg.conv_2d_nhwc_hwcf"]
    elif search_op == "all":
        op_names = [
            "mhlo.dot",
            "mhlo.dot_general",
            "mhlo.convolution",
            "linalg.matmul",
            "linalg.batch_matmul",
            "linalg.conv_2d_nhwc_hwcf",
        ]
    else:
        raise ValueError(f"{search_op} op is not tunable.")

    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                # TODO: This is dumb. Both Operation and OpView should expose
                # 'operation' and 'name' attributes.
                if isinstance(child_op, ir.OpView):
                    child_op = child_op.operation
                if child_op.name in op_names and idx < len(configs):
                    add_attributes(child_op, configs[idx])
                    idx = idx + 1
                    print(f"Updated op {child_op}", file=sys.stderr)
                walk_children(child_op, configs, idx, search_op)


def add_attributes(op: ir.Operation, config: Dict):
    (
        tile_sizes,
        pipeline,
        workgroup_size,
        split_k,
        pipeline_depth,
        devices,
        shard_sizes,
    ) = parse_config(config)

    add_compilation_info(
        op,
        tile_sizes=tile_sizes,
        pipeline=pipeline,
        workgroup_size=workgroup_size,
        pipeline_depth=pipeline_depth,
    )

    if split_k:
        add_attribute_by_name(op, "iree_flow_split_k", split_k)
    if devices:
        add_attribute_by_name(op, "devices", devices)
    if shard_sizes:
        add_attribute_by_name(op, "shard_sizes", shard_sizes)

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
        try:
            devices = config["devices"]
        except:
            devices = None
        try:
            shard_sizes = config["shard_sizes"]
        except:
            shard_sizes = None
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
        devices = None
        shard_sizes = None
    return tile_sizes, pipeline, workgroup_size, split_k, pipeline_depth, devices, shard_sizes


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


def add_attribute_by_name(op: ir.Operation, name: str, val: int):
    if isinstance(val, list):
        attr = ir.ArrayAttr.get(array_from_list(val))    
    else:
        attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)
    op.attributes[name] = attr

def array_from_list(val_list):
    item_list = []
    for arr in val_list:
        if isinstance(arr, list):
            attr = ir.ArrayAttr.get(array_from_list(arr))
            item_list.append(attr)
        else:
            attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), arr)
            item_list.append(attr)
    return item_list

def create_context() -> ir.Context:
    context = ir.Context()
    ireec_trans.register_all_dialects(context)
    context.allow_unregistered_dialects = True
    return context


if __name__ == "__main__":
    with create_context() as ctx:
        module = model_annotation(
            ctx,
            input_contents=sys.argv[1],
            config_path=sys.argv[2],
            search_op="all",
        )
        mlir_str = str(module)
        filename = "tuned_model.mlir"
        with open(filename, "w") as f:
            f.write(mlir_str)
        print(f"Saved mlir in {filename}.")
