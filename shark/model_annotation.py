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

"""
Usage:
This function takes the model mlir file and the tuned config file as input,
and output a new mlir file with lowering configs annotated on certain ops.
There are two ways to utilize the function:
1. Call model_annotation function within another python script
from shark.model_annotation import model_annotation
with create_context() as ctx:
   module = model_annotation(ctx, input_contents=..., config_path=..., search_op=...)
2. Run model_annotation.py directly
python model_annotation.py -model path_to_original_mlir -config_path path_to_config_file
"""

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
    search_op: str,
    winograd: int = 0,
):
    if os.path.isfile(input_contents):
        with open(input_contents, "rb") as f:
            input_contents = f.read()
    module = ir.Module.parse(input_contents)

    configs = load_model_configs(config_path)

    # The Python API does not expose a general walk() function, so we just
    # do it ourselves.
    walk_children(module.operation, configs, search_op, winograd)

    if not module.operation.verify():
        raise RuntimeError("Modified program does not verify!")

    return module


def load_model_configs(config_path: str):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            data = json.loads(line)

            if "identifier" not in data.keys():
                continue
            if data["identifier"] == "matmul":
                matrix_size = [data["m"], data["n"], data["k"]]
            elif data["identifier"] == "bmm":
                matrix_size = [data["b"], data["m"], data["n"], data["k"]]
            elif data["identifier"] == "generic":
                matrix_size = [1, data["b"], data["m"], data["n"], data["k"]]
            elif data["identifier"] == "conv":
                matrix_size = [
                    data["n"],
                    data["ih"],
                    data["iw"],
                    data["c"],
                    data["kh"],
                    data["kw"],
                    data["f"],
                    data["oh"],
                    data["ow"],
                    data["d"],
                    data["s"],
                    data["p"],
                ]
            config[shape_list_to_string(matrix_size)] = data
        f.close()
        return config


def walk_children(
    op: ir.Operation, configs: List[Dict], search_op: str, winograd: int
):
    if search_op == "matmul":
        op_names = ["linalg.matmul", "mhlo.dot"]
    elif search_op == "bmm":
        op_names = ["linalg.batch_matmul", "mhlo.dot_general"]
    elif search_op == "conv":
        op_names = ["mhlo.convolution", "linalg.conv_2d_nhwc_hwcf"]
    elif search_op == "generic":
        op_names = ["linalg.generic"]
    elif search_op == "all":
        op_names = [
            "mhlo.dot",
            "mhlo.dot_general",
            "mhlo.convolution",
            "linalg.matmul",
            "linalg.batch_matmul",
            "linalg.conv_2d_nhwc_hwcf",
            "linalg.generic",
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
                if winograd and child_op.name in ["linalg.conv_2d_nchw_fchw"]:
                    add_winograd_attribute(winograd, child_op)
                if child_op.name in op_names:
                    if child_op.name == "linalg.generic":
                        # This is for generic op that has contractionOpInterface
                        # which is basically einsum("mk,bkn->bmn")
                        op_result = str(child_op.results[0])
                        op_iterator = str(
                            child_op.attributes["iterator_types"]
                        )
                        if len(child_op.operands) != 3:
                            continue
                        if "reduction" not in op_iterator:
                            continue
                        if (
                            "arith.addf" not in op_result
                            or "arith.mulf" not in op_result
                        ):
                            continue
                        if "arith.subf" in op_result:
                            continue

                    child_op_shape = get_op_shape(child_op, search_op)
                    if (
                        child_op_shape in configs.keys()
                        and configs[child_op_shape]["options"][0] != None
                    ):
                        add_attributes(
                            child_op, configs[child_op_shape]["options"][0]
                        )
                    print(f"Updated op {child_op}", file=sys.stderr)

                walk_children(child_op, configs, search_op, winograd)


def get_op_shape(op: ir.Operation, search_op: str):
    shape_list = []
    if search_op in ["generic", "all"]:
        if op.name in ["linalg.generic"]:
            input1 = str(op.operands[0].type)
            input2 = str(op.operands[1].type)
            m = input1.split("tensor<")[1].split("x")[0]
            b = input2.split("tensor<")[1].split("x")[0]
            k = input2.split("tensor<")[1].split("x")[1]
            n = input2.split("tensor<")[1].split("x")[2]
            shape_list = [1, int(b), int(m), int(n), int(k)]

    if search_op in ["matmul", "all"]:
        if op.name in ["mhlo.dot"]:
            op_result = str(op.results[0])
            m = op_result.split("tensor<")[1].split("x")[0]
            k = op_result.split("tensor<")[1].split("x")[1]
            n = op_result.split("tensor<")[2].split("x")[1]
            shape_list = [int(m), int(n), int(k)]
        elif op.name in ["linalg.matmul"]:
            op_result = str(op.results[0]).split("ins(")[1]
            m = op_result.split("tensor<")[1].split("x")[0]
            k = op_result.split("tensor<")[1].split("x")[1]
            n = op_result.split("tensor<")[2].split("x")[1]
            shape_list = [int(m), int(n), int(k)]

    if search_op in ["bmm", "all"]:
        if op.name in ["mhlo.dot_general"]:
            op_result = str(op.results[0])
            b = op_result.split("tensor<")[1].split("x")[1]
            m = op_result.split("tensor<")[1].split("x")[2]
            k = op_result.split("tensor<")[1].split("x")[3]
            n = op_result.split("tensor<")[3].split("x")[3]
            shape_list = [int(b), int(m), int(n), int(k)]
        elif op.name in ["linalg.batch_matmul"]:
            op_result = str(op.results[0]).split("ins(")[1]
            b = op_result.split("tensor<")[1].split("x")[0]
            m = op_result.split("tensor<")[1].split("x")[1]
            k = op_result.split("tensor<")[1].split("x")[2]
            n = op_result.split("tensor<")[3].split("x")[2]
            shape_list = [int(b), int(m), int(n), int(k)]

    if search_op in ["conv", "all"]:
        if op.name in ["mhlo.convolution"]:
            op_result = str(op.results[0])
            dilation = (
                str(op.attributes["rhs_dilation"])
                .split("dense<")[1]
                .split(">")[0]
            )
            stride = (
                str(op.attributes["window_strides"])
                .split("dense<")[1]
                .split(">")[0]
            )
            pad = (
                str(op.attributes["padding"]).split("dense<")[1].split(">")[0]
            )
            n = op_result.split("tensor<")[1].split("x")[0]
            ih = op_result.split("tensor<")[1].split("x")[1]
            iw = op_result.split("tensor<")[1].split("x")[2]
            c = op_result.split("tensor<")[1].split("x")[3]
            kh = op_result.split("tensor<")[2].split("x")[0]
            kw = op_result.split("tensor<")[2].split("x")[1]
            f = op_result.split("tensor<")[2].split("x")[3]
            oh = op_result.split("tensor<")[3].split("x")[1]
            ow = op_result.split("tensor<")[3].split("x")[2]
            shape_list = [
                int(n),
                int(ih),
                int(iw),
                int(c),
                int(kh),
                int(kw),
                int(f),
                int(oh),
                int(ow),
                int(dilation),
                int(stride),
                int(pad),
            ]

        elif op.name in ["linalg.conv_2d_nhwc_hwcf"]:
            op_result = str(op.results[0]).split("ins(")[1]
            dilation = (
                str(op.attributes["dilations"])
                .split("dense<")[1]
                .split(">")[0]
            )
            stride = (
                str(op.attributes["strides"]).split("dense<")[1].split(">")[0]
            )
            pad = 0
            n = op_result.split("tensor<")[1].split("x")[0]
            ih = op_result.split("tensor<")[1].split("x")[1]
            iw = op_result.split("tensor<")[1].split("x")[2]
            c = op_result.split("tensor<")[1].split("x")[3]
            kh = op_result.split("tensor<")[2].split("x")[0]
            kw = op_result.split("tensor<")[2].split("x")[1]
            f = op_result.split("tensor<")[2].split("x")[3]
            oh = op_result.split("tensor<")[3].split("x")[1]
            ow = op_result.split("tensor<")[3].split("x")[2]
            shape_list = [
                int(n),
                int(ih),
                int(iw),
                int(c),
                int(kh),
                int(kw),
                int(f),
                int(oh),
                int(ow),
                int(dilation),
                int(stride),
                int(pad),
            ]

    shape_str = shape_list_to_string(shape_list)
    return shape_str


def add_attributes(op: ir.Operation, config: List[Dict]):
    # Parse the config file
    split_k = None
    pipeline_depth = None
    store_stage = None
    subgroup_size = None

    if "GPU" in config["pipeline"]:
        pipeline = (
            "LLVMGPUMatmulSimt"
            if config["pipeline"] == "GPU"
            else "LLVMGPUMatmulTensorCore"
        )
        tile_sizes = [config["work_group_tile_sizes"]]
        workgroup_size = config["work_group_sizes"]
        if "pipeline_depth" in config.keys():
            pipeline_depth = config["pipeline_depth"]
        if "split_k" in config.keys():
            split_k = config["split_k"]
    elif "SPIRV" in config["pipeline"]:
        pipeline = config["pipeline"]
        tile_sizes = [
            config["work_group_tile_sizes"],
            config["parallel_tile_sizes"],
            config["reduction_tile_sizes"],
        ]
        workgroup_size = config["work_group_sizes"]
        if "vector_tile_sizes" in config.keys():
            tile_sizes += [config["vector_tile_sizes"]]
        if "window_tile_sizes" in config.keys():
            tile_sizes += [config["window_tile_sizes"]]
        if "subgroup_size" in config.keys():
            subgroup_size = config["subgroup_size"]
        if "pipeline_depth" in config.keys():
            pipeline_depth = config["pipeline_depth"]
        if "store_stage" in config.keys():
            store_stage = config["store_stage"]
    else:
        # For IREE CPU pipelines
        pipeline = config["pipeline"]
        tile_sizes = [
            config["work_group_tile_sizes"],
            config["parallel_tile_sizes"],
            config["reduction_tile_sizes"],
        ]
        workgroup_size = []

    # Add compilation info as an attribute. We don't have a Python binding for CompilationInfo,
    # so we just parse its string form.
    if pipeline_depth != None:
        translation_info = f"{pipeline} pipeline_depth = {pipeline_depth}"
        if store_stage != None:
            translation_info += f" store_stage = {store_stage}"
    else:
        translation_info = f"{pipeline}"

    compilation_info = (
        f"#iree_codegen.compilation_info<"
        f"lowering_config = <tile_sizes = {repr(tile_sizes)}>, "
        f"translation_info = <{translation_info}>, "
        f"workgroup_size = {repr(workgroup_size)} "
    )

    if subgroup_size != None:
        compilation_info += f", subgroup_size = {subgroup_size}>"
    else:
        compilation_info += ">"

    attr = ir.Attribute.parse(compilation_info)
    op.attributes["compilation_info"] = attr

    # Add other attributes if required.
    if split_k:
        add_attribute_by_name(op, "iree_flow_split_k", split_k)


def add_winograd_attribute(winograd: int, op: ir.Operation):
    op_result = str(op.results[0]).split("ins(")[1]
    dilation = int(
        str(op.attributes["dilations"]).split("dense<")[1].split(">")[0]
    )
    stride = int(
        str(op.attributes["strides"]).split("dense<")[1].split(">")[0]
    )
    kh = int(op_result.split("tensor<")[2].split("x")[2])
    kw = int(op_result.split("tensor<")[2].split("x")[3])
    c = int(op_result.split("tensor<")[2].split("x")[0])
    f = int(op_result.split("tensor<")[2].split("x")[1])

    # Selected conv ops to use Winograd for 1) Unet fp16 model and 2) VAE fp16 model
    # TODO: Add winograd selections to a config json file
    if winograd == 1:  # Unet fp16 model
        if (
            dilation == 1
            and stride == 1
            and kh == 3
            and kw == 3
            and (
                (c > 4 and c < 1280)
                and (f > 4 and f <= 1280)
                or (c == 1280 and f == 640)
            )
        ):
            op.attributes["iree_winograd_conv"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), 1
            )
            print("Apply Winograd on Unet selected conv op: ", op)
    elif winograd == 2:  # VAE fp16 model
        if (
            dilation == 1
            and stride == 1
            and kh == 3
            and kw == 3
            and c > 3
            and f == 512
        ):
            op.attributes["iree_winograd_conv"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), 1
            )
            print("Apply Winograd on VAE selected conv op: ", op)


def add_attribute_by_name(op: ir.Operation, name: str, val: int):
    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)
    op.attributes[name] = attr


def shape_list_to_string(input):
    return "x".join([str(d) for d in input])


def create_context() -> ir.Context:
    context = ir.Context()
    ireec_trans.register_all_dialects(context)
    context.allow_unregistered_dialects = True
    return context


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    def path_expand(s):
        return Path(s).expanduser().resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=path_expand,
        default="model.mlir",
        help="Path to the input mlir file",
    )
    parser.add_argument(
        "-config_path",
        type=path_expand,
        default="best_configs.json",
        help="Path where stores the op config file",
    )
    parser.add_argument(
        "-output_path",
        type=path_expand,
        default="tuned_model.mlir",
        help="Path to save the annotated mlir file",
    )
    parser.add_argument(
        "-search_op",
        type=str,
        default="all",
        help="Op to be optimized. options are matmul, bmm, conv.",
    )

    args = parser.parse_args()

    with create_context() as ctx:
        module = model_annotation(
            ctx,
            input_contents=args.model,
            config_path=args.config_path,
            search_op=args.search_op,
        )
        mlir_str = str(module)
        with open(args.output_path, "w") as f:
            f.write(mlir_str)
        print(f"Saved mlir in {args.output_path}.")
