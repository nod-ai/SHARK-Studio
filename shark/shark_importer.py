# Lint as: python3
"""SHARK Importer"""

import sys
import tempfile
import os
import hashlib


def create_hash(file_name):
    with open(file_name, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=64)
        while chunk := f.read(2**10):
            file_hash.update(chunk)

    return file_hash.hexdigest()


# List of the supported frontends.
supported_frontends = {
    "tensorflow",
    "tf",
    "pytorch",
    "torch",
    "tf-lite",
    "tflite",
}


class SharkImporter:
    """
    SharkImporter converts frontend modules into a
    mlir_module. The supported frameworks are tensorflow,
    pytorch, and tf-lite.

    ...

    Attributes
    ----------
    module :
        torch, tensorflow or tf-lite module.
    inputs :
        inputs to the module, may be required for the shape
        information.
    frontend: str
        frontend to which the module belongs.
    raw_model_file: str
        temp tflite model path

    Methods
    -------
    import_mlir(is_dynamic, tracing_required, func_name):
        is_dynamic: input shapes to be totally dynamic (pytorch specific).
        tracing_required: whether tracing is required (pytorch specific.
        func_name: The function to be traced out or imported to mlir.

    import_debug(is_dynamic, tracing_required, func_name):
        returns the converted (mlir_module,func_name) with inputs and golden
        outputs.
        The inputs and outputs are converted into np array.
    """

    def __init__(
        self,
        module,
        inputs: tuple = (),
        frontend: str = "torch",
        raw_model_file: str = "",
        return_str: bool = False,
    ):
        self.module = module
        self.inputs = None if len(inputs) == 0 else inputs
        self.frontend = frontend
        if not self.frontend in supported_frontends:
            print(
                f"The frontend is not in the supported_frontends: {supported_frontends}"
            )
            sys.exit(1)
        self.raw_model_file = raw_model_file
        self.return_str = return_str

    # NOTE: The default function for torch is "forward" and tf-lite is "main".

    def _torch_mlir(self, is_dynamic, tracing_required, mlir_type):
        from shark.torch_mlir_utils import get_torch_mlir_module

        return get_torch_mlir_module(
            self.module,
            self.inputs,
            is_dynamic,
            tracing_required,
            self.return_str,
            mlir_type,
        )

    def _tf_mlir(self, func_name, save_dir="."):
        from iree.compiler import tf as tfc

        return tfc.compile_module(
            self.module,
            exported_names=[func_name],
            import_only=True,
            output_file=save_dir,
        )

    def _tflite_mlir(self, func_name, save_dir="."):
        from iree.compiler import tflite as tflitec

        self.mlir_model = tflitec.compile_file(
            self.raw_model_file,  # in tflite, it is a path to .tflite file, not a tflite interpreter
            input_type="tosa",
            import_only=True,
            output_file=save_dir,
        )
        return self.mlir_model

    # Adds the conversion of the frontend with the private function.
    def import_mlir(
        self,
        is_dynamic=False,
        tracing_required=False,
        func_name="forward",
        save_dir="./shark_tmp/",
        mlir_type="linalg",
    ):
        if self.frontend in ["torch", "pytorch"]:
            if self.inputs == None:
                print(
                    "Please pass in the inputs, the inputs are required to determine the shape of the mlir_module"
                )
                sys.exit(1)
            return (
                self._torch_mlir(is_dynamic, tracing_required, mlir_type),
                func_name,
            )
        if self.frontend in ["tf", "tensorflow"]:
            return self._tf_mlir(func_name, save_dir), func_name
        if self.frontend in ["tflite", "tf-lite"]:
            func_name = "main"
            return self._tflite_mlir(func_name, save_dir), func_name

    # Converts the frontend specific tensors into np array.
    def convert_to_numpy(self, array_tuple: tuple):
        if self.frontend in ["torch", "pytorch"]:
            return [x.detach().cpu().numpy() for x in array_tuple]
        if self.frontend in ["tf", "tensorflow"]:
            return [x.numpy() for x in array_tuple]

    # Saves `function_name.npy`, `inputs.npz`, `golden_out.npz` and `model_name.mlir` in the directory `dir`.
    def save_data(
        self,
        dir,
        model_name,
        mlir_data,
        func_name,
        inputs,
        outputs,
        mlir_type="linalg",
    ):
        import numpy as np

        inputs_name = "inputs.npz"
        outputs_name = "golden_out.npz"
        func_file_name = "function_name"
        model_name_mlir = (
            model_name + "_" + self.frontend + "_" + mlir_type + ".mlir"
        )
        print(f"saving {model_name_mlir} to {dir}")
        try:
            inputs = [x.cpu().detach() for x in inputs]
        except AttributeError:
            try:
                inputs = [x.numpy() for x in inputs]
            except AttributeError:
                inputs = [x for x in inputs]
        np.savez(os.path.join(dir, inputs_name), *inputs)
        np.savez(os.path.join(dir, outputs_name), *outputs)
        np.save(os.path.join(dir, func_file_name), np.array(func_name))
        if self.frontend == "torch":
            with open(os.path.join(dir, model_name_mlir), "wb") as mlir_file:
                mlir_file.write(mlir_data)
        hash_gen_attempts = 2
        for i in range(hash_gen_attempts):
            try:
                mlir_hash = create_hash(os.path.join(dir, model_name_mlir))
            except FileNotFoundError as err:
                if i < hash_gen_attempts:
                    continue
                else:
                    raise err

        np.save(os.path.join(dir, "hash"), np.array(mlir_hash))
        return

    def import_debug(
        self,
        is_dynamic=False,
        tracing_required=False,
        func_name="forward",
        dir=tempfile.gettempdir(),
        model_name="model",
        golden_values=None,
        mlir_type="linalg",
    ):
        if self.inputs == None:
            print(
                f"There is no input provided: {self.inputs}, please provide inputs or simply run import_mlir."
            )
            sys.exit(1)
        model_name_mlir = (
            model_name + "_" + self.frontend + "_" + mlir_type + ".mlir"
        )
        artifact_path = os.path.join(dir, model_name_mlir)
        imported_mlir = self.import_mlir(
            is_dynamic,
            tracing_required,
            func_name,
            save_dir=artifact_path,
            mlir_type=mlir_type,
        )
        # TODO: Make sure that any generic function name is accepted. Currently takes in the default function names.
        # TODO: Check for multiple outputs.
        if self.frontend in ["torch", "pytorch"]:
            import torch

            golden_out = None
            if golden_values is not None:
                golden_out = golden_values
            else:
                golden_out = self.module(*self.inputs)
            if torch.is_tensor(golden_out):
                golden_out = tuple(
                    golden_out.detach().cpu().numpy(),
                )
            else:
                golden_out = self.convert_to_numpy(golden_out)
            # Save the artifacts in the directory dir.
            self.save_data(
                dir,
                model_name,
                imported_mlir[0],
                imported_mlir[1],
                self.inputs,
                golden_out,
                mlir_type,
            )
            return (
                imported_mlir,
                self.convert_to_numpy(self.inputs),
                golden_out,
            )
        if self.frontend in ["tf", "tensorflow"]:
            import tensorflow as tf

            golden_out = self.module.forward(*self.inputs)
            if tf.is_tensor(golden_out):
                golden_out = tuple(
                    golden_out.numpy(),
                )
            elif golden_out is tuple:
                golden_out = self.convert_to_numpy(golden_out)
            elif hasattr(golden_out, "logits"):
                # from transformers import TFSequenceClassifierOutput
                golden_out = golden_out.logits
            else:
                golden_out = golden_out.last_hidden_state
            # Save the artifacts in the directory dir.
            self.save_data(
                dir,
                model_name,
                imported_mlir[0],
                imported_mlir[1],
                self.inputs,
                golden_out,
            )
            return (
                imported_mlir,
                self.convert_to_numpy(self.inputs),
                golden_out,
            )
        if self.frontend in ["tflite", "tf-lite"]:
            # TODO(Chi): Validate it for tflite models.
            golden_out = self.module.invoke_tflite(self.inputs)
            self.save_data(
                dir,
                model_name,
                imported_mlir[0],
                imported_mlir[1],
                self.inputs,
                golden_out,
            )
            return (
                imported_mlir,
                self.inputs,
                golden_out,
            )


def get_f16_inputs(inputs, is_f16, f16_input_mask):
    if is_f16 == False:
        return inputs
    if f16_input_mask == None:
        return tuple([x.half() for x in inputs])

    f16_masked_inputs = []
    for i in range(len(inputs)):
        if f16_input_mask[i]:
            f16_masked_inputs.append(inputs[i].half())
        else:
            f16_masked_inputs.append(inputs[i])

    return tuple(f16_masked_inputs)


# Upcasts the block/list of ops.
def add_upcast(fx_g):
    import torch

    for node in fx_g.graph.nodes:
        if node.target in [torch.ops.aten.mul]:
            # This is a very strict check.
            if hasattr(node.args[1], "target"):
                if (
                    node.args[1].target in [torch.ops.aten.rsqrt]
                    and node.args[1].args[0].target in [torch.ops.aten.add]
                    and node.args[1].args[0].args[0].target
                    in [torch.ops.aten.mean]
                    and node.args[1].args[0].args[0].args[0].target
                    in [torch.ops.aten.pow]
                ):
                    print("found an upcasting block let's upcast it.")
                    pow_node = node.args[1].args[0].args[0].args[0]
                    mul_node = node
                    with fx_g.graph.inserting_before(pow_node):
                        lhs = pow_node.args[0]
                        upcast_lhs = fx_g.graph.call_function(
                            torch.ops.aten._to_copy,
                            args=(lhs,),
                            kwargs={"dtype": torch.float32},
                        )
                        pow_node.args = (upcast_lhs, pow_node.args[1])
                    with fx_g.graph.inserting_before(mul_node):
                        new_node = fx_g.graph.call_function(
                            torch.ops.aten._to_copy,
                            args=(mul_node,),
                            kwargs={"dtype": torch.float16},
                        )
                        mul_node.append(new_node)
                        mul_node.replace_all_uses_with(new_node)
                        new_node.args = (mul_node,)
                        new_node.kwargs = {"dtype": torch.float16}

    fx_g.graph.lint()


def transform_fx(fx_g, quantized=False):
    import torch

    kwargs_dict = {
        "dtype": torch.float16,
        "device": torch.device(type="cpu"),
        "pin_memory": False,
    }
    kwargs_dict1 = {
        "dtype": torch.float16,
    }
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            # aten.empty should be filled with zeros.
            if node.target in [torch.ops.aten.empty]:
                with fx_g.graph.inserting_after(node):
                    new_node = fx_g.graph.call_function(
                        torch.ops.aten.zero_,
                        args=(node,),
                    )
                    node.append(new_node)
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node,)
            if quantized:
                continue

            if node.target in [
                torch.ops.aten.arange,
                torch.ops.aten.empty,
                torch.ops.aten.zeros,
                torch.ops.aten.zeros_like,
            ]:
                if node.kwargs.get("dtype") == torch.float32:
                    node.kwargs = kwargs_dict

            # Vicuna
            if node.target in [
                torch.ops.aten._to_copy,
            ]:
                if node.kwargs.get("dtype") == torch.float32:
                    node.kwargs = kwargs_dict1

            if node.target in [
                torch.ops.aten.masked_fill,
            ]:
                if node.args[2] > torch.finfo(torch.half).max:
                    max_val = torch.finfo(torch.half).max
                    node.args = (node.args[0], node.args[1], max_val)
                elif node.args[2] < torch.finfo(torch.half).min:
                    min_val = torch.finfo(torch.half).min
                    node.args = (node.args[0], node.args[1], min_val)

            if node.target in [
                torch.ops.aten.full,
            ]:
                if node.args[1] > torch.finfo(torch.half).max:
                    max_val = torch.finfo(torch.half).max
                    node.args = (node.args[0], max_val)
                    node.kwargs = kwargs_dict
                elif node.args[1] < torch.finfo(torch.half).min:
                    min_val = torch.finfo(torch.half).min
                    node.args = (node.args[0], min_val)
                    node.kwargs = kwargs_dict

            # Inputs and outputs of aten.var.mean should be upcasted to fp32.
            if node.target in [torch.ops.aten.var_mean]:
                with fx_g.graph.inserting_before(node):
                    new_node = fx_g.graph.call_function(
                        torch.ops.prims.convert_element_type,
                        args=(node.args[0], torch.float32),
                        kwargs={},
                    )
                    node.args = (new_node, node.args[1])

            if node.name.startswith("getitem"):
                with fx_g.graph.inserting_before(node):
                    if node.args[0].target in [torch.ops.aten.var_mean]:
                        new_node = fx_g.graph.call_function(
                            torch.ops.aten._to_copy,
                            args=(node,),
                            kwargs={"dtype": torch.float16},
                        )
                        node.append(new_node)
                        node.replace_all_uses_with(new_node)
                        new_node.args = (node,)
                        new_node.kwargs = {"dtype": torch.float16}

    # Required for cuda debugging.
    # for node in fx_g.graph.nodes:
    # if node.op == "call_function":
    # if node.kwargs.get("device") == torch.device(type="cpu"):
    # new_kwargs = node.kwargs.copy()
    # new_kwargs["device"] = torch.device(type="cuda")
    # node.kwargs = new_kwargs

    fx_g.graph.lint()


def gptq_transforms(fx_g):
    import torch

    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            if node.target in [
                torch.ops.aten.arange,
                torch.ops.aten.empty,
                torch.ops.aten.ones,
                torch.ops.aten._to_copy,
            ]:
                if node.kwargs.get("device") == torch.device(device="cuda:0"):
                    updated_kwargs = node.kwargs.copy()
                    updated_kwargs["device"] = torch.device(device="cpu")
                    node.kwargs = updated_kwargs

            if node.target in [
                torch.ops.aten._to_copy,
            ]:
                if node.kwargs.get("dtype") == torch.bfloat16:
                    updated_kwargs = node.kwargs.copy()
                    updated_kwargs["dtype"] = torch.float16
                    node.kwargs = updated_kwargs

            # Inputs of aten.native_layer_norm should be upcasted to fp32.
            if node.target in [torch.ops.aten.native_layer_norm]:
                with fx_g.graph.inserting_before(node):
                    new_node_arg0 = fx_g.graph.call_function(
                        torch.ops.prims.convert_element_type,
                        args=(node.args[0], torch.float32),
                        kwargs={},
                    )
                    node.args = (
                        new_node_arg0,
                        node.args[1],
                        node.args[2],
                        node.args[3],
                        node.args[4],
                    )

            # Downcasting the result of native_layer_norm back to fp16.
            if node.name.startswith("getitem"):
                with fx_g.graph.inserting_before(node):
                    if node.args[0].target in [
                        torch.ops.aten.native_layer_norm
                    ]:
                        new_node = fx_g.graph.call_function(
                            torch.ops.aten._to_copy,
                            args=(node,),
                            kwargs={"dtype": torch.float16},
                        )
                        node.append(new_node)
                        node.replace_all_uses_with(new_node)
                        new_node.args = (node,)
                        new_node.kwargs = {"dtype": torch.float16}

            # Inputs and outputs of aten.mm should be upcasted to fp32.
            if node.target in [torch.ops.aten.mm]:
                with fx_g.graph.inserting_before(node):
                    new_node_arg0 = fx_g.graph.call_function(
                        torch.ops.prims.convert_element_type,
                        args=(node.args[0], torch.float32),
                        kwargs={},
                    )
                    new_node_arg1 = fx_g.graph.call_function(
                        torch.ops.prims.convert_element_type,
                        args=(node.args[1], torch.float32),
                        kwargs={},
                    )
                    node.args = (new_node_arg0, new_node_arg1)

            if type(node.args[0]) == torch.fx.node.Node and node.args[
                0
            ].target in [torch.ops.aten.mm]:
                with fx_g.graph.inserting_before(node):
                    tmp = node.args[0]
                    new_node = fx_g.graph.call_function(
                        torch.ops.aten._to_copy,
                        args=(node.args[0],),
                        kwargs={"dtype": torch.float16},
                    )
                    node.args[0].append(new_node)
                    node.args[0].replace_all_uses_with(new_node)
                    new_node.args = (tmp,)
                    new_node.kwargs = {"dtype": torch.float16}

            # Inputs and outputs of aten._softmax should be upcasted to fp32.
            if node.target in [torch.ops.aten._softmax]:
                with fx_g.graph.inserting_before(node):
                    new_node_arg0 = fx_g.graph.call_function(
                        torch.ops.prims.convert_element_type,
                        args=(node.args[0], torch.float32),
                        kwargs={},
                    )
                    node.args = (new_node_arg0, node.args[1], node.args[2])

            if (
                type(node.args[0]) == torch.fx.node.Node
                and node.args[0].target in [torch.ops.aten._softmax]
                and node.target in [torch.ops.aten.expand]
            ):
                with fx_g.graph.inserting_before(node):
                    tmp = node.args[0]
                    new_node = fx_g.graph.call_function(
                        torch.ops.aten._to_copy,
                        args=(node.args[0],),
                        kwargs={"dtype": torch.float16},
                    )
                    node.args[0].append(new_node)
                    node.args[0].replace_all_uses_with(new_node)
                    new_node.args = (tmp,)
                    new_node.kwargs = {"dtype": torch.float16}

    fx_g.graph.lint()


# Doesn't replace the None type.
def change_fx_graph_return_to_tuple(fx_g):
    for node in fx_g.graph.nodes:
        if node.op == "output":
            # output nodes always have one argument
            node_arg = node.args[0]
            out_nodes = []
            if isinstance(node_arg, list):
                # Don't return NoneType elements.
                for out_node in node_arg:
                    if not isinstance(out_node, type(None)):
                        out_nodes.append(out_node)
                # If there is a single tensor/element to be returned don't
                # a tuple for it.
                if len(out_nodes) == 1:
                    node.args = out_nodes
                else:
                    node.args = (tuple(out_nodes),)
    fx_g.graph.lint()
    fx_g.recompile()
    return fx_g


def flatten_training_input(inputs):
    flattened_input = []
    for i in inputs:
        if isinstance(i, dict):
            for value in i.values():
                flattened_input.append(value.detach())
        elif isinstance(i, tuple):
            for value in i:
                flattened_input.append(value)
        else:
            flattened_input.append(i)
    return tuple(flattened_input)


# TODO: Remove is_f16 and fix all calls with using precision instead
# Applies fx conversion to the model and imports the mlir.
def import_with_fx(
    model,
    inputs,
    is_f16=False,
    f16_input_mask=None,
    debug=False,
    training=False,
    return_str=False,
    save_dir=tempfile.gettempdir(),
    model_name="model",
    mlir_type="linalg",
    is_dynamic=False,
    tracing_required=False,
    precision="fp32",
    is_gptq=False,
):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._decomp import get_decompositions
    from typing import List

    golden_values = None
    if debug:
        try:
            golden_values = model(*inputs)
        except:
            golden_values = None

    def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
        removed_indexes = []
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, (list, tuple)):
                    node_arg = list(node_arg)
                    node_args_len = len(node_arg)
                    for i in range(node_args_len):
                        curr_index = node_args_len - (i + 1)
                        if node_arg[curr_index] is None:
                            removed_indexes.append(curr_index)
                            node_arg.pop(curr_index)
                    node.args = (tuple(node_arg),)
                    break

        if len(removed_indexes) > 0:
            fx_g.graph.lint()
            fx_g.graph.eliminate_dead_code()
            fx_g.recompile()
        removed_indexes.sort()
        return removed_indexes

    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
        """
        Replace tuple with tuple element in functions that return one-element tuples.
        Returns true if an unwrapping took place, and false otherwise.
        """
        unwrapped_tuple = False
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    if len(node_arg) == 1:
                        node.args = (node_arg[0],)
                        unwrapped_tuple = True
                        break

        if unwrapped_tuple:
            fx_g.graph.lint()
            fx_g.recompile()
        return unwrapped_tuple

    # TODO: Control the decompositions.
    decomps_list = [
        torch.ops.aten.embedding_dense_backward,
        torch.ops.aten.native_layer_norm_backward,
        torch.ops.aten.slice_backward,
        torch.ops.aten.select_backward,
        torch.ops.aten.norm.ScalarOpt_dim,
        torch.ops.aten.native_group_norm,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes,
        torch.ops.aten.native_layer_norm,
        torch.ops.aten.masked_fill.Tensor,
        torch.ops.aten.masked_fill.Scalar,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten.index_add,
        torch.ops.aten.index_add_,
    ]
    if precision in ["int4", "int8"] and not is_gptq:
        from brevitas_examples.llm.llm_quant.export import (
            block_quant_layer_level_manager,
        )
        from brevitas_examples.llm.llm_quant.export import (
            brevitas_layer_export_mode,
        )
        from brevitas_examples.llm.llm_quant.sharded_mlir_group_export import (
            LinearWeightBlockQuantHandlerFwd,
        )
        from brevitas_examples.llm.llm_quant.export import (
            replace_call_fn_target,
        )
        from brevitas_examples.llm.llm_quant.sharded_mlir_group_export import (
            matmul_rhs_group_quant_placeholder,
        )
        from brevitas.backport.fx.experimental.proxy_tensor import (
            make_fx as brevitas_make_fx,
        )

        export_context_manager = brevitas_layer_export_mode
        export_class = block_quant_layer_level_manager(
            export_handlers=[LinearWeightBlockQuantHandlerFwd]
        )
        with export_context_manager(model, export_class):
            fx_g = brevitas_make_fx(
                model,
                decomposition_table=get_decompositions(decomps_list),
            )(*inputs)

        transform_fx(fx_g, quantized=True)
        replace_call_fn_target(
            fx_g,
            src=matmul_rhs_group_quant_placeholder,
            target=torch.ops.quant.matmul_rhs_group_quant,
        )

        fx_g.recompile()
        removed_none_indexes = _remove_nones(fx_g)
        was_unwrapped = _unwrap_single_tuple_return(fx_g)
    else:
        fx_g = make_fx(
            model,
            decomposition_table=get_decompositions(decomps_list),
        )(*inputs)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()

    strip_overloads(fx_g)

    if is_f16:
        fx_g = fx_g.half()
        transform_fx(fx_g)
        # TODO: Have to make it more generic.
        add_upcast(fx_g)
        fx_g.recompile()

    if is_gptq:
        gptq_transforms(fx_g)
        fx_g.recompile()

    if mlir_type == "fx":
        return fx_g

    if training:
        change_fx_graph_return_to_tuple(fx_g)
        inputs = flatten_training_input(inputs)

    ts_graph = torch.jit.script(fx_g)
    if mlir_type == "torchscript":
        return ts_graph

    inputs = get_f16_inputs(inputs, is_f16, f16_input_mask)
    mlir_importer = SharkImporter(
        ts_graph,
        inputs,
        frontend="torch",
        return_str=return_str,
    )

    if debug:  # and not is_f16:
        (mlir_module, func_name), _, _ = mlir_importer.import_debug(
            dir=save_dir,
            model_name=model_name,
            golden_values=golden_values,
            mlir_type=mlir_type,
            is_dynamic=is_dynamic,
            tracing_required=tracing_required,
        )
        return mlir_module, func_name

    mlir_module, func_name = mlir_importer.import_mlir(mlir_type=mlir_type)
    return mlir_module, func_name


# Saves a .mlir module python object to the directory 'dir' with 'model_name' and returns a path to the saved file.
def save_mlir(
    mlir_module,
    model_name,
    mlir_dialect="linalg",
    frontend="torch",
    dir=tempfile.gettempdir(),
):
    model_name_mlir = (
        model_name + "_" + frontend + "_" + mlir_dialect + ".mlir"
    )
    if dir == "":
        dir = tempfile.gettempdir()
    mlir_path = os.path.join(dir, model_name_mlir)
    print(f"saving {model_name_mlir} to {dir}")
    if frontend == "torch":
        with open(mlir_path, "wb") as mlir_file:
            mlir_file.write(mlir_module)

    return mlir_path
