# Lint as: python3
"""SHARK Importer"""

import sys
import tempfile
import os

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

    def _torch_mlir(self, is_dynamic, tracing_required):
        from shark.torch_mlir_utils import get_torch_mlir_module

        return get_torch_mlir_module(
            self.module,
            self.inputs,
            is_dynamic,
            tracing_required,
            self.return_str,
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
    ):
        if self.frontend in ["torch", "pytorch"]:
            if self.inputs == None:
                print(
                    "Please pass in the inputs, the inputs are required to determine the shape of the mlir_module"
                )
                sys.exit(1)
            return self._torch_mlir(is_dynamic, tracing_required), func_name
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
        self, dir, model_name, mlir_data, func_name, inputs, outputs
    ):
        import numpy as np

        inputs_name = "inputs.npz"
        outputs_name = "golden_out.npz"
        func_file_name = "function_name"
        model_name_mlir = model_name + "_" + self.frontend + ".mlir"
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

        return

    def import_debug(
        self,
        is_dynamic=False,
        tracing_required=False,
        func_name="forward",
        dir=tempfile.gettempdir(),
        model_name="model",
        golden_values=None,
    ):
        if self.inputs == None:
            print(
                f"There is no input provided: {self.inputs}, please provide inputs or simply run import_mlir."
            )
            sys.exit(1)
        model_name_mlir = model_name + "_" + self.frontend + ".mlir"
        artifact_path = os.path.join(dir, model_name_mlir)
        imported_mlir = self.import_mlir(
            is_dynamic,
            tracing_required,
            func_name,
            save_dir=artifact_path,
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


def transform_fx(fx_g):
    import torch

    kwargs_dict = {
        "dtype": torch.float16,
        "device": torch.device(type="cpu"),
        "pin_memory": False,
    }
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            if node.target in [
                torch.ops.aten.arange,
                torch.ops.aten.empty,
            ]:
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
):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch._decomp import get_decompositions

    golden_values = None
    if debug:
        golden_values = model(*inputs)
    # TODO: Control the decompositions.
    fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
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
            ]
        ),
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
        fx_g.recompile()

    if training:
        change_fx_graph_return_to_tuple(fx_g)
        inputs = flatten_training_input(inputs)

    ts_graph = torch.jit.script(fx_g)
    inputs = get_f16_inputs(inputs, is_f16, f16_input_mask)
    mlir_importer = SharkImporter(
        ts_graph,
        inputs,
        frontend="torch",
        return_str=return_str,
    )

    if debug:  # and not is_f16:
        (mlir_module, func_name), _, _ = mlir_importer.import_debug(
            dir=save_dir, model_name=model_name, golden_values=golden_values
        )
        return mlir_module, func_name

    mlir_module, func_name = mlir_importer.import_mlir()
    return mlir_module, func_name
