import functools
from typing import List, Optional
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.compile_utils import strip_overloads
from shark.shark_inference import SharkInference
from torch._decomp import get_decompositions
from torch.func import functionalize
import io
import torch_mlir


# TODO: Control decompositions.
def default_decompositions():
    return get_decompositions(
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
            torch.ops.aten.masked_fill.Tensor,
            torch.ops.aten.masked_fill.Scalar,
        ]
    )


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


def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False


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


class SharkBackend:
    def __init__(
        self, fx_g: torch.fx.GraphModule, inputs: tuple, options: dict
    ):
        self.fx_g = fx_g
        self.inputs = inputs
        self.shark_module = None
        self.device: str = options.get("device", "cpu")
        self.was_unwrapped: bool = False
        self.none_indices: list = []
        self._modify_fx_g()
        self.compile()

    def _modify_fx_g(self):
        self.none_indices = _remove_nones(self.fx_g)
        self.was_unwrapped = _unwrap_single_tuple_return(self.fx_g)

    def compile(self):
        gm = make_fx(
            functionalize(self.fx_g),
            decomposition_table=default_decompositions(),
        )(*self.inputs)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())
        gm.recompile()
        strip_overloads(gm)
        ts_g = torch.jit.script(gm)
        mlir_module = torch_mlir.compile(
            ts_g, self.inputs, output_type="linalg-on-tensors"
        )
        bytecode_stream = io.BytesIO()
        mlir_module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()
        from shark.shark_inference import SharkInference

        shark_module = SharkInference(
            mlir_module=bytecode,
            device=self.device,
            mlir_dialect="tm_tensor",
        )
        shark_module.compile(extra_args=[])
        self.shark_module = shark_module

    def __call__(self, *inputs):
        np_inputs = [x.contiguous().detach().cpu().numpy() for x in inputs]
        np_outs = self.shark_module("forward", np_inputs)
        if self.was_unwrapped:
            np_outs = [
                np_outs,
            ]

        if not isinstance(np_outs, list):
            res = torch.from_numpy(np_outs)
            return res

        result = [torch.from_numpy(x) for x in np_outs]
        for r_in in self.none_indices:
            result.insert(r_in, None)
        result = tuple(result)
        return result
