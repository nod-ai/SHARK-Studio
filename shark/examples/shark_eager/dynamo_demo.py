import torch
import torchdynamo
from torch_mlir import compile, OutputType
from torchdynamo.optimizations.backends import create_backend
from torchdynamo.optimizations.subgraph import SubGraph

from shark.iree_utils import get_iree_compiled_module

NUM_ITERS = 10


def __torch_mlir(fx_graph, *args, **kwargs):
    assert isinstance(
        fx_graph, torch.fx.GraphModule
    ), "Model must be an FX GraphModule."

    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule):
        """Replace tuple with tuple element in functions that return one-element tuples."""

        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert len(node.args) == 1, "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple) and len(node_arg) == 1:
                    node.args = (node_arg[0],)
        fx_g.graph.lint()
        fx_g.recompile()
        return fx_g

    fx_graph = _unwrap_single_tuple_return(fx_graph)
    ts_graph = torch.jit.script(fx_graph)

    if isinstance(args, tuple):
        args = list(args)
    assert isinstance(args, list)
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    linalg_module = compile(ts_graph, args, output_type=OutputType.LINALG_ON_TENSORS)
    callable, _ = get_iree_compiled_module(linalg_module, "cuda", func_name="forward")

    def forward(*inputs):
        return callable(*inputs)

    return forward


def toy_example(*args):
    a, b = args

    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


with torchdynamo.optimize(__torch_mlir):
    for _ in range(10):
        print(toy_example(torch.randn(10), torch.randn(10)))


@create_backend
def torch_mlir(subgraph, *args, **kwargs):
    assert isinstance(subgraph, SubGraph), "Model must be a dynamo SubGraph."
    return __torch_mlir(subgraph.model, *list(subgraph.example_inputs))


@torchdynamo.optimize("torch_mlir")
def toy_example2(*args):
    a, b = args

    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


for _ in range(10):
    print(toy_example2(torch.randn(10), torch.randn(10)))
