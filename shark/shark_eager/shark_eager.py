from typing import Any, Dict, List, Tuple
from collections import defaultdict
from shark.shark_importer import import_with_fx
import torchvision.models as models
import copy
import io
import numpy as np
import sys
import torch
import torch.fx
from torch.fx.node import Node
from typing import Dict
import torch_mlir


def shark_backend(fx_g: torch.fx.GraphModule, inputs, device: str = "cpu"):
    mlir_module = torch_mlir.compile(
        fx_g, inputs, output_type="linalg-on-tensors"
    )
    bytecode_stream = io.BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    from shark.shark_inference import SharkInference

    shark_module = SharkInference(
        mlir_module=bytecode,
        device=device,
        mlir_dialect="tm_tensor",
    )
    shark_module.compile(extra_args=[])
    return shark_module


def _make_single_op_gm(node, captured_val, compiled_graph):
    """Make a GraphModule that just executes the given node."""
    g = torch.fx.Graph()
    env = {}
    inputs = []
    for arg in node.args:
        if arg and hasattr(arg, "name"):
            env[arg.name] = g.placeholder(arg.name)
            if isinstance(captured_val[arg.name], (list, tuple)):
                for val in captured_val[arg.name]:
                    inputs.append(val)
            else:
                inputs.append(captured_val[arg.name])

    call = g.node_copy(node, lambda n: env[n.name])
    g.output(call)
    g.lint()
    single_node = torch.fx.GraphModule(torch.nn.Module(), g)
    compiled_module = shark_backend(single_node, inputs)
    compiled_graph[node.name] = {
        "module": compiled_module,
        "inputs": [i for i in env],
        "result": None,
    }
    return


def compiled_graph(gm: torch.fx.GraphModule, attr_info):
    compiled_graph = {}
    g = gm.graph
    for node in g.nodes:
        if node.op == "call_function":
            if not (
                node.target in [torch.ops.aten.empty]
                or node.name.startswith("getitem")
            ):
                _make_single_op_gm(node, attr_info, compiled_graph)

            # Currently torch.aten.empty has an compilation issue, so running natively.
            elif node.target in [torch.ops.aten.empty]:
                compiled_graph[node.name] = {
                    "target": node.target,
                    "args": node.args,
                    "kwargs": node.kwargs,
                    "result": None,
                }
            # Get item is a simple case takes a tuple and return the tensor at a particular index.
            elif node.name.startswith("getitem"):
                compiled_graph[node.name] = {
                    "input": node.args[0].name,
                    "pos": node.args[1],
                    "result": None,
                }

    return compiled_graph


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = fetch_attr(node.target)
            elif node.op == "call_function":
                result = node.target(
                    *load_arg(node.args), **load_arg(node.kwargs)
                )
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs)
                )

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return env

        # return load_arg(self.graph.result)


resnet18 = models.resnet18(pretrained=True)
resnet18.train(False)
input = (torch.randn(1, 3, 224, 224),)

print(resnet18(input[0]))

fx_graph = import_with_fx(resnet18, input, mlir_type="fx")

shape_prop = ShapeProp(fx_graph)

x = shape_prop.propagate(input[0])

shark_graph = compiled_graph(fx_graph, x)


for key in shark_graph:
    if key.startswith("getitem"):
        input_val = shark_graph[key]["input"]
        pos = shark_graph[key]["pos"]
        if input_val not in shark_graph:
            shark_graph[key]["result"] = x[input_val][pos].detach()
        else:
            shark_graph[key]["result"] = shark_graph[input_val]["result"][
                pos
            ].detach()
    elif key.startswith("empty"):
        operator = shark_graph[key]["target"]
        args = shark_graph[key]["args"]
        kwargs = shark_graph[key]["kwargs"]
        shark_graph[key]["result"] = operator(*args, **kwargs).detach()
    else:
        input_val = shark_graph[key]["inputs"]
        input_tensors = []
        for input in input_val:
            if input not in shark_graph:
                input_tensors.append(x[input].detach())
            else:
                input_tensors.append(shark_graph[input]["result"])

        val = shark_graph[key]["module"]("forward", input_tensors)
        if isinstance(val, (tuple, list)):
            list_val = []
            for v in val:
                list_val.append(torch.from_numpy(v))
            shark_graph[key]["result"] = list_val
        else:
            shark_graph[key]["result"] = torch.from_numpy(val)


print(shark_graph)
