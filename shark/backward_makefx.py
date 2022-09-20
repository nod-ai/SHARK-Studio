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
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from functorch._src.compile_utils import strip_overloads
from torch.nn.utils import _stateless

from torch import fx
import tempfile


class MakeFxModule:
    def __init__(self, model, inputs, labels=None, custom_inference_fn=None):
        self.model = model
        self.inputs = inputs
        self.custom_inference_fn = custom_inference_fn
        self.training_graph = None

    # Doesn't replace the None type.
    def change_fx_graph_return_to_tuple(self, fx_g: fx.GraphModule):
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

    def generate_graph(self):
        fx_g = make_fx(
            self.custom_inference_fn,
            decomposition_table=get_decompositions(
                [
                    torch.ops.aten.embedding_dense_backward,
                    torch.ops.aten.native_layer_norm_backward,
                    torch.ops.aten.slice_backward,
                    torch.ops.aten.select_backward,
                ]
            ),
        )(
            dict(self.model.named_parameters()),
            dict(self.model.named_buffers()),
            self.inputs,
        )
        fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
        fx_g.recompile()
        fx_g = self.change_fx_graph_return_to_tuple(fx_g)
        strip_overloads(fx_g)
        ts_g = torch.jit.script(fx_g)
        temp = tempfile.NamedTemporaryFile(
            suffix="_shark_ts", prefix="temp_ts_"
        )
        ts_g.save(temp.name)
        new_ts = torch.jit.load(temp.name)
        self.training_graph = new_ts
