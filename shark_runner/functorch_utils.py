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
from functorch.compile import (
    aot_module,
    min_cut_rematerialization_partition,
    memory_efficient_fusion
)
from torch_mlir_utils import get_torch_mlir_module
from torch import optim, fx
from typing import List
import copy


class AOTModule:
    def __init__(self, model, inputs, labels=None):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.forward_graph = None
        self.backward_graph = None
        self.forward_inputs = None
        self.backward_inputs = None

    def inference(self, model, inputs):
        iters = 1
        with torch.no_grad():
            for _ in range(iters):
                out = model(*inputs)

    def train(self, model, inputs, labels):
        # TODO: Pass the criterion and optimizer.
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        iters = 1
        for _ in range(iters):
            optimizer.zero_grad()
            output = model(*inputs)
            loss = criterion(output, *labels)
            loss.backward()
            optimizer.step()

    def change_fx_graph_return_to_tuple(self, fx_g: fx.GraphModule):
        for node in fx_g.graph.nodes:
            if node.op == "output":
                # output nodes always have one argument
                node_arg = node.args[0]
                if isinstance(node_arg, list):
                    # If there is a single tensor/element to be returned don't
                    # a tuple for it.
                    if(len(node_arg) == 1):
                        node.args = node_arg
                    else:
                        node.args = (tuple(node_arg),)
        fx_g.graph.lint()
        fx_g.recompile()
        return fx_g

    def get_forward_graph(self, fx_g: fx.GraphModule, inps):
        fx_g = self.change_fx_graph_return_to_tuple(fx_g)
        f = torch.jit.script(fx_g)
        f = torch.jit.freeze(f.eval())
        torch.jit.save(f, "forw.pt")
        f = torch.jit.load("forw.pt")
        self.forward_graph = f
        self.forward_inputs = copy.deepcopy(inps)
        return f

    def get_backward_graph(self, fx_g: fx.GraphModule, inps):
        fx_g = self.change_fx_graph_return_to_tuple(fx_g)
        f = torch.jit.script(fx_g)
        f = torch.jit.freeze(f.eval())
        torch.jit.save(f, "back.pt")
        f = torch.jit.load("back.pt")
        self.backward_graph = f
        self.backward_inputs = copy.deepcopy(inps)
        return f

    def generate_inference_graph(self):
        aot_model = memory_efficient_fusion(
            self.model,
            fw_compiler=self.get_forward_graph,
            bw_compiler=self.get_backward_graph,
            # partition_fn=min_cut_rematerialization_partition,
        )
        self.inference(aot_model, self.inputs)

    def generate_training_graph(self):
        aot_model = memory_efficient_fusion(
            self.model,
            fw_compiler=self.get_forward_graph,
            bw_compiler=self.get_backward_graph,
            # partition_fn=min_cut_rematerialization_partition,
        )
        self.train(aot_model, self.inputs, self.labels)
