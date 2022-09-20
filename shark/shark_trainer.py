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

from shark.shark_importer import SharkImporter
from shark.parser import shark_args
from shark.shark_runner import SharkRunner
from shark.backward_makefx import MakeFxModule
import numpy as np
from tqdm import tqdm
import sys


# Prints to stderr.
def print_err(*a):
    print(*a, file=sys.stderr)


class SharkTrainer:
    """Training pytorch, tensorflow module on shark runtime."""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = True,
    ):
        self.model = model
        # Change tuple to list.
        self.input = [x for x in input]
        self.dynamic = dynamic
        self.from_aot = from_aot
        self.jit_trace = jit_trace
        self.from_aot = from_aot

        # By default it's the torch frontend.
        self.frontend = "pytorch"
        self.device = device if device is not None else shark_args.device

        self.shark_runner = None

    # Sets the frontend i.e `pytorch` or `tensorflow`.
    def set_frontend(self, frontend: str):
        if frontend not in [
            "pytorch",
            "torch",
            "tensorflow",
            "tf",
            "mhlo",
            "linalg",
            "tosa",
        ]:
            print_err("frontend not supported.")
        else:
            self.frontend = frontend

    # Training function is needed in the case of torch_fn.
    def compile(self, training_fn=None):
        if self.frontend in ["torch", "pytorch"]:
            aot_module = MakeFxModule(
                self.model, tuple(self.input), custom_inference_fn=training_fn
            )
            aot_module.generate_graph()
            # Returns the backward graph.
            training_graph = aot_module.training_graph
            weights = self.get_torch_params()
            mlir_importer = SharkImporter(
                training_graph, weights + self.input, "torch"
            )

            self.imported_mlir, func_name = mlir_importer.import_mlir(
                is_dynamic=self.dynamic, tracing_required=self.jit_trace
            )
            self.shark_runner = SharkRunner(
                self.imported_mlir, func_name, self.device, "tm_tensor"
            )
        elif self.frontend in ["tensorflow", "tf", "mhlo"]:
            self.shark_runner = SharkRunner(
                self.model,
                self.input,
                self.dynamic,
                self.device,
                self.jit_trace,
                self.from_aot,
                self.frontend,
            )
        else:
            print_err("Unknown frontend")
            return

    # The inputs to the mlir-graph are weights, buffers and inputs respectively.
    def get_torch_params(self):
        params = [i.detach() for i in self.model.parameters()]
        buffers = [i.detach() for i in self.model.buffers()]
        return params + buffers

    # Function to train pytorch module.
    def _train_torch(self, num_iters):
        """Returns the updated weights after num_iters"""
        params = self.get_torch_params()
        params = [x.numpy() for x in params]
        print(f"Training started for {num_iters} iterations:")
        for i in tqdm(range(num_iters)):
            params = self.shark_runner.forward(
                params + self.input, self.frontend
            )

        return params

    # Function to train tensorflow module.
    # Output final loss.
    # TODO(raikonenfnu): Save updated weight/states in SHARK.
    def _train_tf(self, num_iters):
        input_list = []
        for x in self.input:
            if isinstance(x, list):
                nested_list = []
                for val in x:
                    if isinstance(val, np.ndarray):
                        nested_list.append(val)
                    else:
                        nested_list.append(val.numpy())
                input_list.append(nested_list)
            elif isinstance(x, np.ndarray):
                input_list.append(x)
            else:
                input_list.append(x.numpy())

        print(f"Training started for {num_iters} iterations:")
        for i in tqdm(range(num_iters)):
            outputs = self.shark_runner.forward(input_list, self.frontend)
        return outputs

    def train(self, num_iters=1):
        if self.frontend in ["torch", "pytorch"]:
            return self._train_torch(num_iters)
        elif self.frontend in ["tf", "tensorflow", "mhlo"]:
            return self._train_tf(num_iters)
        else:
            print_err("Unknown frontend")
            return
