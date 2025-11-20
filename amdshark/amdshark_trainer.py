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

from amdshark.parser import amdshark_args
from amdshark.amdshark_runner import AMDSharkRunner
from amdshark.backward_makefx import MakeFxModule
from amdshark.amdshark_importer import import_with_fx, save_mlir
import numpy as np
from tqdm import tqdm
import sys


# Prints to stderr.
def print_err(*a):
    print(*a, file=sys.stderr)


class AMDSharkTrainer:
    """Training pytorch, tensorflow module on amdshark runtime."""

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
        self.device = device if device is not None else amdshark_args.device

        self.amdshark_runner = None

    # Sets the frontend i.e `pytorch` or `tensorflow`.
    def set_frontend(self, frontend: str):
        if frontend not in [
            "pytorch",
            "torch",
            "tensorflow",
            "tf",
            "stablehlo",
            "mhlo",
            "linalg",
            "tosa",
        ]:
            print_err("frontend not supported.")
        else:
            self.frontend = frontend

    # Training function is needed in the case of torch_fn.
    def compile(self, training_fn=None, mlir_type="linalg", extra_args=[]):
        if self.frontend in ["torch", "pytorch"]:
            packed_inputs = (
                dict(self.model.named_parameters()),
                dict(self.model.named_buffers()),
                tuple(self.input),
            )
            mlir_module, func_name = import_with_fx(
                training_fn,
                packed_inputs,
                False,
                [],
                training=True,
                mlir_type=mlir_type,
            )
            mlir_module = save_mlir(
                mlir_module,
                model_name="amdshark_model",
                frontend="torch",
                mlir_dialect=mlir_type,
            )
            self.amdshark_runner = AMDSharkRunner(
                mlir_module,
                self.device,
                "tm_tensor",
                extra_args=extra_args,
            )
        elif self.frontend in ["tensorflow", "tf", "mhlo", "stablehlo"]:
            self.amdshark_runner = AMDSharkRunner(
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
            params = self.amdshark_runner.run(
                "forward", params + self.input, self.frontend
            )

        return params

    # Function to train tensorflow module.
    # Output final loss.
    # TODO(raikonenfnu): Save updated weight/states in AMDSHARK.
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
            outputs = self.amdshark_runner.forward(input_list, self.frontend)
        return outputs

    def train(self, num_iters=1):
        if self.frontend in ["torch", "pytorch"]:
            return self._train_torch(num_iters)
        elif self.frontend in ["tf", "tensorflow", "mhlo"]:
            return self._train_tf(num_iters)
        else:
            print_err("Unknown frontend")
            return
