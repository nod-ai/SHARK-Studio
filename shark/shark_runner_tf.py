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

from shark.iree_utils_tf import get_iree_compiled_module_tf, export_tf_iree_module_to_vmfb, get_results_tf
import os
import numpy as np
from shark.parser import shark_args
from tqdm import tqdm
import time


class SharkRunnerTF:
    """Base class for Shark Inference and Shark Runner."""

    def __init__(
        self,
        model,
        input: tuple,
        device: str,
    ):
        self.tf_module = model
        self.input = input
        if shark_args.save_vmfb:
            export_tf_iree_module_to_vmfb(self.tf_module,
                                              shark_args.repro_dir)
        (
            self.iree_compilation_module,
            self.iree_config,
        ) = get_iree_compiled_module_tf(self.tf_module, device)

    def forward(self, input):
        return get_results_tf(self.iree_compilation_module, self.input,
                           self.iree_config)


class SharkInferenceTF:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        device: str = None,
    ):
        self.model = model
        self.input = input

        self.device = device if device is not None else shark_args.device

        self.shark_runner = SharkRunnerTF(self.model, self.input, self.device)

    def benchmark_forward(self, inputs):
        inputs = inputs
        input_list = [x.numpy() for x in inputs]
        for i in range(shark_args.num_warmup_iterations):
            self.shark_runner.forward(input_list)

        for i in range(shark_args.num_iterations):
            begin = time.time()
            out = self.shark_runner.forward(input_list)
            end = time.time()
            print("Iteration " + str(i) + ": " + str(end - begin))
            if i == shark_args.num_iterations - 1:
                return out

    def forward(self, inputs):
        # TODO Capture weights and inputs in case of AOT, Also rework the
        # forward pass.
        input_list = [x.numpy() for x in inputs]
        return self.shark_runner.forward(input_list)


class SharkTrainerTF:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        device: str = None,
    ):
        self.model = model
        self.input = input
        self.device = device if device is not None else shark_args.device
        self.shark_runner = SharkRunnerTF(self.model, self.input, self.device)

    def train(self, num_iters=1):
        input_list = []
        for x in self.input:
            if (isinstance(x, list)):
                for val in x:
                    if (isinstance(val, np.ndarray)):
                        input_list.append([val for val in x])
                    else:
                        input_list.append([val.numpy() for val in x])
            elif (isinstance(x, np.ndarray)):
                input_list.append(x)
            else:
                input_list.append(x.numpy())

        print(f"Training started for {num_iters} iterations:")
        for i in tqdm(range(num_iters)):
            outputs = self.shark_runner.forward(input_list)

        return self.model.trainable_variables
