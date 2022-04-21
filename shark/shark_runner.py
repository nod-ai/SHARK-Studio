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
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch_mlir.eager_mode import torch_mlir_tensor
from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor
from torch_mlir_e2e_test.eager_backends.refbackend import EagerModeRefBackend

from shark.iree_eager_backend import EagerModeIREELinalgOnTensorsBackend
from shark.torch_mlir_utils import get_torch_mlir_module, export_module_to_mlir_file, run_on_refbackend
from shark.iree_utils import get_results, get_iree_compiled_module, export_iree_module_to_vmfb
import os
from shark.parser import shark_args
from shark.backward_makefx import MakeFxModule
from tqdm import tqdm
import time


class SharkRunner:
    """Base class for Shark Inference and Shark Runner."""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool,
        device: str,
        tracing_required: bool,
        from_aot: bool,
    ):
        self.torch_module = model
        self.input = input
        self.torch_mlir_module = get_torch_mlir_module(model, input, dynamic,
                                                       tracing_required,
                                                       from_aot)

        if shark_args.save_mlir:
            export_module_to_mlir_file(self.torch_mlir_module,
                                       shark_args.repro_dir)
        if shark_args.save_vmfb:
            export_iree_module_to_vmfb(self.torch_mlir_module, device,
                                       shark_args.repro_dir)
        (
            self.iree_compilation_module,
            self.iree_config,
        ) = get_iree_compiled_module(self.torch_mlir_module, device)

    def forward(self, input):
        return get_results(self.iree_compilation_module, input,
                           self.iree_config)

    # Refbackend is used only for debugging purpose. It can be quite slow.
    def run_on_refbackend(self, input):
        return run_on_refbackend(self.torch_mlir_module, input)


class SharkInference:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = False,
        custom_inference_fn=None,
    ):
        self.model = model
        self.input = input
        self.from_aot = from_aot

        self.device = device if device is not None else shark_args.device

        self.shark_runner = SharkRunner(self.model, self.input, dynamic,
                                        self.device, jit_trace, from_aot)

    def benchmark_forward(self, inputs):
        inputs = self.input if self.from_aot else inputs
        input_list = [x.detach().numpy() for x in inputs]
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
        inputs = self.input if self.from_aot else inputs
        input_list = [x.detach().numpy() for x in inputs]
        return self.shark_runner.forward(input_list)

    def run_on_refbackend(self, inputs):
        self.shark_runner.run_on_refbackend(inputs)


class SharkTrainer:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = True,
        custom_inference_fn=None,
    ):
        self.model = model
        self.input = []
        self.from_aot = from_aot

        self.device = device if device is not None else shark_args.device

        aot_module = MakeFxModule(model,
                                  input,
                                  custom_inference_fn=custom_inference_fn)
        aot_module.generate_graph()
        self.model = aot_module.backward_graph

        self.weights = [
            i.detach() for i in model.parameters()
        ]
        for i in model.buffers():
            self.weights.append(i.detach())

        for i in input:
            self.input.append(i)

        self.shark_runner = SharkRunner(self.model, self.weights + self.input,
                                        dynamic, self.device, jit_trace,
                                        from_aot)

    def train(self, num_iters=1):
        """Returns the updated weights after num_iters"""
        weights = [x.detach().numpy() for x in self.weights]
        inputs = [x.detach().numpy() for x in self.input]

        print(f"Training started for {num_iters} iterations:")
        for i in tqdm(range(num_iters)):
            weights = self.shark_runner.forward(weights + inputs)

        return weights


class SharkMode:
    def __init__(self, device="cpu"):
        if device == "refbackend":
            torch_mlir_tensor.backend = EagerModeRefBackend()
        else:
            torch_mlir_tensor.backend = EagerModeIREELinalgOnTensorsBackend(device)
        self.guard = enable_torch_dispatch_mode(TorchMLIRTensor)
        self.guard.__enter__()

    def __del__(self):
        self.guard.__exit__(None, None, None)
