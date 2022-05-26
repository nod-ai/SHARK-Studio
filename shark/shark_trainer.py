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

        self.weights = [i.detach() for i in model.parameters()]
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
