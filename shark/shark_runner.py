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
from shark.iree_utils import get_results, get_iree_compiled_module
import os
from shark.parser import shark_args
from tqdm import tqdm
import time


class SharkRunner:
    """Base class for Shark Inference and Shark Runner."""

    def __init__(
        self,
        iree_compilation_module,
        iree_config,
    ):

        self.iree_compilation_module = iree_compilation_module
        self.iree_config = iree_config

    # All the timings and benchmarking can be done here.
    def forward(self, input, frontend):
        return get_results(self.iree_compilation_module, input,
                           self.iree_config, frontend)


class SharkMode:

    def __init__(self, device="cpu"):
        if device == "refbackend":
            torch_mlir_tensor.backend = EagerModeRefBackend()
        else:
            torch_mlir_tensor.backend = EagerModeIREELinalgOnTensorsBackend(
                device)
        self.guard = enable_torch_dispatch_mode(TorchMLIRTensor)
        self.guard.__enter__()

    def __del__(self):
        self.guard.__exit__(None, None, None)
