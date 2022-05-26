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
from shark.torch_mlir_utils import get_torch_mlir_module, run_on_refbackend
from shark.iree_utils import get_results, get_iree_compiled_module
import os
from shark.parser import shark_args
from shark.shark_runner import SharkRunner
from tqdm import tqdm
import time


class SharkInference:
    """Inference API targeting pytorch, tensorflow, linalg, mhlo and tosa frontend."""

    def __init__(
        self,
        model,
        input: tuple,
        device: str = None,
        dynamic: bool = False,
        jit_trace: bool = False,
    ):
        self.model = model
        self.input = input
        self.dynamic = dynamic
        self.jit_trace = jit_trace

        # By default it's torch frontend.
        self.frontend = "pytorch"

        # Sets the device.
        self.device = device if device is not None else shark_args.device

        self.shark_runner = None

    # Sets the frontend i.e `pytorch` `tensorflow`, `linalg`, `mhlo`, `tosa`.
    def set_frontend(self, frontend: str):
        self.frontend = frontend

    def compile(self):
        if self.frontend in ["pytorch", "torch"]:
            self.model = get_torch_mlir_module(self.model, self.input,
                                               self.dynamic, self.jit_trace)

        iree_compilation_module, iree_config = get_iree_compiled_module(
            self.model, self.device, self.frontend)

        self.shark_runner = SharkRunner(iree_compilation_module, iree_config)

    # inputs are considered to be np.array.
    def forward(self, inputs):
        input_list = inputs
        # converts the inputs to numpy.
        if self.frontend in ["pytorch", "torch"]:
            input_list = [x.detach().numpy() for x in inputs]
        elif self.frontend in ["tensorflow", "tf"]:
            input_list = [x.numpy() for x in inputs]
        return self.shark_runner.forward(input_list, self.frontend)
