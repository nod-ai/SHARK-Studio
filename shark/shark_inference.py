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
import os
from shark.parser import shark_args
from shark.shark_runner import SharkRunner, SharkBenchmarkRunner
import time
import sys


# Prints to stderr.
def print_err(*a):
    print(*a, file=sys.stderr)


class SharkInference:
    """Inference API targeting pytorch, tensorflow, linalg, mhlo and tosa frontend."""

    def __init__(self,
                 model,
                 input: tuple,
                 device: str = None,
                 dynamic: bool = False,
                 jit_trace: bool = False,
                 benchmark_mode: bool = False):
        self.model = model
        self.input = input
        self.dynamic = dynamic
        self.jit_trace = jit_trace
        self.benchmark_mode = benchmark_mode

        # By default it's torch frontend.
        self.frontend = "pytorch"

        # Sets the device.
        self.device = device if device is not None else shark_args.device

        self.model_config_path = shark_args.model_config_path

        self.shark_runner = None

    # Sets the frontend i.e `pytorch` or `tensorflow`.
    def set_frontend(self, frontend: str):
        if frontend not in [
                "pytorch", "torch", "tensorflow", "tf", "mhlo", "linalg",
                "tosa", "tflite"
        ]:
            print_err("frontend not supported.")
        else:
            self.frontend = frontend

    def compile(self):
        # Inference do not use AOT.
        from_aot = False
        if (self.benchmark_mode == True):
            self.shark_runner = SharkBenchmarkRunner(self.model, self.input,
                                                     self.dynamic, self.device,
                                                     self.jit_trace, from_aot,
                                                     self.frontend)
        else:
            self.shark_runner = SharkRunner(self.model, self.input,
                                            self.dynamic, self.device,
                                            self.jit_trace, from_aot,
                                            self.frontend,
                                            self.model_config_path)

    # inputs are considered to be np.array.
    def forward(self, inputs):
        input_list = inputs
        # converts the inputs to numpy.
        if self.frontend in ["pytorch", "torch"]:
            input_list = [x.detach().numpy() for x in inputs]
        elif self.frontend in ["tensorflow", "tf"]:
            input_list = [x.numpy() for x in inputs]
        return self.shark_runner.forward(input_list, self.frontend)

    # Saves the .vmfb module.
    def save_module(self, dir=None):
        if dir is None:
            return self.shark_runner.save_module()
        return self.shark_runner.save_module(dir)

    ######### Benchmark Related Functions #########
    def benchmark_mode(func):

        def inner(self, *args, **kwargs):
            assert self.benchmark_mode, "SharkRunner needs to be in benchmark mode to run benchmark methods."
            return func(self, *args, **kwargs)

        return inner

    @benchmark_mode
    def benchmark_all(self, inputs):
        self.shark_runner.benchmark_all(inputs)

    @benchmark_mode
    def benchmark_frontend(self, inputs):
        self.shark_runner.benchmark_frontend(inputs)

    @benchmark_mode
    def benchmark_python(self, inputs):
        self.shark_runner.benchmark_python(inputs)

    @benchmark_mode
    def benchmark_c(self):
        self.shark_runner.benchmark_c()
