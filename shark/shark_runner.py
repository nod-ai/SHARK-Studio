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
from iree.compiler import tf as tfc
import iree.compiler.tflite as ireec_tflite
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch_mlir.eager_mode import torch_mlir_tensor
from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor
from torch_mlir_e2e_test.eager_backends.refbackend import EagerModeRefBackend

from shark.iree_eager_backend import EagerModeIREELinalgOnTensorsBackend
from shark.torch_mlir_utils import get_torch_mlir_module, run_on_refbackend
from shark.iree_utils import get_results, get_iree_compiled_module, export_iree_module_to_vmfb, export_module_to_mlir_file, build_benchmark_args, run_benchmark
import os
from shark.parser import shark_args
from tqdm import tqdm
import time


class SharkRunner:
    """Base class for Shark Inference and Shark Runner."""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = False,
        frontend: str = "torch",
        model_config_path: str = None,
    ):
        self.model = model
        self.frontend_model = model
        self.from_aot = from_aot
        self.input = input
        self.frontend = frontend
        self.vmfb_file = None
        func_name = "forward"
        device = device if device is not None else shark_args.device
        if self.frontend in ["pytorch", "torch"]:
            #get torch-mlir dialet
            # self.model = torch.Module
            # TODO assert
            self.model = get_torch_mlir_module(self.model, input, dynamic,
                                               jit_trace, from_aot)
        elif frontend in ["tensorflow", "tf"]:
            # get mhlo dialect
            # self.model = tf.Module
            # TODO assert
            self.model = tfc.compile_module(self.model,
                                            exported_names=[func_name],
                                            import_only=True)
        elif frontend in ["tflite"]:
            print("Setting up for IREE tflite")
            # get tosa dialect
            # self.model = model.tflite
            # TODO assert
            self.model = ireec_tflite.compile_file(self.model,
                                                   input_type="tosa",
                                                   import_only=True)
        (
            self.iree_compilation_module,
            self.iree_config,
        ) = get_iree_compiled_module(self.model,
                                     device,
                                     self.frontend,
                                     model_config_path=model_config_path)

        # Debugging Options:
        if shark_args.save_mlir:
            export_module_to_mlir_file(self.model, self.frontend,
                                       shark_args.repro_dir)
        if shark_args.save_vmfb:
            self.vmfb_file = export_iree_module_to_vmfb(self.model, device,
                                                        shark_args.repro_dir,
                                                        frontend)

    # All the timings and benchmarking can be done here.
    def forward(self, input, frontend):
        return get_results(self.iree_compilation_module, input,
                           self.iree_config, frontend)


class SharkEagerMode:

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


class SharkBenchmarkRunner(SharkRunner):
    # SharkRunner derived class with Benchmarking capabilities.
    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = None,
        jit_trace: bool = False,
        from_aot: bool = False,
        frontend: str = "torch",
    ):
        SharkRunner.__init__(self, model, input, dynamic, device, jit_trace,
                             from_aot, frontend)
        if (self.vmfb_file == None):
            self.vmfb_file = export_iree_module_to_vmfb(self.model, device,
                                                        shark_args.repro_dir,
                                                        frontend)
        self.benchmark_cl = build_benchmark_args(self.vmfb_file, device, input,
                                                 frontend, from_aot)

    def benchmark_frontend(self, inputs):
        if self.frontend in ["pytorch", "torch"]:
            self.benchmark_torch(inputs)
        elif self.frontend in ["tensorflow", "tf"]:
            self.benchmark_tf(inputs)

    def benchmark_torch(self, inputs):
        inputs = self.input if self.from_aot else inputs
        inputs = inputs[0]
        for i in range(shark_args.num_warmup_iterations):
            self.frontend_model.forward(inputs)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.frontend_model.forward(inputs)
            if i == shark_args.num_iterations - 1:
                end = time.time()
                break
        print(
            f"Torch benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )

    def benchmark_tf(self, inputs):
        for i in range(shark_args.num_warmup_iterations):
            self.frontend_model.forward(*inputs)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.frontend_model.forward(*inputs)
            if i == shark_args.num_iterations - 1:
                end = time.time()
                break
        print(
            f"TF benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )
        return

    def benchmark_c(self):
        result = run_benchmark(self.benchmark_cl)
        print(f"Shark-{self.frontend} C-benchmark:{result} iter/second")

    def benchmark_python(self, inputs):
        inputs = self.input if self.from_aot else inputs
        input_list = [x for x in inputs]
        for i in range(shark_args.num_warmup_iterations):
            self.forward(input_list, self.frontend)

        begin = time.time()
        for i in range(shark_args.num_iterations):
            out = self.forward(input_list, self.frontend)
            if i == shark_args.num_iterations - 1:
                end = time.time()
        print(
            f"Shark-{self.frontend} Python-benchmark:{shark_args.num_iterations/(end-begin)} iter/second, Total Iterations:{shark_args.num_iterations}"
        )

    def benchmark_all(self, inputs):
        self.benchmark_frontend(inputs)
        self.benchmark_python(inputs)
        self.benchmark_c()
