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
from shark.parser import shark_args
from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    export_iree_module_to_vmfb,
    export_module_to_mlir_file,
    get_results,
)
import os


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
        self.device = device if device is not None else shark_args.device

        if self.frontend in ["tflite-tosa"]:
            func_name = "main"
        elif self.frontend in ["pytorch", "torch"]:
            # get torch-mlir dialect
            # self.model = torch.Module
            # Lowers in linalg dialect.
            # TODO assert
            # TODO tosa dialect from torch_module.
            self.model = get_torch_mlir_module(
                self.model, input, dynamic, jit_trace, from_aot
            )
        elif self.frontend in ["tensorflow", "tf"]:
            # get mhlo dialect
            # self.model = tf.Module
            # TODO assert
            self.model = tfc.compile_module(
                self.model, exported_names=[func_name], import_only=True
            )
        elif self.frontend in ["tflite"]:
            print("Setting up for IREE compiler tflite")
            # get tosa dialect
            # self.model = model.tflite
            # TODO assert
            self.model = ireec_tflite.compile_file(
                self.model, input_type="tosa", import_only=True
            )
            func_name = "main"

        # TODO: We can capture the .vmfb module here and later use it for saving
        # rather than recompiling it again, if used for saving.
        (
            self.iree_compilation_module,
            self.iree_config,
        ) = get_iree_compiled_module(
            self.model,
            self.device,
            self.frontend,
            func_name=func_name,
            model_config_path=model_config_path,
        )

        # Debugging Options:
        if shark_args.save_mlir:
            export_module_to_mlir_file(
                self.model, self.frontend, shark_args.repro_dir
            )
        if shark_args.save_vmfb:
            self.vmfb_file = self.save_module(shark_args.repro_dir)

    # All the timings and benchmarking can be done here.
    def forward(self, input, frontend):
        return get_results(
            self.iree_compilation_module, input, self.iree_config, frontend
        )

    # Saves the .mlir file, can be in tosa, linalg or mhlo dialect.
    # torch-mlir can export tosa or linalg dialects.
    # tensorflow models get exported to mhlo dialect.
    def import_mlir(self, model_name, dir):
        filename = os.path.join(dir, f"{model_name}_{self.frontend}.mlir")
        with open(filename, "w") as f:
            f.write(self.model)
        print(f"Saved mlir in {filename}.")
        return filename

    # TODO: Instead of passing directory and having names decided by the module
    # , user may want to save the module with manual names.
    def save_module(self, dir=os.getcwd()):
        return export_iree_module_to_vmfb(
            self.model, self.device, dir, self.frontend
        )

    # TODO: Load a module and directly use it, we will need to set the frontend
    # in this case.
    def load_module(self, name):
        pass


# TODO: Document shark_eager mode.
class SharkEagerMode:
    def __init__(self, device="cpu"):
        if device == "refbackend":
            torch_mlir_tensor.backend = EagerModeRefBackend()
        else:
            torch_mlir_tensor.backend = EagerModeIREELinalgOnTensorsBackend(
                device
            )
        self.guard = enable_torch_dispatch_mode(TorchMLIRTensor)
        self.guard.__enter__()

    def __del__(self):
        self.guard.__exit__(None, None, None)
