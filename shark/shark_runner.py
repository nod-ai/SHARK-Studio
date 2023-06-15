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

from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    get_results,
    export_iree_module_to_vmfb,
    load_flatbuffer,
)
from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.parser import shark_args
import os
import sys


# supported dialects by the shark-runtime.
supported_dialects = {
    "linalg",
    "auto",
    "stablehlo",
    "tosa",
    "tf-lite",
    "tm_tensor",
}


class SharkRunner:
    """
    Base class for SharkInference and SharkTrainer
    used to execute an mlir_module.

    ...

    Attributes
    ----------
    mlir_module : str
        mlir_module represented in string.
    device : str
        device to execute the mlir_module on.
        currently supports cpu, cuda, vulkan, and metal backends.
    mlir_dialect: str
        The dialect in which the given mlir_module is in.
        Refer to {https://mlir.llvm.org/docs/Dialects/}

    Methods
    -------
    run(function_name, inputs=None):
        Runs the function with `function_name` within the mlir_module along
        with the given inputs, if the inputs are not given it autogenerates the
        inputs. Also, the inputs should be a numpy array.
    input_info():
        Gives the information about the inputs required by the `function_name`.
        This can be expensive as it does string matching to do so.
    """

    def __init__(
        self,
        mlir_module: bytes = None,
        device: str = "none",
        mlir_dialect: str = "linalg",
        extra_args: list = [],
        compile_vmfb: bool = True,
        device_idx: int = None,
    ):
        self.mlir_module = mlir_module
        self.device = shark_args.device if device == "none" else device
        self.mlir_dialect = mlir_dialect
        self.extra_args = extra_args
        self.device_idx = device_idx

        if check_device_drivers(self.device):
            print(device_driver_info(self.device))
            sys.exit(1)

        if compile_vmfb == True:
            # Compile the module to get the .vmfb.
            params = get_iree_compiled_module(
                self.mlir_module,
                self.device,
                self.mlir_dialect,
                extra_args=self.extra_args,
                device_idx=self.device_idx,
            )
            self.iree_compilation_module = params["vmfb"]
            self.iree_config = params["config"]
            self.temp_file_to_unlink = params["temp_file_to_unlink"]
            del params

    def run(self, function_name, inputs: tuple, send_to_host=False):
        return get_results(
            self.iree_compilation_module,
            function_name,
            inputs,
            self.iree_config,
            self.mlir_dialect,
            send_to_host,
        )

    # Get all function names defined within the compiled module.
    def get_functions_in_module(self):
        return self.iree_compilation_module._vm_module.function_names
