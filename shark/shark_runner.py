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
)
from shark.iree_utils._common import check_device_drivers, device_driver_info
import os
import sys


# supported dialects by the shark-runtime.
supported_dialects = {"linalg", "mhlo", "tosa", "tf-lite"}


class SharkRunner:
    """
    Base class for SharkInference and SharkTrainer
    used to execute an mlir_module.

    ...

    Attributes
    ----------
    mlir_module : str
        mlir_module represented in string.
    function_name : str
        function to execute in the given mlir_module.
    device : str
        device to execute the mlir_module on.
        currently supports cpu, cuda, vulkan, and metal backends.
    mlir_dialect: str
        The dialect in which the given mlir_module is in.
        Refer to {https://mlir.llvm.org/docs/Dialects/}

    Methods
    -------
    run(inputs=None):
        Runs the mlir_module with the given inputs, if the inputs are not
        given it autogenerates the inputs. Also, the inputs should be a
        numpy array.
    input_info():
        Gives the information about the inputs required by the `function_name`.
        This can be expensive as it does string matching to do so.
    """

    def __init__(
        self,
        mlir_module: str,
        function_name: str = "forward",
        device: str = "cpu",
        mlir_dialect: str = "linalg",
    ):
        self.mlir_module = mlir_module
        self.function_name = function_name
        self.device = device
        self.mlir_dialect = mlir_dialect

        if check_device_drivers(self.device):
            device_driver_info(self.device)
            sys.exit(1)

        # Compile the module to get the .vmfb.
        (self.iree_compilation_module, self.iree_config,) = get_iree_compiled_module(
            self.mlir_module,
            self.device,
            self.mlir_dialect,
            func_name=self.function_name,
        )

    def run(self, inputs: tuple):
        return get_results(
            self.iree_compilation_module,
            inputs,
            self.iree_config,
            self.mlir_dialect,
        )

    # TODO: Instead of passing directory and having names decided by the module
    # , user may want to save the module with manual names.
    def save_module(self, dir=os.getcwd()):
        return export_iree_module_to_vmfb(self.model, self.device, dir, self.mlir_dialect)
