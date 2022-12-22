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

from typing import Dict, Any

import iree
import iree.runtime as ireert
import numpy as np
import torch
from iree.runtime import DeviceArray
from torch_mlir._mlir_libs._mlir.ir import Module
from torch_mlir.compiler_utils import (
    run_pipeline_with_repro_report,
)
from torch_mlir.eager_mode.torch_mlir_eager_backend import (
    TorchMLIREagerBackend,
    TensorMetaData,
)
from torch_mlir_e2e_test.eager_backends.refbackend import (
    NUMPY_TO_TORCH_DTYPE_DICT,
)

from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    IREE_DEVICE_MAP,
)


class EagerModeIREELinalgOnTensorsBackend(TorchMLIREagerBackend):
    """Main entry-point for the iree backend for torch-mlir eager mode.

    EagerModeIREELinalgOnTensorsBackend uses iree.DeviceArray representations of tensors and
    thus all of the wrapping and unwrapping and munging here is done to between torch.Tensor and iree.DeviceArray,
    with np.ndarray as an intermediary.
    """

    def __init__(self, device: str):
        self.torch_device_str = device
        self.config = ireert.Config(IREE_DEVICE_MAP[device])
        self.raw_device_str = device

    def get_torch_metadata(
        self, tensor: DeviceArray, kwargs: Dict[str, Any]
    ) -> TensorMetaData:
        return TensorMetaData(
            size=tensor.shape,
            dtype=NUMPY_TO_TORCH_DTYPE_DICT[tensor.dtype.type],
            device=torch.device(self.torch_device_str),
            requires_grad=tensor.dtype.type
            in {np.float, np.float32, np.float64}
            and kwargs.get("requires_grad", False),
        )

    def compile(self, imported_module: Module):
        run_pipeline_with_repro_report(
            imported_module,
            "torch-function-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline",
            "EagerMode",
        )
        callable, _ = get_iree_compiled_module(
            imported_module, self.raw_device_str
        )
        return callable

    def copy_into(self, dst, src):
        """Copy output back to appropriate arg that it should alias."""
        np.copyto(dst, src)

    def transfer_from_device_to_torch(self, e):
        return torch.from_numpy(e.to_host())

    def transfer_from_torch_to_device(
        self, tensor: torch.Tensor
    ) -> DeviceArray:
        return iree.runtime.asdevicearray(self.config.device, tensor.numpy())
