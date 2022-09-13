# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
import contextlib
import re
import traceback
import warnings
from typing import Any
import numpy as np

import torch
from torch.utils._pytree import tree_map

from torch_mlir.eager_mode.ir_building import build_mlir_module
from torch_mlir.eager_mode.torch_mlir_dispatch import (
    UnsupportedByTorchMlirEagerMode,
    normalize_args_kwargs,
    check_get_aliased_arg,
)
from torch_mlir.eager_mode import EAGER_MODE_DEBUG
from torch_mlir.eager_mode.torch_mlir_tensor import (
    TorchMLIRTensor,
    check_requires_grad,
    make_wrapper_subclass_from_torch_tensor,
    make_bare_wrapper_subclass,
    UNSUPPORTED_OPS,
    no_dispatch,
)
from torch_mlir.eager_mode import torch_mlir_tensor
from shark.iree_eager_backend import EagerModeIREELinalgOnTensorsBackend


backend = EagerModeIREELinalgOnTensorsBackend("cpu")
torch_mlir_tensor.backend = backend
rtol = 1e-04
atol = 1e-05


class TorchMLIRLockstepTensor(TorchMLIRTensor):
    """This class overrides the dispatching for TorchMLIRTensor to allow for an op-by-op numerical comparison between PyTorch and the Torch-MLIR -> IREE backend compilation pipeline. This only supports the IREE backend and focuses on op-by-op level verification.

    TODO: Extend this to do a cumulative trace with summary statistics at the end. Possibly requires a wrapper environment to store full trace info.
    """

    def __new__(cls, elem, **kwargs):
        if kwargs.get("constructing_from_device_tensor", False):
            tensor_meta_data = backend.get_torch_metadata(elem, kwargs)
            r = make_bare_wrapper_subclass(
                cls=cls,
                size=tensor_meta_data.size,
                strides=tensor_meta_data.strides,
                storage_offset=tensor_meta_data.storage_offset,
                dtype=tensor_meta_data.dtype,
                layout=tensor_meta_data.layout,
                device=tensor_meta_data.device,
                requires_grad=tensor_meta_data.requires_grad,
            )
            r.elem = elem
        elif isinstance(elem, torch.nn.Parameter):
            r = make_wrapper_subclass_from_torch_tensor(
                cls, elem.data, **kwargs
            )
            # This is a hack to handle non-contiguous data through IREE-backend
            nt = elem.detach().data.numpy()
            if not nt.flags["C_CONTIGUOUS"]:
                nt = np.ascontiguousarray(nt, dtype=nt.dtype)
            r.elem = backend.transfer_from_torch_to_device(
                torch.from_numpy(nt)
            )
        elif isinstance(elem, torch.Tensor):
            r = make_wrapper_subclass_from_torch_tensor(cls, elem, **kwargs)
            # Ditto TODO: Find a better way to handle this
            nt = elem.numpy()
            if not nt.flags["C_CONTIGUOUS"]:
                nt = np.ascontiguousarray(nt, dtype=nt.dtype)
            r.elem = backend.transfer_from_torch_to_device(
                torch.from_numpy(nt)
            )
        # This branch handles the case when a python scalar is passed to some op
        # or is returned from some aten op, such as _local_scalar_dense.
        elif isinstance(elem, (int, float, bool)):
            return elem
        else:
            raise ValueError(f"Unknown element type: {type(elem)}")
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"TorchMLIRLockstepTensor({self.elem}, backend={backend.__class__.__name__}, grad_fn={self.grad_fn})"
        else:
            return f"TorchMLIRLockstepTensor({self.elem}, backend={backend.__class__.__name__})"

    """This does essentially the same dispatch as TorchMLIRTensor but operates as if debug mode is enabled. The numeric verification happens after the Torch-MLIR result is obtained by comparing against the 
    """

    @classmethod
    def __torch_dispatch__(cls, func, _types, args=(), kwargs=None):
        requires_grad = check_requires_grad(*args, **kwargs)
        try:
            with no_dispatch():
                if hasattr(func, "op_name"):
                    op_name = func.op_name
                elif hasattr(func, "__name__"):
                    # Handle builtin_function_or_method.
                    op_name = func.__name__
                else:
                    raise RuntimeError(f"op {func} has no name")

                if UNSUPPORTED_OPS.match(op_name):
                    raise UnsupportedByTorchMlirEagerMode(op_name)

                if not hasattr(func, "_schema"):
                    raise RuntimeError(f"op {func} has no schema.")

                normalized_kwargs = normalize_args_kwargs(func, args, kwargs)

                if "layout" in normalized_kwargs and normalized_kwargs[
                    "layout"
                ] not in {0, None}:
                    raise UnsupportedByTorchMlirEagerMode(
                        f"{normalized_kwargs['layout']} layout not supported."
                    )
                if "memory_format" in normalized_kwargs and normalized_kwargs[
                    "memory_format"
                ] not in {0, None}:
                    raise UnsupportedByTorchMlirEagerMode(
                        f"{normalized_kwargs['memory_format']} memory format not supported."
                    )
                eager_module = build_mlir_module(func, normalized_kwargs)
            device_tensor_args = [
                kwarg.elem
                for _, kwarg in normalized_kwargs.items()
                if isinstance(kwarg, cls)
            ]
            assert len(eager_module.body.operations[0].arguments) == len(
                device_tensor_args
            ), "Number of parameters and number of arguments differs."
            op_mlir_backend_callable = backend.compile(eager_module)
            out = op_mlir_backend_callable(*device_tensor_args)
            out = tree_map(
                lambda x: cls(
                    x,
                    requires_grad=requires_grad,
                    constructing_from_device_tensor=True,
                ),
                out,
            )

            # Numeric verification; Value for comparison comes from PyTorch eager
            with no_dispatch():
                unwrapped_args = tree_map(cls.unwrap, args)
                unwrapped_kwargs = tree_map(cls.unwrap, kwargs)
                native_out = func(*unwrapped_args, **unwrapped_kwargs)

            native_out = tree_map(
                lambda x: cls(x, requires_grad=requires_grad), native_out
            ).elem
            tmp_out = out.elem

            try:
                np.testing.assert_allclose(
                    native_out.to_host(),
                    tmp_out.to_host(),
                    rtol=rtol,
                    atol=atol,
                )
            except Exception as e:
                shaped_args = [
                    arg.shape if torch.is_tensor(arg) else arg
                    for arg in unwrapped_args
                ]
                shaped_kwargs = [
                    kwarg.shape if torch.is_tensor(kwarg) else kwarg
                    for kwarg in unwrapped_kwargs
                ]
                warnings.warn(
                    f"Lockstep accuracy verification failed with error: *{str(e)}*; "
                    f"Dispatched function name: *{str(func)}*; "
                    f"Dispatched function args: *{str(shaped_args)}*; "
                    f"Dispatched function kwargs: *{str(shaped_kwargs)}*; "
                )
        except Exception as e:
            warnings.warn(traceback.format_exc())
            if isinstance(e, UnsupportedByTorchMlirEagerMode):
                warnings.warn(
                    f"Couldn't use TorchMLIR eager because current incompatibility: *{str(e)}*; running through PyTorch eager."
                )
            else:
                warnings.warn(
                    f"Couldn't use TorchMLIR eager because of error: *{str(e)}*; "
                    f"Running through PyTorch eager"
                )

            with no_dispatch():
                unwrapped_args = tree_map(cls.unwrap, args)
                unwrapped_kwargs = tree_map(cls.unwrap, kwargs)
                out = func(*unwrapped_args, **unwrapped_kwargs)

            out = tree_map(lambda x: cls(x, requires_grad=requires_grad), out)

        maybe_aliased_arg_name = check_get_aliased_arg(func)
        if maybe_aliased_arg_name is not None:
            warnings.warn(
                f"Found aliased arg, but didn't copy tensor contents. This could lead to incorrect results for E2E model execution but doesn't affect the validity of the lockstep op verification."
            )
            # TODO: Find a way to handle argument aliasing for IREE backend
            # backend.copy_into(normalized_kwargs[maybe_aliased_arg_name].elem, out.elem)

        return out
