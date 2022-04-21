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

import torch
from torch.utils.cpp_extension import load_inline, include_paths
from torch_mlir.eager_mode import torch_mlir_tensor
from torch_mlir.eager_mode.torch_mlir_tensor import TorchMLIRTensor

from shark.iree_eager_backend import EagerModeIREELinalgOnTensorsBackend
from shark.shark_runner import SharkEagerMode


def test_cpu():
    torch_mlir_tensor.backend = EagerModeIREELinalgOnTensorsBackend("cpu")

    t = torch.ones((10, 10), device="cpu")
    u = 2 * torch.ones((10, 10), device="cpu")

    tt = TorchMLIRTensor(t)
    print(tt)
    uu = TorchMLIRTensor(u)
    print(uu)

    for i in range(NUM_ITERS):
        yy = tt + uu
        print(type(yy))
        print(yy.elem.to_host())
        yy = tt * uu
        print(type(yy))
        print(yy.elem.to_host())


def test_gpu():
    source = """
    #include <iostream>
    #include "cuda.h"
    #include "cuda_runtime_api.h"

    using namespace std;

    void print_free_mem() {
        int num_gpus;
        size_t free, total;
        cudaSetDevice(0);
        int id;
        cudaGetDevice(&id);
        cudaMemGetInfo(&free, &total);
        cout << "GPU " << id << " memory: used=" << (total-free)/(1<<20) << endl;
    }
    """
    gpu_stats = load_inline(
        name="inline_extension",
        cpp_sources=[source],
        extra_include_paths=include_paths(cuda=True),
        functions=["print_free_mem"],
    )
    torch_mlir_tensor.backend = EagerModeIREELinalgOnTensorsBackend("gpu")

    t = torch.ones((10, 10), device="cpu")
    u = 2 * torch.ones((10, 10), device="cpu")

    tt = TorchMLIRTensor(t)
    print(tt)
    uu = TorchMLIRTensor(u)
    print(uu)

    for i in range(NUM_ITERS):
        yy = tt + uu
        print(yy.elem.to_host())
        yy = tt * uu
        print(yy.elem.to_host())
        gpu_stats.print_free_mem()


def test_python_mode_ref_backend():
    # hide this wherever you want?
    _ = SharkEagerMode("refbackend")

    t = torch.ones((10, 10), device="cpu")
    u = torch.ones((10, 10), device="cpu")

    print(t)
    print(u)

    for i in range(NUM_ITERS):
        print(i)
        yy = t + u
        print(yy.elem)
        yy = t * u
        print(yy.elem)


def test_python_mode_iree_cpu():
    # hide this wherever you want?
    _ = SharkEagerMode("cpu")

    t = torch.ones((10, 10), device="cpu")
    u = torch.ones((10, 10), device="cpu")

    print(t)
    print(u)

    for i in range(NUM_ITERS):
        yy = t + u
        print(type(yy))
        print(yy.elem.to_host())
        yy = t * u
        print(type(yy))
        print(yy.elem.to_host())


def test_python_mode_iree_gpu():
    _ = SharkEagerMode("gpu")

    t = torch.ones((10, 10), device="cpu")
    u = torch.ones((10, 10), device="cpu")

    print(t)
    print(u)

    for i in range(NUM_ITERS):
        yy = t + u
        print(type(yy))
        print(yy.elem.to_host())
        yy = t * u
        print(type(yy))
        print(yy.elem.to_host())


if __name__ == "__main__":
    NUM_ITERS = 10
    test_cpu()
    if torch.cuda.is_available():
        test_gpu()
    test_python_mode_ref_backend()
    test_python_mode_iree_cpu()
    test_python_mode_iree_gpu()
