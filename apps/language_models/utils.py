import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from pathlib import Path


# expects a Path / str as arg
# returns None if path not found or SharkInference module
def get_vmfb_from_path(vmfb_path, device, mlir_dialect):
    if not isinstance(vmfb_path, Path):
        vmfb_path = Path(vmfb_path)

    from shark.shark_inference import SharkInference

    if not vmfb_path.exists():
        return None

    print("Loading vmfb from: ", vmfb_path)
    shark_module = SharkInference(
        None, device=device, mlir_dialect=mlir_dialect
    )
    shark_module.load_module(vmfb_path)
    print("Successfully loaded vmfb")
    return shark_module
