import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from pathlib import Path
from shark.shark_downloader import download_public_file


# expects a Path / str as arg
# returns None if path not found or SharkInference module
def get_vmfb_from_path(vmfb_path, device, mlir_dialect, device_id=None):
    if not isinstance(vmfb_path, Path):
        vmfb_path = Path(vmfb_path)

    from shark.shark_inference import SharkInference

    if not vmfb_path.exists():
        return None

    print("Loading vmfb from: ", vmfb_path)
    print("Device from get_vmfb_from_path - ", device)
    shark_module = SharkInference(
        None, device=device, mlir_dialect=mlir_dialect, device_idx=device_id
    )
    shark_module.load_module(vmfb_path)
    print("Successfully loaded vmfb")
    return shark_module


def get_vmfb_from_config(
    shark_container,
    model,
    precision,
    device,
    vmfb_path,
    padding=None,
    device_id=None,
):
    vmfb_url = (
        f"gs://shark_tank/{shark_container}/{model}_{precision}_{device}"
    )
    if padding:
        vmfb_url = vmfb_url + f"_{padding}"
    vmfb_url = vmfb_url + ".vmfb"
    download_public_file(vmfb_url, vmfb_path.absolute(), single_file=True)
    return get_vmfb_from_path(
        vmfb_path, device, "tm_tensor", device_id=device_id
    )
