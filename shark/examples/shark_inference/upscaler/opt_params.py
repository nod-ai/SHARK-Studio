import sys
from model_wrappers import (
    get_vae_mlir,
    get_unet_mlir,
    get_clip_mlir,
)
from upscaler_args import args
from utils import get_shark_model

BATCH_SIZE = len(args.prompts)
if BATCH_SIZE != 1:
    sys.exit("Only batch size 1 is supported.")


unet_flag = [
    "--iree-flow-enable-padding-linalg-ops",
    "--iree-flow-linalg-ops-padding-size=32",
    "--iree-flow-enable-conv-img2col-transform",
]

vae_flag = [
    "--iree-flow-enable-conv-nchw-to-nhwc-transform",
    "--iree-flow-enable-padding-linalg-ops",
    "--iree-flow-linalg-ops-padding-size=16",
]

clip_flag = [
    "--iree-flow-linalg-ops-padding-size=16",
    "--iree-flow-enable-padding-linalg-ops",
]

bucket = "gs://shark_tank/stable_diffusion/"


def get_unet():
    model_name = "upscaler_unet"
    if args.import_mlir:
        return get_unet_mlir(model_name, unet_flag)
    return get_shark_model(bucket, model_name, unet_flag)


def get_vae():
    model_name = "upscaler_vae"
    if args.import_mlir:
        return get_vae_mlir(model_name, vae_flag)
    return get_shark_model(bucket, model_name, vae_flag)


def get_clip():
    model_name = "upscaler_clip"
    if args.import_mlir:
        return get_clip_mlir(model_name, clip_flag)
    return get_shark_model(bucket, model_name, clip_flag)

get_unet_mlir()
