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
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))"
]

vae_flag = [
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-flow-convert-conv-nchw-to-nhwc,iree-preprocessing-pad-linalg-ops{pad-size=16}))"
]

clip_flag = [
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))"
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
