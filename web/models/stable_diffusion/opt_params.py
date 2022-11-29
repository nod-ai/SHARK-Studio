import sys
from models.stable_diffusion.model_wrappers import (
    get_vae_mlir,
    get_unet_mlir,
    get_clip_mlir,
)
from models.stable_diffusion.utils import get_shark_model


def get_unet(args):
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    # Tuned model is present for `fp16` precision.
    if args.precision == "fp16":
        if args.use_tuned:
            bucket = "gs://shark_tank/quinn"
            model_name = "unet_22nov_fp16_tuned"
            return get_shark_model(args, bucket, model_name, iree_flags)
        else:
            bucket = "gs://shark_tank/prashant_nod"
            model_name = "unet_23nov_fp16"
            iree_flags += [
                "--iree-flow-enable-padding-linalg-ops",
                "--iree-flow-linalg-ops-padding-size=32",
                "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            ]
            if args.import_mlir:
                return get_unet_mlir(args, model_name, iree_flags)
            return get_shark_model(args, bucket, model_name, iree_flags)

    # Tuned model is not present for `fp32` case.
    if args.precision == "fp32":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "unet_23nov_fp32"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            return get_unet_mlir(args, model_name, iree_flags)
        return get_shark_model(args, bucket, model_name, iree_flags)


def get_vae(args):
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    if args.precision == "fp16":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "vae_22nov_fp16"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
        ]
        if args.import_mlir:
            return get_vae_mlir(args, model_name, iree_flags)
        return get_shark_model(args, bucket, model_name, iree_flags)

    if args.precision == "fp32":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "vae_22nov_fp32"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            return get_vae_mlir(args, model_name, iree_flags)
        return get_shark_model(args, bucket, model_name, iree_flags)


def get_clip(args):
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    bucket = "gs://shark_tank/prashant_nod"
    model_name = "clip_18nov_fp32"
    iree_flags += [
        "--iree-flow-linalg-ops-padding-size=16",
        "--iree-flow-enable-padding-linalg-ops",
    ]
    if args.import_mlir:
        return get_clip_mlir(args, model_name, iree_flags)
    return get_shark_model(args, bucket, model_name, iree_flags)
