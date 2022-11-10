import sys
from model_wrappers import (
    get_vae32,
    get_vae16,
    get_unet16_wrapped,
    get_unet32_wrapped,
    get_clipped_text,
)
from stable_args import args
from utils import get_shark_model

BATCH_SIZE = len(args.prompts)
if BATCH_SIZE != 1:
    sys.exit("Only batch size 1 is supported.")


def get_unet():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    # Tuned model is present for `fp16` precision.
    if args.precision == "fp16":
        if args.use_tuned:
            sys.exit("The tuned version is in maintenance, stay tuned.")
            bucket = "gs://shark_tank/quinn"
            model_name = "unet_fp16_tunedv2"
            iree_flags += [
                "--iree-flow-enable-padding-linalg-ops",
                "--iree-flow-linalg-ops-padding-size=32",
            ]
            # TODO: Pass iree_flags to the exported model.
            if args.import_mlir:
                return get_unet16_wrapped(model_name=model_name)
            return get_shark_model(bucket, model_name, iree_flags)
        else:
            bucket = "gs://shark_tank/prashant_nod"
            model_name = "unet_fp16_v2"
            iree_flags += [
                "--iree-flow-enable-padding-linalg-ops",
                "--iree-flow-linalg-ops-padding-size=32",
                "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            ]
            if args.import_mlir:
                return get_unet16_wrapped(model_name=model_name)
            return get_shark_model(bucket, model_name, iree_flags)

    # Tuned model is not present for `fp32` case.
    if args.precision == "fp32":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "unet_fp32_v2"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            return get_unet32_wrapped(model_name=model_name)
        return get_shark_model(bucket, model_name, iree_flags)

    if args.precision == "int8":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "unet_int8"
        iree_flags += [
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
        ]
        # TODO: Pass iree_flags to the exported model.
        if args.import_mlir:
            sys.exit(
                "--import_mlir is not supported for the int8 model, try --no-import_mlir flag."
            )
        return get_shark_model(bucket, model_name, iree_flags)


def get_vae():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    if args.precision in ["fp16", "int8"]:
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "vae_fp16"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=32",
        ]
        if args.import_mlir:
            return get_vae16(model_name)
        return get_shark_model(bucket, model_name, iree_flags)

    if args.precision == "fp32":
        bucket = "gs://shark_tank/prashant_nod"
        model_name = "vae_fp32"
        iree_flags += [
            "--iree-flow-enable-conv-nchw-to-nhwc-transform",
            "--iree-flow-enable-padding-linalg-ops",
            "--iree-flow-linalg-ops-padding-size=16",
        ]
        if args.import_mlir:
            return get_vae32(model_name)
        return get_shark_model(bucket, model_name, iree_flags)


def get_clip():
    iree_flags = []
    if len(args.iree_vulkan_target_triple) > 0:
        iree_flags.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
    bucket = "gs://shark_tank/prashant_nod"
    model_name = "clip_text"
    iree_flags = [
        "--iree-flow-linalg-ops-padding-size=16",
        "--iree-flow-enable-padding-linalg-ops",
    ]
    if args.import_mlir:
        return get_clipped_text(model_name)
    return get_shark_model(bucket, model_name, iree_flags)
