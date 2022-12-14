import argparse

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

p.add_argument(
    "--prompts",
    nargs="+",
    default=["a photograph of an astronaut riding a horse"],
    help="text of which images to be generated.",
)

p.add_argument(
    "--device", type=str, default="vulkan", help="device to run the model."
)

p.add_argument(
    "--steps",
    type=int,
    default=50,
    help="the no. of steps to do the sampling.",
)

p.add_argument(
    "--version",
    type=str,
    default="v2.1base",
    help="Specify version of stable diffusion model",
)

p.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed to use.",
)

p.add_argument(
    "--height",
    type=int,
    default=512,
    help="the height to use.",
)

p.add_argument(
    "--width",
    type=int,
    default=512,
    help="the width to use.",
)

p.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="the value to be used for guidance scaling.",
)

p.add_argument(
    "--scheduler",
    type=str,
    default="EulerDiscrete",
    help="can be [PNDM, LMSDiscrete, DDIM, DPMSolverMultistep, EulerDiscrete]",
)

p.add_argument(
    "--import_mlir",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="imports the model from torch module to shark_module otherwise downloads the model from shark_tank.",
)

p.add_argument(
    "--precision", type=str, default="fp16", help="precision to run the model."
)

p.add_argument(
    "--max_length",
    type=int,
    default=77,
    help="max length of the tokenizer output.",
)

p.add_argument(
    "--cache",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="attempts to load the model from a precompiled flatbuffer and compiles + saves it if not found.",
)

p.add_argument(
    "--iree-vulkan-target-triple",
    type=str,
    default="",
    help="Specify target triple for vulkan",
)

p.add_argument(
    "--use_tuned",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Download and use the tuned version of the model if available",
)

p.add_argument(
    "--vulkan_large_heap_block_size",
    default="4294967296",
    help="flag for setting VMA preferredLargeHeapBlockSize for vulkan device, default is 4G",
)

args = p.parse_args()
