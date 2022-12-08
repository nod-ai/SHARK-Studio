import argparse
from pathlib import Path


def path_expand(s):
    return Path(s).expanduser().resolve()


p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

##############################################################################
### Stable Diffusion Params
##############################################################################

p.add_argument(
    "-p",
    "--prompts",
    action="append",
    default=[],
    help="text of which images to be generated.",
)

p.add_argument(
    "--negative-prompts",
    nargs="+",
    default=[""],
    help="text you don't want to see in the generated image.",
)

p.add_argument(
    "--steps",
    type=int,
    default=50,
    help="the no. of steps to do the sampling.",
)

p.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed to use.",
)

p.add_argument(
    "--batch_size",
    type=int,
    default=1,
    choices=range(1, 4),
    help="the number of inferences to be made in a single `run`.",
)

p.add_argument(
    "--height",
    type=int,
    default=512,
    help="the height of the output image.",
)

p.add_argument(
    "--width",
    type=int,
    default=512,
    help="the width of the output image.",
)

p.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="the value to be used for guidance scaling.",
)

p.add_argument(
    "--max_length",
    type=int,
    default=64,
    help="max length of the tokenizer output, options are 64 and 77.",
)

##############################################################################
### Model Config and Usage Params
##############################################################################

p.add_argument(
    "--device", type=str, default="vulkan", help="device to run the model."
)

p.add_argument(
    "--precision", type=str, default="fp16", help="precision to run the model."
)

p.add_argument(
    "--import_mlir",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="imports the model from torch module to shark_module otherwise downloads the model from shark_tank.",
)

p.add_argument(
    "--load_vmfb",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="attempts to load the model from a precompiled flatbuffer and compiles + saves it if not found.",
)

p.add_argument(
    "--save_vmfb",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="saves the compiled flatbuffer to the local directory",
)

p.add_argument(
    "--use_tuned",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Download and use the tuned version of the model if available",
)

p.add_argument(
    "--use_base_vae",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Do conversion from the VAE output to pixel space on cpu.",
)

p.add_argument(
    "--scheduler",
    type=str,
    default="SharkEulerDiscrete",
    help="other supported schedulers are [PNDM, DDIM, LMSDiscrete, EulerDiscrete, DPMSolverMultistep]",
)

p.add_argument(
    "--output_img_format",
    type=str,
    default="png",
    help="specify the format in which output image is save. Supported options: jpg / png",
)

p.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory path to save the output images and json",
)

p.add_argument(
    "--runs",
    type=int,
    default=1,
    help="number of images to be generated with random seeds in single execution",
)

p.add_argument(
    "--ckpt_loc",
    type=str,
    default="",
    help="Path to SD's .ckpt file.",
)

p.add_argument(
    "--hf_model_id",
    type=str,
    default="stabilityai/stable-diffusion-2-1-base",
    help="The repo-id of hugging face.",
)

p.add_argument(
    "--enable_stack_trace",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Enable showing the stack trace when retrying the base model configuration",
)

##############################################################################
### IREE - Vulkan supported flags
##############################################################################

p.add_argument(
    "--iree-vulkan-target-triple",
    type=str,
    default="",
    help="Specify target triple for vulkan",
)

p.add_argument(
    "--vulkan_debug_utils",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Profiles vulkan device and collects the .rdc info",
)

p.add_argument(
    "--vulkan_large_heap_block_size",
    default="4147483648",
    help="flag for setting VMA preferredLargeHeapBlockSize for vulkan device, default is 4G",
)

p.add_argument(
    "--vulkan_validation_layers",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="flag for disabling vulkan validation layers when benchmarking",
)

##############################################################################
### Misc. Debug and Optimization flags
##############################################################################

p.add_argument(
    "--use_compiled_scheduler",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="use the default scheduler precompiled into the model if available",
)

p.add_argument(
    "--local_tank_cache",
    default="",
    help="Specify where to save downloaded shark_tank artifacts. If this is not set, the default is ~/.local/shark_tank/.",
)

p.add_argument(
    "--dump_isa",
    default=False,
    action="store_true",
    help="When enabled call amdllpc to get ISA dumps. use with dispatch benchmarks.",
)

p.add_argument(
    "--dispatch_benchmarks",
    default=None,
    help='dispatches to return benchamrk data on.  use "All" for all, and None for none.',
)

p.add_argument(
    "--dispatch_benchmarks_dir",
    default="temp_dispatch_benchmarks",
    help='directory where you want to store dispatch data generated with "--dispatch_benchmarks"',
)

p.add_argument(
    "--enable_rgp",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="flag for inserting debug frames between iterations for use with rgp.",
)

p.add_argument(
    "--hide_steps",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="flag for hiding the details of iteration/sec for each step.",
)

p.add_argument(
    "--warmup_count",
    type=int,
    default=0,
    help="flag setting warmup count for clip and vae [>= 0].",
)

p.add_argument(
    "--clear_all",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="flag to clear all mlir and vmfb from common locations. Recompiling will take several minutes",
)

##############################################################################
### Web UI flags
##############################################################################

p.add_argument(
    "--progress_bar",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="flag for removing the pregress bar animation during image generation",
)

p.add_argument(
    "--share",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="flag for generating a public URL",
)

p.add_argument(
    "--server_port",
    type=int,
    default=8080,
    help="flag for setting server port",
)

##############################################################################
### SD model auto-annotation flags
##############################################################################

p.add_argument(
    "--annotation_output",
    type=path_expand,
    default="./",
    help="Directory to save the annotated mlir file",
)

p.add_argument(
    "--annotation_model",
    type=str,
    default="unet",
    help="Options are unet and vae.",
)

p.add_argument(
    "--use_winograd",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Apply Winograd on selected conv ops.",
)

args = p.parse_args()
