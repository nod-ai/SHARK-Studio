import argparse
import os
from pathlib import Path


def path_expand(s):
    return Path(s).expanduser().resolve()


def is_valid_file(arg):
    if not os.path.exists(arg):
        return None
    else:
        return arg


p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

##############################################################################
### Stable Diffusion Params
##############################################################################

p.add_argument(
    "-p",
    "--prompts",
    nargs="+",
    default=["cyberpunk forest by Salvador Dali"],
    help="text of which images to be generated.",
)

p.add_argument(
    "--negative_prompts",
    nargs="+",
    default=["trees, green"],
    help="text you don't want to see in the generated image.",
)

p.add_argument(
    "--img_path",
    type=str,
    help="Path to the image input for img2img/inpainting",
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
    default=-1,
    help="the seed to use. -1 for a random one.",
)

p.add_argument(
    "--batch_size",
    type=int,
    default=1,
    choices=range(1, 4),
    help="the number of inferences to be made in a single `batch_count`.",
)

p.add_argument(
    "--height",
    type=int,
    default=512,
    choices=range(384, 769, 8),
    help="the height of the output image.",
)

p.add_argument(
    "--width",
    type=int,
    default=512,
    choices=range(384, 769, 8),
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

p.add_argument(
    "--strength",
    type=float,
    default=0.8,
    help="the strength of change applied on the given input image for img2img",
)

##############################################################################
### Inpainting and Outpainting Params
##############################################################################

p.add_argument(
    "--mask_path",
    type=str,
    help="Path to the mask image input for inpainting",
)

p.add_argument(
    "--inpaint_full_res",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If inpaint only masked area or whole picture",
)

p.add_argument(
    "--inpaint_full_res_padding",
    type=int,
    default=32,
    choices=range(0, 257, 4),
    help="Number of pixels for only masked padding",
)

p.add_argument(
    "--pixels",
    type=int,
    default=128,
    choices=range(8, 257, 8),
    help="Number of expended pixels for one direction for outpainting",
)

p.add_argument(
    "--mask_blur",
    type=int,
    default=8,
    choices=range(0, 65),
    help="Number of blur pixels for outpainting",
)

p.add_argument(
    "--left",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If expend left for outpainting",
)

p.add_argument(
    "--right",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If expend right for outpainting",
)

p.add_argument(
    "--top",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If expend top for outpainting",
)

p.add_argument(
    "--bottom",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If expend bottom for outpainting",
)

p.add_argument(
    "--noise_q",
    type=float,
    default=1.0,
    help="Fall-off exponent for outpainting (lower=higher detail) (min=0.0, max=4.0)",
)

p.add_argument(
    "--color_variation",
    type=float,
    default=0.05,
    help="Color variation for outpainting (min=0.0, max=1.0)",
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
    "--batch_count",
    type=int,
    default=1,
    help="number of batch to be generated with random seeds in single execution",
)

p.add_argument(
    "--ckpt_loc",
    type=str,
    default="",
    help="Path to SD's .ckpt file.",
)

p.add_argument(
    "--custom_vae",
    type=str,
    default="",
    help="HuggingFace repo-id or path to SD model's checkpoint whose Vae needs to be plugged in.",
)

p.add_argument(
    "--hf_model_id",
    type=str,
    default="stabilityai/stable-diffusion-2-1-base",
    help="The repo-id of hugging face.",
)

p.add_argument(
    "--low_cpu_mem_usage",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Use the accelerate package to reduce cpu memory consumption",
)

p.add_argument(
    "--attention_slicing",
    type=str,
    default="none",
    help="Amount of attention slicing to use (one of 'max', 'auto', 'none', or an integer)",
)

p.add_argument(
    "--use_stencil",
    choices=["canny", "openpose"],
    help="Enable the stencil feature.",
)

##############################################################################
### IREE - Vulkan supported flags
##############################################################################

p.add_argument(
    "--iree_vulkan_target_triple",
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

p.add_argument(
    "--save_metadata_to_json",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="flag for whether or not to save a generation information json file with the image.",
)

p.add_argument(
    "--write_metadata_to_png",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="flag for whether or not to save generation information in PNG chunk text to generated images.",
)

p.add_argument(
    "--import_debug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="if import_mlir is True, saves mlir via the debug option in shark importer. Does nothing if import_mlir is false (the default)",
)
##############################################################################
### Web UI flags
##############################################################################

p.add_argument(
    "--progress_bar",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="flag for removing the progress bar animation during image generation",
)

p.add_argument(
    "--ckpt_dir",
    type=str,
    default="",
    help="Path to directory where all .ckpts are stored in order to populate them in the web UI",
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
    "--save_annotation",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Save annotated mlir file",
)

args, unknown = p.parse_known_args()
