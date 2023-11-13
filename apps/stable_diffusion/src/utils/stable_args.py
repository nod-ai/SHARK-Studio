import argparse
import os
from pathlib import Path

from apps.stable_diffusion.src.utils.resamplers import resampler_list


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
# Stable Diffusion Params
##############################################################################

p.add_argument(
    "-a",
    "--app",
    default="txt2img",
    help="Which app to use, one of: txt2img, img2img, outpaint, inpaint.",
)
p.add_argument(
    "-p",
    "--prompts",
    nargs="+",
    default=[
        "a photo taken of the front of a super-car drifting on a road near "
        "mountains at high speeds with smokes coming off the tires, front "
        "angle, front point of view, trees in the mountains of the "
        "background, ((sharp focus))"
    ],
    help="Text of which images to be generated.",
)

p.add_argument(
    "--negative_prompts",
    nargs="+",
    default=[
        "watermark, signature, logo, text, lowres, ((monochrome, grayscale)), "
        "blurry, ugly, blur, oversaturated, cropped"
    ],
    help="Text you don't want to see in the generated image.",
)

p.add_argument(
    "--img_path",
    type=str,
    help="Path to the image input for img2img/inpainting.",
)

p.add_argument(
    "--steps",
    type=int,
    default=50,
    help="The number of steps to do the sampling.",
)

p.add_argument(
    "--seed",
    type=str,
    default=-1,
    help="The seed or list of seeds to use. -1 for a random one.",
)

p.add_argument(
    "--batch_size",
    type=int,
    default=1,
    choices=range(1, 4),
    help="The number of inferences to be made in a single `batch_count`.",
)

p.add_argument(
    "--height",
    type=int,
    default=512,
    choices=range(128, 769, 8),
    help="The height of the output image.",
)

p.add_argument(
    "--width",
    type=int,
    default=512,
    choices=range(128, 769, 8),
    help="The width of the output image.",
)

p.add_argument(
    "--guidance_scale",
    type=float,
    default=7.5,
    help="The value to be used for guidance scaling.",
)

p.add_argument(
    "--noise_level",
    type=int,
    default=20,
    help="The value to be used for noise level of upscaler.",
)

p.add_argument(
    "--max_length",
    type=int,
    default=64,
    help="Max length of the tokenizer output, options are 64 and 77.",
)

p.add_argument(
    "--max_embeddings_multiples",
    type=int,
    default=5,
    help="The max multiple length of prompt embeddings compared to the max "
    "output length of text encoder.",
)

p.add_argument(
    "--strength",
    type=float,
    default=0.8,
    help="The strength of change applied on the given input image for "
    "img2img.",
)

p.add_argument(
    "--use_hiresfix",
    type=bool,
    default=False,
    help="Use Hires Fix to do higher resolution images, while trying to "
    "avoid the issues that come with it. This is accomplished by first "
    "generating an image using txt2img, then running it through img2img.",
)

p.add_argument(
    "--hiresfix_height",
    type=int,
    default=768,
    choices=range(128, 769, 8),
    help="The height of the Hires Fix image.",
)

p.add_argument(
    "--hiresfix_width",
    type=int,
    default=768,
    choices=range(128, 769, 8),
    help="The width of the Hires Fix image.",
)

p.add_argument(
    "--hiresfix_strength",
    type=float,
    default=0.6,
    help="The denoising strength to apply for the Hires Fix.",
)

p.add_argument(
    "--resample_type",
    type=str,
    default="Nearest Neighbor",
    choices=resampler_list,
    help="The resample type to use when resizing an image before being run "
    "through stable diffusion.",
)

##############################################################################
# Stable Diffusion Training Params
##############################################################################

p.add_argument(
    "--lora_save_dir",
    type=str,
    default="models/lora/",
    help="Directory to save the lora fine tuned model.",
)

p.add_argument(
    "--training_images_dir",
    type=str,
    default="models/lora/training_images/",
    help="Directory containing images that are an example of the prompt.",
)

p.add_argument(
    "--training_steps",
    type=int,
    default=2000,
    help="The number of steps to train.",
)

##############################################################################
# Inpainting and Outpainting Params
##############################################################################

p.add_argument(
    "--mask_path",
    type=str,
    help="Path to the mask image input for inpainting.",
)

p.add_argument(
    "--inpaint_full_res",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If inpaint only masked area or whole picture.",
)

p.add_argument(
    "--inpaint_full_res_padding",
    type=int,
    default=32,
    choices=range(0, 257, 4),
    help="Number of pixels for only masked padding.",
)

p.add_argument(
    "--pixels",
    type=int,
    default=128,
    choices=range(8, 257, 8),
    help="Number of expended pixels for one direction for outpainting.",
)

p.add_argument(
    "--mask_blur",
    type=int,
    default=8,
    choices=range(0, 65),
    help="Number of blur pixels for outpainting.",
)

p.add_argument(
    "--left",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If extend left for outpainting.",
)

p.add_argument(
    "--right",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If extend right for outpainting.",
)

p.add_argument(
    "--up",
    "--top",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If extend top for outpainting.",
)

p.add_argument(
    "--down",
    "--bottom",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If extend bottom for outpainting.",
)

p.add_argument(
    "--noise_q",
    type=float,
    default=1.0,
    help="Fall-off exponent for outpainting (lower=higher detail) "
    "(min=0.0, max=4.0).",
)

p.add_argument(
    "--color_variation",
    type=float,
    default=0.05,
    help="Color variation for outpainting (min=0.0, max=1.0).",
)

##############################################################################
# Model Config and Usage Params
##############################################################################

p.add_argument(
    "--device", type=str, default="vulkan", help="Device to run the model."
)

p.add_argument(
    "--precision", type=str, default="fp16", help="Precision to run the model."
)

p.add_argument(
    "--import_mlir",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Imports the model from torch module to shark_module otherwise "
    "downloads the model from shark_tank.",
)

p.add_argument(
    "--load_vmfb",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Attempts to load the model from a precompiled flat-buffer "
    "and compiles + saves it if not found.",
)

p.add_argument(
    "--save_vmfb",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Saves the compiled flat-buffer to the local directory.",
)

p.add_argument(
    "--use_tuned",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Download and use the tuned version of the model if available.",
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
    help="Other supported schedulers are [DDIM, PNDM, LMSDiscrete, "
    "DPMSolverMultistep, DPMSolverMultistep++, DPMSolverMultistepKarras, "
    "DPMSolverMultistepKarras++, EulerDiscrete, EulerAncestralDiscrete, "
    "DEISMultistep, KDPM2AncestralDiscrete, DPMSolverSinglestep, DDPM, "
    "HeunDiscrete].",
)

p.add_argument(
    "--output_img_format",
    type=str,
    default="png",
    help="Specify the format in which output image is save. "
    "Supported options: jpg / png.",
)

p.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory path to save the output images and json.",
)

p.add_argument(
    "--batch_count",
    type=int,
    default=1,
    help="Number of batches to be generated with random seeds in "
    "single execution.",
)

p.add_argument(
    "--repeatable_seeds",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="The seed of the first batch will be used as the rng seed to "
    "generate the subsequent seeds for subsequent batches in that run.",
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
    help="HuggingFace repo-id or path to SD model's checkpoint whose VAE "
    "needs to be plugged in.",
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
    help="Use the accelerate package to reduce cpu memory consumption.",
)

p.add_argument(
    "--attention_slicing",
    type=str,
    default="none",
    help="Amount of attention slicing to use (one of 'max', 'auto', 'none', "
    "or an integer).",
)

p.add_argument(
    "--use_stencil",
    choices=["canny", "openpose", "scribble", "zoedepth"],
    help="Enable the stencil feature.",
)

p.add_argument(
    "--use_lora",
    type=str,
    default="",
    help="Use standalone LoRA weight using a HF ID or a checkpoint "
    "file (~3 MB).",
)

p.add_argument(
    "--use_quantize",
    type=str,
    default="none",
    help="Runs the quantized version of stable diffusion model. "
    "This is currently in experimental phase. "
    "Currently, only runs the stable-diffusion-2-1-base model in "
    "int8 quantization.",
)

p.add_argument(
    "--ondemand",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Load and unload models for low VRAM.",
)

p.add_argument(
    "--hf_auth_token",
    type=str,
    default=None,
    help="Specify your own huggingface authentication tokens for models like Llama2.",
)

p.add_argument(
    "--device_allocator_heap_key",
    type=str,
    default="",
    help="Specify heap key for device caching allocator."
    "Expected form: max_allocation_size;max_allocation_capacity;max_free_allocation_count"
    "Example: --device_allocator_heap_key='*;1gib' (will limit caching on device to 1 gigabyte)",
)
##############################################################################
# IREE - Vulkan supported flags
##############################################################################

p.add_argument(
    "--iree_vulkan_target_triple",
    type=str,
    default="",
    help="Specify target triple for vulkan.",
)

p.add_argument(
    "--iree_metal_target_platform",
    type=str,
    default="",
    help="Specify target triple for metal.",
)

##############################################################################
# Misc. Debug and Optimization flags
##############################################################################

p.add_argument(
    "--use_compiled_scheduler",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Use the default scheduler precompiled into the model if available.",
)

p.add_argument(
    "--local_tank_cache",
    default="",
    help="Specify where to save downloaded shark_tank artifacts. "
    "If this is not set, the default is ~/.local/shark_tank/.",
)

p.add_argument(
    "--dump_isa",
    default=False,
    action="store_true",
    help="When enabled call amdllpc to get ISA dumps. "
    "Use with dispatch benchmarks.",
)

p.add_argument(
    "--dispatch_benchmarks",
    default=None,
    help="Dispatches to return benchmark data on. "
    'Use "All" for all, and None for none.',
)

p.add_argument(
    "--dispatch_benchmarks_dir",
    default="temp_dispatch_benchmarks",
    help="Directory where you want to store dispatch data "
    'generated with "--dispatch_benchmarks".',
)

p.add_argument(
    "--enable_rgp",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag for inserting debug frames between iterations "
    "for use with rgp.",
)

p.add_argument(
    "--hide_steps",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Flag for hiding the details of iteration/sec for each step.",
)

p.add_argument(
    "--warmup_count",
    type=int,
    default=0,
    help="Flag setting warmup count for CLIP and VAE [>= 0].",
)

p.add_argument(
    "--clear_all",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag to clear all mlir and vmfb from common locations. "
    "Recompiling will take several minutes.",
)

p.add_argument(
    "--save_metadata_to_json",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag for whether or not to save a generation information "
    "json file with the image.",
)

p.add_argument(
    "--write_metadata_to_png",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Flag for whether or not to save generation information in "
    "PNG chunk text to generated images.",
)

p.add_argument(
    "--import_debug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="If import_mlir is True, saves mlir via the debug option "
    "in shark importer. Does nothing if import_mlir is false (the default).",
)

p.add_argument(
    "--compile_debug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag to toggle debug assert/verify flags for imported IR in the"
    "iree-compiler. Default to false.",
)

p.add_argument(
    "--iree_constant_folding",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Controls constant folding in iree-compile for all SD models.",
)

##############################################################################
# Web UI flags
##############################################################################

p.add_argument(
    "--progress_bar",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Flag for removing the progress bar animation during "
    "image generation.",
)

p.add_argument(
    "--ckpt_dir",
    type=str,
    default="",
    help="Path to directory where all .ckpts are stored in order to populate "
    "them in the web UI.",
)
# TODO: replace API flag when these can be run together
p.add_argument(
    "--ui",
    type=str,
    default="app" if os.name == "nt" else "web",
    help="One of: [api, app, web].",
)

p.add_argument(
    "--share",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag for generating a public URL.",
)

p.add_argument(
    "--server_port",
    type=int,
    default=8080,
    help="Flag for setting server port.",
)

p.add_argument(
    "--api",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag for enabling rest API.",
)

p.add_argument(
    "--api_accept_origin",
    action="append",
    type=str,
    help="An origin to be accepted by the REST api for Cross Origin"
    "Resource Sharing (CORS). Use multiple times for multiple origins, "
    'or use --api_accept_origin="*" to accept all origins. If no origins '
    "are set no CORS headers will be returned by the api. Use, for "
    "instance, if you need to access the REST api from Javascript running "
    "in a web browser.",
)

p.add_argument(
    "--debug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag for enabling debugging log in WebUI.",
)

p.add_argument(
    "--output_gallery",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Flag for removing the output gallery tab, and avoid exposing "
    "images under --output_dir in the UI.",
)

p.add_argument(
    "--output_gallery_followlinks",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Flag for whether the output gallery tab in the UI should "
    "follow symlinks when listing subdirectories under --output_dir.",
)


##############################################################################
# SD model auto-annotation flags
##############################################################################

p.add_argument(
    "--annotation_output",
    type=path_expand,
    default="./",
    help="Directory to save the annotated mlir file.",
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
    help="Save annotated mlir file.",
)
##############################################################################
# SD model auto-tuner flags
##############################################################################

p.add_argument(
    "--tuned_config_dir",
    type=path_expand,
    default="./",
    help="Directory to save the tuned config file.",
)

p.add_argument(
    "--num_iters",
    type=int,
    default=400,
    help="Number of iterations for tuning.",
)

p.add_argument(
    "--search_op",
    type=str,
    default="all",
    help="Op to be optimized, options are matmul, bmm, conv and all.",
)

##############################################################################
# DocuChat Flags
##############################################################################

p.add_argument(
    "--run_docuchat_web",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Specifies whether the docuchat's web version is running or not.",
)

##############################################################################
# rocm Flags
##############################################################################

p.add_argument(
    "--iree_rocm_target_chip",
    type=str,
    default="",
    help="Add the rocm device architecture ex gfx1100, gfx90a, etc. Use `hipinfo` "
    "or `iree-run-module --dump_devices=rocm` or `hipinfo` to get desired arch name",
)

args, unknown = p.parse_known_args()
if args.import_debug:
    os.environ["IREE_SAVE_TEMPS"] = os.path.join(
        os.getcwd(), args.hf_model_id.replace("/", "_")
    )
