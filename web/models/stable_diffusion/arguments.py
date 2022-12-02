from transformers import CLIPTokenizer
from diffusers import (
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)
from models.stable_diffusion.opt_params import get_unet, get_vae, get_clip
from models.stable_diffusion.utils import set_iree_runtime_flags
import os


class Arguments:
    def __init__(
        self,
        prompt: str = "an astronaut riding a horse",
        scheduler: str = "LMS",
        iteration_count: int = 1,
        batch_size: int = 1,
        steps: int = 50,
        guidance: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: int = 42,
        precision: str = "fp16",
        device: str = "cpu",
        cache: bool = True,
        iree_vulkan_target_triple: str = "",
        live_preview: bool = False,
        save_img: bool = False,
        import_mlir: bool = False,
        max_length: int = 77,
        use_tuned: bool = False,
        vulkan_large_heap_block_size: int = 4294967296,
        version: str = "v1.4",
    ):
        self.prompt = prompt
        self.scheduler = scheduler
        self.iteration_count = iteration_count
        self.batch_size = batch_size
        self.steps = steps
        self.guidance = guidance
        self.height = height
        self.width = width
        self.seed = seed
        self.precision = precision
        self.device = device
        self.cache = cache
        self.iree_vulkan_target_triple = iree_vulkan_target_triple
        self.live_preview = live_preview
        self.save_img = save_img
        self.import_mlir = import_mlir
        self.max_length = max_length
        self.use_tuned = use_tuned
        self.vulkan_large_heap_block_size = vulkan_large_heap_block_size
        self.version = version

    def set_params(
        self,
        prompt: str,
        scheduler: str,
        iteration_count: int,
        batch_size: int,
        steps: int,
        guidance: float,
        height: int,
        width: int,
        seed: int,
        precision: str,
        device: str,
        cache: bool,
        iree_vulkan_target_triple: str,
        live_preview: bool,
        save_img: bool,
        import_mlir: bool,
    ):
        self.prompt = prompt
        self.scheduler = scheduler
        self.iteration_count = iteration_count
        self.batch_size = batch_size
        self.steps = steps
        self.guidance = guidance
        self.height = height
        self.width = width
        self.seed = seed
        self.precision = precision
        self.device = device
        self.cache = cache
        self.iree_vulkan_target_triple = iree_vulkan_target_triple
        self.live_preview = live_preview
        self.save_img = save_img
        self.import_mlir = import_mlir


output_dir = "./stored_results/stable_diffusion"
os.makedirs(output_dir, exist_ok=True)

schedulers = dict()
# set scheduler value
schedulers["PNDM"] = PNDMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)
schedulers["LMSDiscrete"] = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)
schedulers["DDIM"] = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
)
schedulers["DPMSolverMultistep"] = DPMSolverMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="scheduler",
)

cache_obj = dict()
# cache tokenizer and text_encoder
cache_obj["tokenizer"] = CLIPTokenizer.from_pretrained(
    "openai/clip-vit-large-patch14"
)

# cache vae and unet.
args = Arguments()
args.device = "vulkan"
set_iree_runtime_flags(args)
(
    cache_obj["vae_fp16_vulkan"],
    cache_obj["unet_fp16_vulkan"],
    cache_obj["clip_vulkan"],
) = (get_vae(args), get_unet(args), get_clip(args))
