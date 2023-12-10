from turbine_models.custom_models.sd_inference import clip, unet, vae
from shark.iree_utils.compile_utils import get_iree_compiled_module
from apps.shark_studio.api.utils import get_resource_path
import iree.runtime as ireert
import gc
import torch

sd_model_map = {
    "CompVis/stable-diffusion-v1-4": {
        "clip": {
            "initializer": clip.export_clip_model,
            "max_tokens": 64,
        },
        "vae_encode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
        "unet": {
            "initializer": unet.export_unet_model,
            "max_tokens": 512,
        },
        "vae_decode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
    },
    "runwayml/stable-diffusion-v1-5": {
        "clip": {
            "initializer": clip.export_clip_model,
            "max_tokens": 64,
        },
        "vae_encode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
        "unet": {
            "initializer": unet.export_unet_model,
            "max_tokens": 512,
        },
        "vae_decode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
    },
    "stabilityai/stable-diffusion-2-1-base": {
        "clip": {
            "initializer": clip.export_clip_model,
            "max_tokens": 64,
        },
        "vae_encode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
        "unet": {
            "initializer": unet.export_unet_model,
            "max_tokens": 512,
        },
        "vae_decode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
    },
    "stabilityai/stable_diffusion-xl-1.0": {
        "clip_1": {
            "initializer": clip.export_clip_model,
            "max_tokens": 64,
        },
        "clip_2": {
            "initializer": clip.export_clip_model,
            "max_tokens": 64,
        },
        "vae_encode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
        "unet": {
            "initializer": unet.export_unet_model,
            "max_tokens": 512,
        },
        "vae_decode": {
            "initializer": vae.export_vae_model,
            "max_tokens": 64,
        },
    },
}


class StableDiffusion(SharkPipelineBase):

    # This class is responsible for executing image generation and creating
    # /managing a set of compiled modules to run Stable Diffusion. The init
    # aims to be as general as possible, and the class will infer and compile
    # a list of necessary modules or a combined "pipeline module" for a
    # specified job based on the inference task.
    # 
    # custom_model_ids: a dict of submodel + HF ID pairs for custom submodels.
    # e.g. {"vae_decode": "madebyollin/sdxl-vae-fp16-fix"}
    # 
    # embeddings: a dict of embedding checkpoints or model IDs to use when
    # initializing the compiled modules.

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        height: int = 512,
        width: int = 512,
        precision: str = "fp16",
        device: str = None,
        custom_model_map: dict = {},
        custom_weights_map: dict = {},
        embeddings: dict = {},
        import_ir: bool = True,
    ):
        super().__init__(sd_model_map[base_model_id], device, import_ir)
        self.base_model_id = base_model_id
        self.device = device
        self.precision = precision
        self.iree_module_dict = None
        self.get_compiled_map()


    def generate_images(
            self,
            prompt,
            ):
        return result_output,

if __name__ == "__main__":
    sd = StableDiffusion(
        "runwayml/stable-diffusion-v1-5",
        device="vulkan",
    )
    print("model loaded")
