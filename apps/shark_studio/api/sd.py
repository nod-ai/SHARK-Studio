from turbine_models.custom_models.sd_inference import clip, unet, vae
from shark.iree_utils.compile_utils import get_iree_compiled_module
from apps.shark_studio.api.utils import get_resource_path
from apps.shark_studio.api.controlnet import control_adapter_map
from apps.shark_studio.web.utils.state import status_label
from apps.shark_studio.modules.pipeline import SharkPipelineBase
import iree.runtime as ireert
import gc
import torch
import gradio as gr

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
        embeddings: dict = {},
        import_ir: bool = True,
    ):
        super().__init__(sd_model_map[base_model_id], device, import_ir)
        self.base_model_id = base_model_id
        self.device = device
        self.precision = precision
        self.iree_module_dict = None
        self.get_compiled_map()

    def prepare_pipeline(self, scheduler, custom_model_map):
        return None

    def generate_images(
        self,
        prompt,
        negative_prompt,
        steps,
        strength,
        guidance_scale,
        seed,
        ondemand,
        repeatable_seeds,
        resample_type,
        control_mode,
        preprocessed_hints,
    ):
        return None, None, None, None, None


# NOTE: Each `hf_model_id` should have its own starting configuration.

# model_vmfb_key = ""


def shark_sd_fn(
    prompt,
    negative_prompt,
    image_dict,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: str | int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    base_model_id: str,
    custom_weights: str,
    custom_vae: str,
    precision: str,
    device: str,
    lora_weights: str | list,
    ondemand: bool,
    repeatable_seeds: bool,
    resample_type: str,
    control_mode: str,
    stencils: list,
    images: list,
    preprocessed_hints: list,
    progress=gr.Progress(),
):
    # Handling gradio ImageEditor datatypes so we have unified inputs to the SD API
    for i, stencil in enumerate(stencils):
        if images[i] is None and stencil is not None:
            continue
        elif stencil is None and any(
            img is not None for img in [images[i], preprocessed_hints[i]]
        ):
            images[i] = None
            preprocessed_hints[i] = None
        elif images[i] is not None:
            if isinstance(images[i], dict):
                images[i] = images[i]["composite"]
            images[i] = images[i].convert("RGB")

    if isinstance(image_dict, PIL.Image.Image):
        image = image_dict.convert("RGB")
    elif image_dict:
        image = image_dict["image"].convert("RGB")
    else:
        image = None
        is_img2img = False
    if image:
        (
            image,
            _,
            _,
        ) = resize_stencil(image, width, height)
        is_img2img = True
    print("Performing Stable Diffusion Pipeline setup...")

    device_id = None

    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    custom_model_map = {}
    if custom_weights != "None":
        custom_model_map["unet"] = {"custom_weights": custom_weights}
    if custom_vae != "None":
        custom_model_map["vae"] = {"custom_weights": custom_vae}
    if stencils:
        for i, stencil in enumerate(stencils):
            if "xl" not in base_model_id.lower():
                custom_model_map[f"control_adapter_{i}"] = stencil_adapter_map[
                    "runwayml/stable-diffusion-v1-5"
                ][stencil]
            else:
                custom_model_map[f"control_adapter_{i}"] = stencil_adapter_map[
                    "stabilityai/stable-diffusion-xl-1.0"
                ][stencil]

    submit_pipe_kwargs = {
        "base_model_id": base_model_id,
        "height": height,
        "width": width,
        "precision": precision,
        "device": device,
        "custom_model_map": custom_model_map,
        "import_ir": cmd_opts.import_mlir,
        "is_img2img": is_img2img,
    }
    submit_prep_kwargs = {
        "scheduler": scheduler,
        "custom_model_map": custom_model_map,
        "embeddings": lora_weights,
    }
    submit_run_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "ondemand": ondemand,
        "repeatable_seeds": repeatable_seeds,
        "resample_type": resample_type,
        "control_mode": control_mode,
        "preprocessed_hints": preprocessed_hints,
    }

    global sd_pipe
    global sd_pipe_kwargs

    if sd_pipe_kwargs and sd_pipe_kwargs != submit_pipe_kwargs:
        sd_pipe = None
        sd_pipe_kwargs = submit_pipe_kwargs
        gc.collect()

    if sd_pipe is None:
        history[-1][-1] = "Getting the pipeline ready..."
        yield history, ""

        # Initializes the pipeline and retrieves IR based on all
        # parameters that are static in the turbine output format,
        # which is currently MLIR in the torch dialect.

        sd_pipe = SharkStableDiffusionPipeline(
            **submit_pipe_kwargs,
        )

    sd_pipe.prepare_pipe(**submit_prep_kwargs)

    for prompt, msg, exec_time in progress.tqdm(
        out_imgs=sd_pipe.generate_images(**submit_run_kwargs),
        desc="Generating Image...",
    ):
        text_output = get_generation_text_info(
            seeds[: current_batch + 1], device
        )
        save_output_img(
            out_imgs[0],
            seeds[current_batch],
            extra_info,
        )
        generated_imgs.extend(out_imgs)
        yield generated_imgs, text_output, status_label(
            "Stable Diffusion", current_batch + 1, batch_count, batch_size
        ), stencils, images

    return generated_imgs, text_output, "", stencils, images


def cancel_sd():
    print("Inject call to cancel longer API calls.")
    return


if __name__ == "__main__":
    sd = StableDiffusion(
        "runwayml/stable-diffusion-v1-5",
        device="vulkan",
    )
    print("model loaded")
