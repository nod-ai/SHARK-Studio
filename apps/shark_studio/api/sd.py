import gc
from unittest import registerResult
import torch
import time
import os
import json

from turbine_models.custom_models.sd_inference import clip, unet, vae
from apps.shark_studio.api.controlnet import control_adapter_map
from apps.shark_studio.web.utils.state import status_label
from apps.shark_studio.web.utils.file_utils import safe_name, get_resource_path
from apps.shark_studio.modules.pipeline import SharkPipelineBase
from apps.shark_studio.modules.img_processing import (
    resize_stencil,
    save_output_img,
)
from apps.shark_studio.modules.ckpt_processing import (
    process_custom_pipe_weights,
)
from math import ceil
from PIL import Image

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


def get_spec(custom_sd_map: dict, sd_embeds: dict):
    spec = []
    for key in custom_sd_map:
        if "control" in key.split("_"):
            spec.append("controlled")
        elif key == "custom_vae":
            spec.append(custom_sd_map[key]["custom_weights"].split(".")[0])
    num_embeds = 0
    embeddings_spec = None
    for embed in sd_embeds:
        if embed is not None:
            num_embeds += 1
            embeddings_spec = str(num_embeds) + "embeds"
    if embeddings_spec:
        spec.append(embeddings_spec)
    return "_".join(spec)


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
        is_img2img: bool = False,
    ):
        super().__init__(
            sd_model_map[base_model_id], base_model_id, device, import_ir
        )
        self.precision = precision
        self.is_img2img = is_img2img
        self.pipe_id = (
            safe_name(base_model_id)
            + str(height)
            + str(width)
            + precision
            + device
            + get_spec(custom_model_map, embeddings)
        )
        print(f"\n[LOG] Pipeline initialized with pipe_id: {self.pipe_id}")

    def prepare_pipe(self, scheduler, custom_model_map, embeddings):
        print(
            f"\n[LOG] Preparing pipeline with scheduler {scheduler}, custom map {json.dumps(custom_model_map)}, and embeddings {json.dumps(embeddings)}."
        )
        self.get_compiled_map(device=self.device, pipe_id=self.pipe_id)
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
        hints,
    ):
        print("\n[LOG] Generating images...")
        test_img = [
            Image.open(
                get_resource_path("../../tests/jupiter.png"), mode="r"
            ).convert("RGB")
        ]
        return test_img


def shark_sd_fn_dict_input(
    sd_kwargs: dict,
):
    print("[LOG] Submitting Request...")
    input_imgs = []
    img_paths = sd_kwargs["sd_init_image"]

    for img_path in img_paths:
        if img_path:
            if os.path.isfile(img_path):
                input_imgs.append(
                    Image.open(img_path, mode="r").convert("RGB")
                )
    sd_kwargs["sd_init_image"] = input_imgs
    # result = shark_sd_fn(**sd_kwargs)
    # for i in range(sd_kwargs["batch_count"]):
    #    yield from result
    # return result
    for i in range(1):
        generated_imgs = yield from shark_sd_fn(**sd_kwargs)
        yield generated_imgs
    return generated_imgs


def shark_sd_fn(
    prompt,
    negative_prompt,
    sd_init_image,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: list,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    base_model_id: str,
    custom_weights: str,
    custom_vae: str,
    precision: str,
    device: str,
    ondemand: bool,
    repeatable_seeds: bool,
    resample_type: str,
    controlnets: dict,
    embeddings: dict,
):
    sd_kwargs = locals()
    if isinstance(sd_init_image, Image.Image):
        image = sd_init_image.convert("RGB")
    elif sd_init_image:
        image = sd_init_image["image"].convert("RGB")
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
    print("\n[LOG] Performing Stable Diffusion Pipeline setup...")

    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    custom_model_map = {}
    control_mode = None
    hints = []
    if custom_weights != "None":
        custom_model_map["unet"] = {"custom_weights": custom_weights}
    if custom_vae != "None":
        custom_model_map["vae"] = {"custom_weights": custom_vae}
    if "model" in controlnets:
        for i, model in enumerate(controlnets["model"]):
            if "xl" not in base_model_id.lower():
                custom_model_map[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map[
                        "runwayml/stable-diffusion-v1-5"
                    ][model],
                    "strength": controlnets["strength"][i],
                }
            else:
                custom_model_map[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map[
                        "stabilityai/stable-diffusion-xl-1.0"
                    ][model],
                    "strength": controlnets["strength"][i],
                }
        control_mode = controlnets["control_mode"]
        for i in controlnets["hint"]:
            hints.append[i]

    submit_pipe_kwargs = {
        "base_model_id": base_model_id,
        "height": height,
        "width": width,
        "precision": precision,
        "device": device,
        "custom_model_map": custom_model_map,
        "embeddings": embeddings,
        "import_ir": cmd_opts.import_mlir,
        "is_img2img": is_img2img,
    }
    submit_prep_kwargs = {
        "scheduler": scheduler,
        "custom_model_map": custom_model_map,
        "embeddings": embeddings,
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
        "hints": hints,
    }
    if (
        not global_obj.get_sd_obj()
        or global_obj.get_pipe_kwargs() != submit_pipe_kwargs
    ):
        print("\n[LOG] Initializing new pipeline...")
        global_obj.clear_cache()
        gc.collect()
        global_obj.set_pipe_kwargs(submit_pipe_kwargs)

        # Initializes the pipeline and retrieves IR based on all
        # parameters that are static in the turbine output format,
        # which is currently MLIR in the torch dialect.

        sd_pipe = StableDiffusion(
            **submit_pipe_kwargs,
        )
        global_obj.set_sd_obj(sd_pipe)

    global_obj.get_sd_obj().prepare_pipe(**submit_prep_kwargs)
    generated_imgs = []
    for current_batch in range(batch_count):
        start_time = time.time()
        out_imgs = global_obj.get_sd_obj().generate_images(**submit_run_kwargs)
        total_time = time.time() - start_time
        text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        print(f"\n[LOG] {text_output}")
        # if global_obj.get_sd_status() == SD_STATE_CANCEL:
        #     break
        # else:
        try:
            this_seed = seed[current_batch]
        except:
            this_seed = seed[0]
        save_output_img(
            out_imgs[0],
            this_seed,
            sd_kwargs,
        )
        generated_imgs.extend(out_imgs)
        yield generated_imgs, status_label(
            "Stable Diffusion", current_batch + 1, batch_count, batch_size
        )
    return generated_imgs, ""


def cancel_sd():
    print("Inject call to cancel longer API calls.")
    return


if __name__ == "__main__":
    sd = StableDiffusion(
        "runwayml/stable-diffusion-v1-5",
        device="vulkan",
    )
    print("model loaded")
