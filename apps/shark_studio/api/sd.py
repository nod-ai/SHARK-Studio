import gc
import torch
import time
import os
import json
import numpy as np

from pathlib import Path
from turbine_models.custom_models.sd_inference import clip, unet, vae
from apps.shark_studio.api.controlnet import control_adapter_map
from apps.shark_studio.web.utils.state import status_label
from apps.shark_studio.web.utils.file_utils import safe_name, get_resource_path, get_checkpoints_path
from apps.shark_studio.modules.pipeline import SharkPipelineBase
from apps.shark_studio.modules.schedulers import get_schedulers
from apps.shark_studio.modules.prompt_encoding import get_weighted_text_embeddings
from apps.shark_studio.modules.img_processing import (
    resize_stencil,
    save_output_img,
)

from apps.shark_studio.modules.ckpt_processing import (
    preprocessCKPT,
    process_custom_pipe_weights,
)
from transformers import CLIPTokenizer
from math import ceil
from PIL import Image

sd_model_map = {
    "clip": {
        "initializer": clip.export_clip_model,
        "external_weight_file": None,
        "ireec_flags": ["--iree-flow-collapse-reduction-dims",
                        "--iree-opt-const-expr-hoisting=False",
                        "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
        ],
    },
    "vae_encode": {
        "initializer": vae.export_vae_model,
        "external_weight_file": None,
    },
    "unet": {
        "initializer": unet.export_unet_model,
        "ireec_flags": ["--iree-flow-collapse-reduction-dims",
                        "--iree-opt-const-expr-hoisting=False",
                        "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
        ],
        "external_weight_file": None,
    },
    "vae_decode": {
        "initializer": vae.export_vae_model,
        "external_weight_file": None,
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
        base_model_id,
        height: int,
        width: int,
        batch_size: int,
        precision: str,
        device: str,
        custom_vae: str = None,
        num_loras: int = 0,
        import_ir: bool = True,
        is_img2img: bool = False,
        is_controlled: bool = False,
    ):
        self.model_max_length = 77
        self.batch_size = batch_size
        self.precision = precision
        self.is_img2img = is_img2img
        self.scheduler_obj = {}
        self.precision = precision
        static_kwargs = {
            "pipe": {},
            "clip": {"hf_model_name": base_model_id},
            "unet": {
                "hf_model_name": base_model_id,
                "unet_model": unet.UnetModel(hf_model_name=base_model_id, hf_auth_token=None),
                "batch_size": batch_size,
                #"is_controlled": is_controlled,
                #"num_loras": num_loras,
                "height": height,
                "width": width,
            },
            "vae_encode": {
                "hf_model_name": custom_vae if custom_vae else base_model_id,
                "vae_model": vae.VaeModel(hf_model_name=base_model_id, hf_auth_token=None),
                "batch_size": batch_size,
                "height": height,
                "width": width,
            },
            "vae_decode": {
                "hf_model_name": custom_vae,
                "vae_model": vae.VaeModel(hf_model_name=base_model_id, hf_auth_token=None),
                "batch_size": batch_size,
                "height": height,
                "width": width,
            },
        }
        super().__init__(
            sd_model_map, base_model_id, static_kwargs, device, import_ir
        )
        pipe_id_list = [
            safe_name(base_model_id),
            str(batch_size),
            f"{str(height)}x{str(width)}",
            precision,
        ]
        if num_loras > 0:
            pipe_id_list.append(str(num_loras)+"lora")
        if is_controlled:
            pipe_id_list.append("controlled")
        if custom_vae:
            pipe_id_list.append(custom_vae)
        self.pipe_id = "_".join(pipe_id_list)
        print(f"\n[LOG] Pipeline initialized with pipe_id: {self.pipe_id}.")
        del static_kwargs
        gc.collect()


    def prepare_pipe(self, scheduler, custom_weights, adapters, embeddings):
        print(
            f"\n[LOG] Preparing pipeline with scheduler {scheduler}"
            f"\n[LOG] Custom embeddings currently unsupported."
        )
        schedulers = get_schedulers(self.base_model_id)
        self.weights_path = get_checkpoints_path(self.pipe_id)
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)
        # accepting a list of schedulers in batched cases.
        for i in scheduler:
            self.scheduler_obj[i] = schedulers[i]
            print(f"[LOG] Loaded scheduler: {i}")
        for model in adapters:
            self.model_map[model] = adapters[model]
        if os.path.isfile(custom_weights):
            for i in self.model_map:
               self.model_map[i]["external_weights_file"] = None
        elif custom_weights != "":
            print(f"\n[LOG][WARNING] Custom weights were not found at {custom_weights}. Did you mean to pass a base model ID?")
        self.static_kwargs["pipe"] = {
        #    "external_weight_path": self.weights_path,
#            "external_weights": "safetensors",
        }
        self.get_compiled_map(pipe_id=self.pipe_id)
        print("\n[LOG] Pipeline successfully prepared for runtime.")
        return


    def encode_prompts_weight(
        self,
        prompt,
        negative_prompt,
        do_classifier_free_guidance=True,
    ):
        # Encodes the prompt into text encoder hidden states.
        self.load_submodels(["clip"])
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_model_id,
            subfolder="tokenizer",
        )
        clip_inf_start = time.time()


        text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
            pipe=self,
            prompt=prompt,
            uncond_prompt=negative_prompt
            if do_classifier_free_guidance
            else None,
        )

        if do_classifier_free_guidance:
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        pad = (0, 0) * (len(text_embeddings.shape) - 2)
        pad = pad + (0, 512 - text_embeddings.shape[1])
        text_embeddings = torch.nn.functional.pad(text_embeddings, pad)

        # SHARK: Report clip inference time
        clip_inf_time = (time.time() - clip_inf_start) * 1000
        if self.ondemand:
            self.unload_clip()
            gc.collect()
        print(f"\n[LOG] Clip Inference time (ms) = {clip_inf_time:.3f}")

        return text_embeddings.numpy().astype(np.float16)
        

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
        batched_args=[
            prompt,
            negative_prompt,
            steps,
            strength,
            guidance_scale,
            seed,
            resample_type,
            control_mode,
            hints,
        ]
        for arg in batched_args:
            if not isinstance(arg, list):
                arg = [arg] * self.batch_size
            if len(arg) < self.batch_size:
                arg = arg * self.batch_size
            else:
                arg = [arg[i] for i in range(self.batch_size)]

        text_embeddings = self.encode_prompts_weight(
            prompt,
            negative_prompt,
        )
        print(text_embeddings)
        test_img = [
            Image.open(
                get_resource_path("../../tests/jupiter.png"), mode="r"
            ).convert("RGB")
        ] * self.batch_size
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

    adapters = {}
    is_controlled = False
    control_mode = None
    hints = []
    num_loras = 0
    for i in embeddings:
        num_loras += 1 if embeddings[i] else 0
    if "model" in controlnets:
        for i, model in enumerate(controlnets["model"]):
            if "xl" not in base_model_id.lower():
                adapters[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map[
                        "runwayml/stable-diffusion-v1-5"
                    ][model],
                    "strength": controlnets["strength"][i],
                }
            else:
                adapters[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map[
                        "stabilityai/stable-diffusion-xl-1.0"
                    ][model],
                    "strength": controlnets["strength"][i],
                }
            if model is not None:
                is_controlled=True
        control_mode = controlnets["control_mode"]
        for i in controlnets["hint"]:
            hints.append[i]

    submit_pipe_kwargs = {
        "base_model_id": base_model_id,
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "precision": precision,
        "device": device,
        "custom_vae": custom_vae,
        "num_loras": num_loras,
        "import_ir": cmd_opts.import_mlir,
        "is_img2img": is_img2img,
        "is_controlled": is_controlled,
    }
    submit_prep_kwargs = {
        "scheduler": scheduler,
        "custom_weights": custom_weights,
        "adapters": adapters,
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
    print(submit_pipe_kwargs)
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


def view_json_file(file_path):
    content = ""
    with open(file_path, "r") as fopen:
        content = fopen.read()
    return content


if __name__ == "__main__":
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj 
    global_obj._init()

    sd_json = view_json_file(get_resource_path("../configs/default_sd_config.json"))
    sd_kwargs = json.loads(sd_json)
    for i in shark_sd_fn_dict_input(sd_kwargs):
        print(i)