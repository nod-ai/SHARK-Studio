import gc
import torch
import time
import os
import json
import numpy as np
import copy
from tqdm.auto import tqdm

from pathlib import Path
from random import randint
from turbine_models.custom_models.sd_inference.sd_pipeline import SharkSDPipeline
from turbine_models.custom_models.sdxl_inference.sdxl_compiled_pipeline import (
    SharkSDXLPipeline,
)


from apps.shark_studio.api.controlnet import control_adapter_map
from apps.shark_studio.api.utils import parse_device
from apps.shark_studio.web.utils.state import status_label
from apps.shark_studio.web.utils.file_utils import (
    safe_name,
    get_resource_path,
    get_checkpoints_path,
)

from apps.shark_studio.modules.img_processing import (
    save_output_img,
)

from apps.shark_studio.modules.ckpt_processing import (
    preprocessCKPT,
    save_irpa,
)

EMPTY_SD_MAP = {
    "clip": None,
    "scheduler": None,
    "unet": None,
    "vae_decode": None,
}

EMPTY_SDXL_MAP = {
    "prompt_encoder": None,
    "scheduled_unet": None,
    "vae_decode": None,
    "pipeline": None,
    "full_pipeline": None,
}

EMPTY_FLAGS = {
    "clip": None,
    "unet": None,
    "vae": None,
    "pipeline": None,
}


class StableDiffusion:
    # This class is responsible for executing image generation and creating
    # /managing a set of compiled modules to run Stable Diffusion. The init
    # aims to be as general as possible, and the class will infer and compile
    # a list of necessary modules or a combined "pipeline module" for a
    # specified job based on the inference task.

    def __init__(
        self,
        base_model_id,
        height: int,
        width: int,
        batch_size: int,
        steps: int,
        scheduler: str,
        precision: str,
        device: str,
        custom_vae: str = None,
        num_loras: int = 0,
        import_ir: bool = True,
        is_controlled: bool = False,
    ):
        self.precision = precision
        self.compiled_pipeline = False
        self.base_model_id = base_model_id
        self.custom_vae = custom_vae
        self.is_sdxl = "xl" in self.base_model_id.lower()
        if self.is_sdxl:
            self.turbine_pipe = SharkSDXLPipeline
            self.model_map = EMPTY_SDXL_MAP
        else:
            self.turbine_pipe = SharkSDPipeline
            self.model_map = EMPTY_SD_MAP
        external_weights = "safetensors"
        max_length = 64
        target_backend, self.rt_device, triple = parse_device(device)
        pipe_id_list = [
            safe_name(base_model_id),
            str(batch_size),
            str(max_length),
            f"{str(height)}x{str(width)}",
            precision,
            triple,
        ]
        if num_loras > 0:
            pipe_id_list.append(str(num_loras) + "lora")
        if is_controlled:
            pipe_id_list.append("controlled")
        if custom_vae:
            pipe_id_list.append(custom_vae)
        self.pipe_id = "_".join(pipe_id_list)
        self.pipeline_dir = Path(os.path.join(get_checkpoints_path(), self.pipe_id))
        self.weights_path = Path(
            os.path.join(
                get_checkpoints_path(), safe_name(self.base_model_id + "_" + precision)
            )
        )
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)

        decomp_attn = True
        attn_spec = None
        if triple in ["gfx940", "gfx942", "gfx90a"]:
            decomp_attn = False
            attn_spec = "mfma"
        elif triple in ["gfx1100", "gfx1103"]:
            decomp_attn = False
            attn_spec = "wmma"
        elif target_backend == "llvm-cpu":
            decomp_attn = False

        self.sd_pipe = self.turbine_pipe(
            hf_model_name=base_model_id,
            scheduler_id=scheduler,
            height=height,
            width=width,
            precision=precision,
            max_length=max_length,
            batch_size=batch_size,
            num_inference_steps=steps,
            device=target_backend,
            iree_target_triple=triple,
            ireec_flags=EMPTY_FLAGS,
            attn_spec=attn_spec,
            decomp_attn=decomp_attn,
            pipeline_dir=self.pipeline_dir,
            external_weights_dir=self.weights_path,
            external_weights=external_weights,
            custom_vae=custom_vae,
        )
        print(f"\n[LOG] Pipeline initialized with pipe_id: {self.pipe_id}.")
        gc.collect()

    def prepare_pipe(self, custom_weights, adapters, embeddings, is_img2img):
        print(f"\n[LOG] Preparing pipeline...")
        self.is_img2img = False
        mlirs = copy.deepcopy(self.model_map)
        vmfbs = copy.deepcopy(self.model_map)
        weights = copy.deepcopy(self.model_map)

        if custom_weights:
            custom_weights = os.path.join(
                get_checkpoints_path("checkpoints"),
                safe_name(self.base_model_id.split("/")[-1]),
                custom_weights,
            )
            diffusers_weights_path = preprocessCKPT(custom_weights, self.precision)
            for key in weights:
                if key in ["scheduled_unet", "unet"]:
                    unet_weights_path = os.path.join(
                        diffusers_weights_path,
                        "unet",
                        "diffusion_pytorch_model.safetensors",
                    )
                    weights[key] = save_irpa(unet_weights_path, "unet.")

                elif key in ["clip", "prompt_encoder"]:
                    if not self.is_sdxl:
                        sd1_path = os.path.join(
                            diffusers_weights_path, "text_encoder", "model.safetensors"
                        )
                        weights[key] = save_irpa(sd1_path, "text_encoder_model.")
                    else:
                        clip_1_path = os.path.join(
                            diffusers_weights_path, "text_encoder", "model.safetensors"
                        )
                        clip_2_path = os.path.join(
                            diffusers_weights_path,
                            "text_encoder_2",
                            "model.safetensors",
                        )
                        weights[key] = [
                            save_irpa(clip_1_path, "text_encoder_model_1."),
                            save_irpa(clip_2_path, "text_encoder_model_2."),
                        ]

                elif key in ["vae_decode"] and weights[key] is None:
                    vae_weights_path = os.path.join(
                        diffusers_weights_path,
                        "vae",
                        "diffusion_pytorch_model.safetensors",
                    )
                    weights[key] = save_irpa(vae_weights_path, "vae.")

        vmfbs, weights = self.sd_pipe.check_prepared(
            mlirs, vmfbs, weights, interactive=False
        )
        print(f"\n[LOG] Loading pipeline to device {self.rt_device}.")
        self.sd_pipe.load_pipeline(
            vmfbs, weights, self.rt_device, self.compiled_pipeline
        )
        print(
            "\n[LOG] Pipeline successfully prepared for runtime. Generating images..."
        )
        return

    def generate_images(
        self,
        prompt,
        negative_prompt,
        image,
        strength,
        guidance_scale,
        seed,
        ondemand,
        repeatable_seeds,
        resample_type,
        control_mode,
        hints,
    ):
        img = self.sd_pipe.generate_images(
            prompt,
            negative_prompt,
            1,
            guidance_scale,
            seed,
            return_imgs=True,
        )
        return img


def shark_sd_fn_dict_input(
    sd_kwargs: dict,
):
    print("[LOG] Submitting Request...")

    for key in sd_kwargs:
        if sd_kwargs[key] in [None, []]:
            sd_kwargs[key] = None
        if sd_kwargs[key] in ["None"]:
            sd_kwargs[key] = ""
        if key == "seed":
            sd_kwargs[key] = int(sd_kwargs[key])

    for i in range(1):
        generated_imgs = yield from shark_sd_fn(**sd_kwargs)
        yield generated_imgs


def shark_sd_fn(
    prompt,
    negative_prompt,
    sd_init_image: list,
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
    if not isinstance(sd_init_image, list):
        sd_init_image = [sd_init_image]
    is_img2img = True if sd_init_image[0] is not None else False

    print("\n[LOG] Performing Stable Diffusion Pipeline setup...")

    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    adapters = {}
    is_controlled = False
    control_mode = None
    hints = []
    num_loras = 0
    import_ir = True
    for i in embeddings:
        num_loras += 1 if embeddings[i] else 0
    if "model" in controlnets:
        for i, model in enumerate(controlnets["model"]):
            if "xl" not in base_model_id.lower():
                adapters[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map["runwayml/stable-diffusion-v1-5"][
                        model
                    ],
                    "strength": controlnets["strength"][i],
                }
            else:
                adapters[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map["stabilityai/stable-diffusion-xl-1.0"][
                        model
                    ],
                    "strength": controlnets["strength"][i],
                }
            if model is not None:
                is_controlled = True
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
        "import_ir": import_ir,
        "is_controlled": is_controlled,
        "steps": steps,
        "scheduler": scheduler,
    }
    submit_prep_kwargs = {
        "custom_weights": custom_weights,
        "adapters": adapters,
        "embeddings": embeddings,
        "is_img2img": is_img2img,
    }
    submit_run_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": sd_init_image,
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

        # Initializes the pipeline and retrieves IR based on all
        # parameters that are static in the turbine output format,
        # which is currently MLIR in the torch dialect.

        sd_pipe = StableDiffusion(
            **submit_pipe_kwargs,
        )
        global_obj.set_sd_obj(sd_pipe)
        global_obj.set_pipe_kwargs(submit_pipe_kwargs)
    if (
        not global_obj.get_prep_kwargs()
        or global_obj.get_prep_kwargs() != submit_prep_kwargs
    ):
        global_obj.set_prep_kwargs(submit_prep_kwargs)
        global_obj.get_sd_obj().prepare_pipe(**submit_prep_kwargs)

    generated_imgs = []
    for current_batch in range(batch_count):
        start_time = time.time()
        out_imgs = global_obj.get_sd_obj().generate_images(**submit_run_kwargs)
        # total_time = time.time() - start_time
        # text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        # print(f"\n[LOG] {text_output}")
        # if global_obj.get_sd_status() == SD_STATE_CANCEL:
        #     break
        # else:
        save_output_img(
            out_imgs[current_batch],
            seed,
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


def safe_name(name):
    return name.replace("/", "_").replace("\\", "_").replace(".", "_")


if __name__ == "__main__":
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    global_obj._init()

    sd_json = view_json_file(
        get_resource_path(os.path.join(cmd_opts.config_dir, "default_sd_config.json"))
    )
    sd_kwargs = json.loads(sd_json)
    for arg in vars(cmd_opts):
        if arg in sd_kwargs:
            sd_kwargs[arg] = getattr(cmd_opts, arg)
    for i in shark_sd_fn_dict_input(sd_kwargs):
        print(i)
