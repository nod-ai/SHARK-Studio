import gc
import torch
import time
import os
import json
import numpy as np
from tqdm.auto import tqdm

from pathlib import Path
from random import randint
from turbine_models.custom_models.sd_inference import clip, unet, vae
from apps.shark_studio.api.controlnet import control_adapter_map
from apps.shark_studio.web.utils.state import status_label
from apps.shark_studio.web.utils.file_utils import (
    safe_name,
    get_resource_path,
    get_checkpoints_path,
)
from apps.shark_studio.modules.pipeline import SharkPipelineBase
from apps.shark_studio.modules.schedulers import get_schedulers
from apps.shark_studio.modules.prompt_encoding import (
    get_weighted_text_embeddings,
)
from apps.shark_studio.modules.img_processing import (
    resize_stencil,
    save_output_img,
    resamplers,
    resampler_list,
)

from apps.shark_studio.modules.ckpt_processing import (
    preprocessCKPT,
    process_custom_pipe_weights,
)
from transformers import CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor

sd_model_map = {
    "clip": {
        "initializer": clip.export_clip_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
        ],
    },
    "vae_encode": {
        "initializer": vae.export_vae_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-detach-elementwise-from-named-ops,iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
        ],
    },
    "unet": {
        "initializer": unet.export_unet_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))",
        ],
    },
    "vae_decode": {
        "initializer": vae.export_vae_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-detach-elementwise-from-named-ops,iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
        ],
    },
}


class SharkDiffusionPipeline(SharkPipelineBase):
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
        precision: str,
        device: str,
        custom_vae: str = None,
        num_loras: int = 0,
        import_ir: bool = True,
        is_controlled: bool = False,
    ):
        self.model_max_length = 77
        self.batch_size = batch_size
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.height = height
        self.width = width
        self.scheduler_obj = {}
        compile_static_args = {
            "pipe": {
                "external_weights": "safetensors",
            },
            "clip": {"hf_model_name": base_model_id},
            "unet": {
                "hf_model_name": base_model_id,
                "unet_model": unet.UnetModel(
                    hf_model_name=base_model_id, hf_auth_token=None
                ),
                "batch_size": batch_size,
                # "is_controlled": is_controlled,
                # "num_loras": num_loras,
                "height": height,
                "width": width,
                "precision": precision,
                "max_length": self.model_max_length,
            },
            "vae_encode": {
                "hf_model_name": base_model_id,
                "vae_model": self.vae_encode,
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "precision": precision,
            },
            "vae_decode": {
                "hf_model_name": base_model_id,
                "vae_model": self.vae_decode,
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "precision": precision,
            },
        }
        super().__init__(sd_model_map, base_model_id, compile_static_args, device, import_ir)
        pipe_id_list = [
            safe_name(base_model_id),
            str(batch_size),
            str(static_kwargs["unet"]["max_length"]),
            f"{str(height)}x{str(width)}",
            precision,
        ]
        if num_loras > 0:
            pipe_id_list.append(str(num_loras) + "lora")
        if is_controlled:
            pipe_id_list.append("controlled")
        if custom_vae:
            pipe_id_list.append(custom_vae)
        self.pipe_id = "_".join(pipe_id_list)
        print(f"\n[LOG] Pipeline initialized with pipe_id: {self.pipe_id}.")
        del static_kwargs
        gc.collect()

    def prepare_pipe(self, custom_weights, adapters, embeddings, is_img2img):
        print(f"\n[LOG] Preparing pipeline...")
        self.is_img2img = is_img2img
        self.schedulers = get_schedulers(self.base_model_id)

        self.weights_path = os.path.join(
            get_checkpoints_path(), self.safe_name(self.base_model_id)
        )
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)

        for model in adapters:
            self.model_map[model] = adapters[model]

        for submodel in self.static_kwargs:
            if custom_weights:
                custom_weights_params, _ = process_custom_pipe_weights(custom_weights)
                if submodel not in ["clip", "clip2"]:
                    self.static_kwargs[submodel][
                        "external_weight_file"
                    ] = custom_weights_params
                else:
                    self.static_kwargs[submodel]["external_weight_path"] = os.path.join(
                        self.weights_path, submodel + ".safetensors"
                    )
            else:
                self.static_kwargs[submodel]["external_weight_path"] = os.path.join(
                    self.weights_path, submodel + ".safetensors"
                )

        self.get_compiled_map(pipe_id=self.pipe_id)
        print("\n[LOG] Pipeline successfully prepared for runtime.")
        return


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
        "import_ir": cmd_opts.import_mlir,
        "is_controlled": is_controlled,
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
        "steps": steps,
        "scheduler": scheduler,
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

        sd_pipe = SharkDiffusionPipeline(
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
        total_time = time.time() - start_time
        text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        print(f"\n[LOG] {text_output}")
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


if __name__ == "__main__":
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    global_obj._init()

    sd_json = view_json_file(get_resource_path("../configs/default_sd_config.json"))
    sd_kwargs = json.loads(sd_json)
    for arg in vars(cmd_opts):
        if arg in sd_kwargs:
            sd_kwargs[arg] = getattr(cmd_opts, arg)
    for i in shark_sd_fn_dict_input(sd_kwargs):
        print(i)
