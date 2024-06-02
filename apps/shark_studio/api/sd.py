import gc
import torch
import gradio as gr
import time
import os
import json
import numpy as np
import copy
import importlib.util
import sys
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


def load_script(source, module_name):
    """
    reads file source and loads it as a module

    :param source: file to load
    :param module_name: name of module to register in sys.modules
    :return: loaded module
    """

    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


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
        target_triple: str = None,
        custom_vae: str = None,
        num_loras: int = 0,
        import_ir: bool = True,
        is_controlled: bool = False,
        external_weights: str = "safetensors",
        progress=gr.Progress(),
    ):
        progress(0, desc="Initializing pipeline...")
        self.ui_device = device
        self.precision = precision
        self.compiled_pipeline = False
        self.base_model_id = base_model_id
        self.custom_vae = custom_vae
        self.is_sdxl = "xl" in self.base_model_id.lower()
        self.is_custom = ".py" in self.base_model_id.lower()
        if self.is_custom:
            custom_module = load_script(
                os.path.join(get_checkpoints_path("scripts"), self.base_model_id),
                "custom_pipeline",
            )
            self.turbine_pipe = custom_module.StudioPipeline
            self.dynamic_steps = False
            self.model_map = custom_module.MODEL_MAP
        elif self.is_sdxl:
            self.turbine_pipe = SharkSDXLPipeline
            self.dynamic_steps = False
            self.model_map = EMPTY_SDXL_MAP
        else:
            self.turbine_pipe = SharkSDPipeline
            self.dynamic_steps = True
            self.model_map = EMPTY_SD_MAP
        max_length = 64
        target_backend, self.rt_device, triple = parse_device(device, target_triple)
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
        elif triple in ["gfx1100", "gfx1103", "gfx1150"]:
            decomp_attn = False
            attn_spec = "wmma"
            if triple in ["gfx1103", "gfx1150"]:
                # external weights have issues on igpu
                external_weights = None
        elif target_backend == "llvm-cpu":
            decomp_attn = False
        progress(0.5, desc="Initializing pipeline...")
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
        progress(1, desc="Pipeline initialized!...")
        gc.collect()

    def prepare_pipe(
        self,
        custom_weights,
        adapters,
        embeddings,
        is_img2img,
        compiled_pipeline,
        progress=gr.Progress(),
    ):
        progress(0, desc="Preparing models...")

        self.is_img2img = False
        mlirs = copy.deepcopy(self.model_map)
        vmfbs = copy.deepcopy(self.model_map)
        weights = copy.deepcopy(self.model_map)
        if not self.is_sdxl:
            compiled_pipeline = False
        self.compiled_pipeline = compiled_pipeline

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
        progress(0.25, desc=f"Preparing pipeline for {self.ui_device}...")

        vmfbs, weights = self.sd_pipe.check_prepared(
            mlirs, vmfbs, weights, interactive=False
        )
        progress(0.5, desc=f"Artifacts ready!")
        progress(0.75, desc=f"Loading models and weights...")

        self.sd_pipe.load_pipeline(
            vmfbs, weights, self.rt_device, self.compiled_pipeline
        )
        progress(1, desc="Pipeline loaded! Generating images...")
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
        resample_type,
        control_mode,
        hints,
        progress=gr.Progress(),
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


def shark_sd_fn_dict_input(sd_kwargs: dict, *, progress=gr.Progress()):
    print("\n[LOG] Submitting Request...")

    for key in sd_kwargs:
        if sd_kwargs[key] in [None, []]:
            sd_kwargs[key] = None
        if sd_kwargs[key] in ["None"]:
            sd_kwargs[key] = ""
        if key in ["steps", "height", "width", "batch_count", "batch_size"]:
            sd_kwargs[key] = int(sd_kwargs[key])
        if key == "seed":
            sd_kwargs[key] = int(sd_kwargs[key])

    # TODO: move these checks into the UI code so we don't have gradio warnings in a generalized dict input function.
    if not sd_kwargs["device"]:
        gr.Warning("No device specified. Please specify a device.")
        return None, ""
    if sd_kwargs["height"] not in [512, 1024]:
        gr.Warning("Height must be 512 or 1024. This is a temporary limitation.")
        return None, ""
    if sd_kwargs["height"] != sd_kwargs["width"]:
        gr.Warning("Height and width must be the same. This is a temporary limitation.")
        return None, ""
    if sd_kwargs["base_model_id"] == "stabilityai/sdxl-turbo":
        if sd_kwargs["steps"] > 10:
            gr.Warning("Max steps for sdxl-turbo is 10. 1 to 4 steps are recommended.")
            return None, ""
        if sd_kwargs["guidance_scale"] > 3:
            gr.Warning(
                "sdxl-turbo CFG scale should be less than 2.0 if using negative prompt, 0 otherwise."
            )
            return None, ""
    if sd_kwargs["target_triple"] == "":
        if not parse_device(sd_kwargs["device"], sd_kwargs["target_triple"])[2]:
            gr.Warning(
                "Target device architecture could not be inferred. Please specify a target triple, e.g. 'gfx1100' for a Radeon 7900xtx."
            )
            return None, ""

    generated_imgs = yield from shark_sd_fn(**sd_kwargs)
    return generated_imgs


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
    target_triple: str,
    ondemand: bool,
    compiled_pipeline: bool,
    resample_type: str,
    controlnets: dict,
    embeddings: dict,
    seed_increment: str | int = 1,
    progress=gr.Progress(),
):
    sd_kwargs = locals()
    if not isinstance(sd_init_image, list):
        sd_init_image = [sd_init_image]
    is_img2img = True if sd_init_image[0] is not None else False

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
        "target_triple": target_triple,
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
        "compiled_pipeline": compiled_pipeline,
    }
    submit_run_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": sd_init_image,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "ondemand": ondemand,
        "resample_type": resample_type,
        "control_mode": control_mode,
        "hints": hints,
    }
    if global_obj.get_sd_obj() and global_obj.get_sd_obj().dynamic_steps:
        submit_run_kwargs["steps"] = submit_pipe_kwargs["steps"]
        submit_pipe_kwargs.pop("steps")
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
    if seed in [-1, "-1"]:
        seed = randint(0, 4294967295)
        seed_increment = "random"
        print(f"\n[LOG] Random seed: {seed}")
    progress(None, desc=f"Generating...")

    for current_batch in range(batch_count):
        start_time = time.time()
        out_imgs = global_obj.get_sd_obj().generate_images(**submit_run_kwargs)
        if not isinstance(out_imgs, list):
            out_imgs = [out_imgs]
        # total_time = time.time() - start_time
        # text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        # print(f"\n[LOG] {text_output}")
        # if global_obj.get_sd_status() == SD_STATE_CANCEL:
        #     break
        # else:
        for batch in range(batch_size):
            save_output_img(
                out_imgs[batch],
                seed,
                sd_kwargs,
            )
        generated_imgs.extend(out_imgs)
        seed = get_next_seed(seed, seed_increment)
        yield generated_imgs, status_label(
            "Stable Diffusion", current_batch + 1, batch_count, batch_size
        )
    return (generated_imgs, "")


def get_next_seed(seed, seed_increment: str | int = 10):
    if isinstance(seed_increment, int):
        print(f"\n[LOG] Seed after batch increment: {seed + seed_increment}")
        return int(seed + seed_increment)
    elif seed_increment == "random":
        seed = randint(0, 4294967295)
        print(f"\n[LOG] Random seed: {seed}")
        return seed


def unload_sd():
    print("Unloading models.")
    import apps.shark_studio.web.utils.globals as global_obj

    global_obj.clear_cache()
    gc.collect()


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
