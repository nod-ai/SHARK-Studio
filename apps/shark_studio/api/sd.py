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
from apps.shark_studio.web.utils.file_utils import safe_name, get_resource_path, get_checkpoints_path
from apps.shark_studio.modules.pipeline import SharkPipelineBase
from apps.shark_studio.modules.schedulers import get_schedulers
from apps.shark_studio.modules.prompt_encoding import get_weighted_text_embeddings
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
from math import ceil
from PIL import Image

sd_model_map = {
    "clip": {
        "initializer": clip.export_clip_model,
        "external_weight_file": None,
        "ireec_flags": ["--iree-flow-collapse-reduction-dims"],
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
        "ireec_flags": ["--iree-flow-collapse-reduction-dims",
                        "--iree-opt-const-expr-hoisting=False",
                        "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
        ],
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
        is_controlled: bool = False,
    ):
        self.model_max_length = 77
        self.batch_size = batch_size
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.height = height
        self.width = width
        self.scheduler_obj = {}
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
                "precision": precision,
                "max_length": 77 * 8,
            },
            "vae_encode": {
                "hf_model_name": custom_vae if custom_vae else base_model_id,
                "vae_model": vae.VaeModel(hf_model_name=base_model_id, hf_auth_token=None),
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "precision": precision,
            },
            "vae_decode": {
                "hf_model_name": custom_vae if custom_vae else base_model_id,
                "vae_model": vae.VaeModel(hf_model_name=base_model_id, hf_auth_token=None),
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "precision": precision,
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


    def prepare_pipe(self, scheduler, custom_weights, adapters, embeddings, is_img2img):
        print(
            f"\n[LOG] Preparing pipeline with scheduler {scheduler}"
            f"\n[LOG] Custom embeddings currently unsupported."
        )
        self.is_img2img = is_img2img
        schedulers = get_schedulers(self.base_model_id)
        self.weights_path = get_checkpoints_path(self.safe_name(self.pipe_id))
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)
        self.scheduler = schedulers[scheduler]
        print(f"[LOG] Loaded scheduler: {scheduler}")
        for model in adapters:
            self.model_map[model] = adapters[model]
        if custom_weights:
            if os.path.isfile(custom_weights):
                for i in self.model_map:
                    self.model_map[i]["external_weights_file"] = None
            elif custom_weights:
                print(f"\n[LOG][WARNING] Custom weights were not found at {custom_weights}. Did you mean to pass a base model ID?")
        self.static_kwargs["pipe"] = {
        #    "external_weight_path": self.weights_path,
#            "external_weights": "safetensors",
        }
        self.get_compiled_map(pipe_id=self.pipe_id)
        print("\n[LOG] Pipeline successfully prepared for runtime.")
        return
        

    def generate_images(
        self,
        prompt,
        negative_prompt,
        image,
        steps,
        strength,
        guidance_scale,
        seed,
        ondemand,
        repeatable_seeds,
        use_base_vae,
        resample_type,
        control_mode,
        hints,
    ):
        #TODO: Batched args
        self.ondemand = ondemand
        if self.is_img2img:
            image, _ = self.process_sd_init_image(image, resample_type)
        else: 
            image = None

        print("\n[LOG] Generating images...")
        batched_args=[
            prompt,
            negative_prompt,
            #steps,
            #strength,
            #guidance_scale,
            #seed,
            #resample_type,
            #control_mode,
            #hints,
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
        
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)

        generator = torch.manual_seed(seed)

        init_latents, final_timesteps = self.prepare_latents(
            generator=generator,
            num_inference_steps=steps,
            image=image,
            strength=strength,
        )

        latents = self.produce_img_latents(
            latents=init_latents,
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            total_timesteps=final_timesteps,
            cpu_scheduling=True, # until we have schedulers through Turbine
        )

        # Img latents -> PIL images
        all_imgs = []
        self.load_submodels(["vae_decode"])
        for i in tqdm(range(0, latents.shape[0], self.batch_size)):
            imgs = self.decode_latents(
                latents=latents[i : i + self.batch_size],
                use_base_vae=use_base_vae,
                cpu_scheduling=True,
            )
            all_imgs.extend(imgs)
        if self.ondemand:
            self.unload_submodels(["vae_decode"])

        return all_imgs


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
        pad = pad + (0, 77 * 8 - text_embeddings.shape[1])
        text_embeddings = torch.nn.functional.pad(text_embeddings, pad)

        # SHARK: Report clip inference time
        clip_inf_time = (time.time() - clip_inf_start) * 1000
        if self.ondemand:
            self.unload_submodels(["clip"])
            gc.collect()
        print(f"\n[LOG] Clip Inference time (ms) = {clip_inf_time:.3f}")

        return text_embeddings.numpy().astype(np.float16)


    def prepare_latents(
        self,
        generator,
        num_inference_steps,
        image,
        strength,
    ):
        noise = torch.randn(
            (
                self.batch_size,
                4,
                self.height // 8,
                self.width // 8,
            ),
            generator=generator,
            dtype=self.dtype,
        ).to("cpu")
        
        self.scheduler.set_timesteps(num_inference_steps)
        if self.is_img2img:
            init_timestep = min(
                int(num_inference_steps * strength), num_inference_steps
            )
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start:]
            latents = self.encode_image(image)
            latents = self.scheduler.add_noise(
                latents, noise, timesteps[0].repeat(1)
            )
            return latents, [timesteps]
        else:
            self.scheduler.is_scale_input_called = True
            latents = noise * self.scheduler.init_noise_sigma
            return latents, self.scheduler.timesteps


    def encode_image(self, input_image):
        self.load_submodels(["vae_encode"])
        vae_encode_start = time.time()
        latents = self.run("vae_encode", input_image)
        vae_inf_time = (time.time() - vae_encode_start) * 1000
        if self.ondemand:
            self.unload_submodels(["vae_encode"])
        print(f"\n[LOG] VAE Encode Inference time (ms): {vae_inf_time:.3f}")

        return latents


    def produce_img_latents(
        self,
        latents,
        text_embeddings,
        guidance_scale,
        total_timesteps,
        cpu_scheduling,
        mask=None,
        masked_image_latents=None,
        return_all_latents=False,
    ):
        # self.status = SD_STATE_IDLE
        step_time_sum = 0
        latent_history = [latents]
        text_embeddings = torch.from_numpy(text_embeddings).to(torch.float16)
        text_embeddings_numpy = text_embeddings.detach().numpy()
        guidance_scale = np.asarray([guidance_scale], dtype=np.float16)
        self.load_submodels(["unet"])
        for i, t in tqdm(enumerate(total_timesteps)):
            step_start_time = time.time()
            timestep = torch.tensor([t]).to(self.dtype).detach().numpy()
            latent_model_input = self.scheduler.scale_model_input(latents, t).to(self.dtype)
            if mask is not None and masked_image_latents is not None:
                latent_model_input = torch.cat(
                    [
                        torch.from_numpy(np.asarray(latent_model_input)).to(torch.float16),
                        mask,
                        masked_image_latents,
                    ],
                    dim=1,
                ).to(self.dtype)
            if cpu_scheduling:
                latent_model_input = latent_model_input.detach().numpy()

            # Profiling Unet.
            # profile_device = start_profiling(file_path="unet.rdc")
            noise_pred = self.run(
                "unet",
                [
                    latent_model_input,
                    timestep,
                    text_embeddings_numpy,
                    guidance_scale,
                ],
            )
            # end_profiling(profile_device)

            if cpu_scheduling:
                noise_pred = torch.from_numpy(noise_pred.to_host())
                latents = self.scheduler.step(
                    noise_pred, t, latents
                ).prev_sample
            else:
                latents = self.run("scheduler_step", (noise_pred, t, latents))

            latent_history.append(latents)
            step_time = (time.time() - step_start_time) * 1000
            #  self.log += (
            #      f"\nstep = {i} | timestep = {t} | time = {step_time:.2f}ms"
            #  )
            step_time_sum += step_time

            #if self.status == SD_STATE_CANCEL:
            #    break

        if self.ondemand:
            self.unload_submodels(["unet"])
            gc.collect()

        avg_step_time = step_time_sum / len(total_timesteps)
        print(f"\n[LOG] Average step time: {avg_step_time}ms/it")

        if not return_all_latents:
            return latents
        all_latents = torch.cat(latent_history, dim=0)
        return all_latents


    def decode_latents(self, latents, use_base_vae, cpu_scheduling):
        if use_base_vae:
            latents = 1 / 0.18215 * latents

        latents_numpy = latents.to(self.dtype)
        if cpu_scheduling:
            latents_numpy = latents.detach().numpy()

        #profile_device = start_profiling(file_path="vae.rdc")
        vae_start = time.time()
        images = self.run("vae_decode", latents_numpy).to_host()
        vae_inf_time = (time.time() - vae_start) * 1000
        #end_profiling(profile_device)
        print(f"\n[LOG] VAE Inference time (ms): {vae_inf_time:.3f}")

        if use_base_vae:
            images = torch.from_numpy(images)
            images = (images.detach().cpu() * 255.0).numpy()
            images = images.round()

        images = torch.from_numpy(images).to(torch.uint8).permute(0, 2, 3, 1)
        pil_images = [Image.fromarray(image) for image in images.numpy()]
        return pil_images


    def process_sd_init_image(self, sd_init_image, resample_type):
        if isinstance(sd_init_image, list):
            images = []
            for img in sd_init_image:
                img, _ = self.process_sd_init_image(img, resample_type)
                images.append(img)
                is_img2img = True
                return images, is_img2img
        if isinstance(sd_init_image, str):
            if os.path.isfile(sd_init_image):
                sd_init_image = Image.open(sd_init_image, mode="r").convert("RGB")
                image, is_img2img = self.process_sd_init_image(sd_init_image, resample_type)
            else:
                image = None
                is_img2img = False
        elif isinstance(sd_init_image, Image.Image):
            image = sd_init_image.convert("RGB")
        elif sd_init_image:
            image = sd_init_image["image"].convert("RGB")
        else:
            image = None
            is_img2img = False
        if image:
            resample_type = (
                resamplers[resample_type]
                if resample_type in resampler_list
                # Fallback to Lanczos
                else Image.Resampling.LANCZOS
            )
            image = image.resize((self.width, self.height), resample=resample_type)
            image_arr = np.stack([np.array(i) for i in (image,)], axis=0)
            image_arr = image_arr / 255.0
            image_arr = torch.from_numpy(image_arr).permute(0, 3, 1, 2).to(self.dtype)
            image_arr = 2 * (image_arr - 0.5)
            is_img2img = True
            image = image_arr
        return image, is_img2img


def shark_sd_fn_dict_input(
    sd_kwargs: dict,
):
    print("[LOG] Submitting Request...")

    for key in sd_kwargs:
        if sd_kwargs[key] in ["None", "", None, []]:
            sd_kwargs[key] = None
        if key == "seed":
            sd_kwargs[key] = int(sd_kwargs[key])

    for i in range(1):
        generated_imgs = yield from shark_sd_fn(**sd_kwargs)
        yield generated_imgs
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
    use_base_vae: bool,
    precision: str,
    device: str,
    ondemand: bool,
    repeatable_seeds: bool,
    resample_type: str,
    controlnets: dict,
    embeddings: dict,
):
    sd_kwargs = locals()
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
        "is_controlled": is_controlled,
    }
    submit_prep_kwargs = {
        "scheduler": scheduler,
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
        "strength": strength,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "ondemand": ondemand,
        "repeatable_seeds": repeatable_seeds,
        "use_base_vae": use_base_vae,
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
        save_output_img(
            out_imgs[0],
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
    for i in shark_sd_fn_dict_input(sd_kwargs):
        print(i)