from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from collections import defaultdict
from pathlib import Path
import torch
import safetensors.torch
import traceback
import subprocess
import sys
import os
import requests
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    get_opt_flags,
    base_models,
    args,
    preprocessCKPT,
    convert_original_vae,
    get_path_to_diffusers_checkpoint,
    get_civitai_checkpoint,
    fetch_and_update_base_model_id,
    get_path_stem,
    get_extended_name,
    get_stencil_model_id,
    update_lora_weight,
)
from shark.shark_downloader import download_public_file
from shark.shark_inference import SharkInference


# These shapes are parameter dependent.
def replace_shape_str(shape, max_len, width, height, batch_size):
    new_shape = []
    for i in range(len(shape)):
        if shape[i] == "max_len":
            new_shape.append(max_len)
        elif shape[i] == "height":
            new_shape.append(height)
        elif shape[i] == "width":
            new_shape.append(width)
        elif isinstance(shape[i], str):
            if "*" in shape[i]:
                mul_val = int(shape[i].split("*")[0])
                if "batch_size" in shape[i]:
                    new_shape.append(batch_size * mul_val)
                elif "height" in shape[i]:
                    new_shape.append(height * mul_val)
                elif "width" in shape[i]:
                    new_shape.append(width * mul_val)
            elif "/" in shape[i]:
                import math

                div_val = int(shape[i].split("/")[1])
                if "batch_size" in shape[i]:
                    new_shape.append(math.ceil(batch_size / div_val))
                elif "height" in shape[i]:
                    new_shape.append(math.ceil(height / div_val))
                elif "width" in shape[i]:
                    new_shape.append(math.ceil(width / div_val))
            elif "+" in shape[i]:
                # Currently this case only hits for SDXL. So, in case any other
                # case requires this operator, change this.
                new_shape.append(height + width)
        else:
            new_shape.append(shape[i])
    return new_shape


def check_compilation(model, model_name):
    if not model:
        raise Exception(
            f"Could not compile {model_name}. Please create an issue with the detailed log at https://github.com/nod-ai/SHARK/issues"
        )


def shark_compile_after_ir(
    module_name,
    device,
    vmfb_path,
    precision,
    ir_path=None,
):
    if ir_path:
        print(f"[DEBUG] mlir found at {ir_path.absolute()}")

    module = SharkInference(
        mlir_module=ir_path,
        device=device,
        mlir_dialect="tm_tensor",
    )
    print(f"Will get extra flag for {module_name} and precision = {precision}")
    path = module.save_module(
        vmfb_path.parent.absolute(),
        vmfb_path.stem,
        extra_args=get_opt_flags(module_name, precision=precision),
    )
    print(f"Saved {module_name} vmfb at {path}")
    module.load_module(path)
    return module


def process_vmfb_ir_sdxl(extended_model_name, model_name, device, precision):
    name_split = extended_model_name.split("_")
    if "vae" in model_name:
        name_split[5] = "fp32"
    extended_model_name_for_vmfb = "_".join(name_split)
    extended_model_name_for_mlir = "_".join(name_split[:-1])
    vmfb_path = Path(extended_model_name_for_vmfb + ".vmfb")
    if "vulkan" in device:
        _device = args.iree_vulkan_target_triple
        _device = _device.replace("-", "_")
        vmfb_path = Path(extended_model_name_for_vmfb + f"_vulkan.vmfb")
    if vmfb_path.exists():
        shark_module = SharkInference(
            None,
            device=device,
            mlir_dialect="tm_tensor",
        )
        print(f"loading existing vmfb from: {vmfb_path}")
        shark_module.load_module(vmfb_path, extra_args=[])
        return shark_module, None
    mlir_path = Path(extended_model_name_for_mlir + ".mlir")
    if not mlir_path.exists():
        print(f"Looking into gs://shark_tank/SDXL/mlir/{mlir_path.name}")
        download_public_file(
            f"gs://shark_tank/SDXL/mlir/{mlir_path.name}",
            mlir_path.absolute(),
            single_file=True,
        )
    if mlir_path.exists():
        return (
            shark_compile_after_ir(
                model_name, device, vmfb_path, precision, mlir_path
            ),
            None,
        )
    return None, None


class SharkifyStableDiffusionModel:
    def __init__(
        self,
        model_id: str,
        custom_weights: str,
        custom_vae: str,
        precision: str,
        max_len: int = 64,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        use_base_vae: bool = False,
        use_tuned: bool = False,
        low_cpu_mem_usage: bool = False,
        debug: bool = False,
        sharktank_dir: str = "",
        generate_vmfb: bool = True,
        is_inpaint: bool = False,
        is_upscaler: bool = False,
        is_sdxl: bool = False,
        stencils: list[str] = [],
        use_lora: str = "",
        use_quantize: str = None,
        return_mlir: bool = False,
    ):
        self.check_params(max_len, width, height)
        self.max_len = max_len
        self.is_sdxl = is_sdxl
        self.height = height // 8
        self.width = width // 8
        self.batch_size = batch_size
        self.custom_weights = custom_weights.strip()
        self.use_quantize = use_quantize
        if custom_weights != "":
            if custom_weights.startswith("https://civitai.com/api/"):
                # download the checkpoint from civitai if we don't already have it
                weights_path = get_civitai_checkpoint(custom_weights)

                # act as if we were given the local file as custom_weights originally
                custom_weights = get_path_to_diffusers_checkpoint(weights_path)
                self.custom_weights = weights_path

                # needed to ensure webui sets the correct model name metadata
                args.ckpt_loc = weights_path
            else:
                assert custom_weights.lower().endswith(
                    (".ckpt", ".safetensors")
                ), "checkpoint files supported can be any of [.ckpt, .safetensors] type"
                custom_weights = get_path_to_diffusers_checkpoint(
                    custom_weights
                )

        self.model_id = model_id if custom_weights == "" else custom_weights
        self.custom_vae = custom_vae
        self.precision = precision
        self.base_vae = use_base_vae
        self.model_name = (
            "_"
            + str(batch_size)
            + "_"
            + str(max_len)
            + "_"
            + str(height)
            + "_"
            + str(width)
            + "_"
            + precision
        )
        self.model_namedata = self.model_name
        print(f"use_tuned? sharkify: {use_tuned}")
        self.use_tuned = use_tuned
        if use_tuned:
            self.model_name = self.model_name + "_tuned"
        self.model_name = self.model_name + "_" + get_path_stem(self.model_id)
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.is_inpaint = is_inpaint
        self.is_upscaler = is_upscaler
        self.stencils = [get_stencil_model_id(x) for x in stencils]
        if use_lora != "":
            self.model_name = self.model_name + "_" + get_path_stem(use_lora)
        self.use_lora = use_lora

        self.model_name = self.get_extended_name_for_all_model()
        self.debug = debug
        self.sharktank_dir = sharktank_dir
        self.generate_vmfb = generate_vmfb

        self.inputs = dict()
        self.model_to_run = ""
        if self.custom_weights != "":
            self.model_to_run = self.custom_weights
            assert self.custom_weights.lower().endswith(
                (".ckpt", ".safetensors")
            ), "checkpoint files supported can be any of [.ckpt, .safetensors] type"
            preprocessCKPT(self.custom_weights, self.is_inpaint)
        else:
            self.model_to_run = args.hf_model_id
        self.custom_vae = self.process_custom_vae()
        self.base_model_id = fetch_and_update_base_model_id(self.model_to_run)
        if self.base_model_id != "" and args.ckpt_loc != "":
            args.hf_model_id = self.base_model_id
        self.return_mlir = return_mlir

    def get_extended_name_for_all_model(self, model_list=None):
        model_name = {}
        sub_model_list = [
            "clip",
            "clip2",
            "unet",
            "unet512",
            "stencil_unet",
            "stencil_unet_512",
            "vae",
            "vae_encode",
            "stencil_adapter",
            "stencil_adapter_512",
        ]
        if model_list:
            sub_model_list=model_list
        index = 0
        for model in sub_model_list:
            sub_model = model
            model_config = self.model_name
            if "vae" == model:
                if self.custom_vae != "":
                    model_config = model_config + get_path_stem(
                        self.custom_vae
                    )
                if self.base_vae:
                    sub_model = "base_vae"
            if "stencil_adapter" in model:
                stencil_names = []
                for i, stencil in enumerate(self.stencils):
                    if stencil is not None:
                        cnet_config = (
                            self.model_namedata
                            + "_v1-5"
                            + stencil.split("_")[-1]
                        )
                        stencil_names.append(
                            get_extended_name(sub_model + cnet_config)
                        )

                model_name[model] = stencil_names
            else:
                model_name[model] = get_extended_name(sub_model + model_config)
        index += 1
        print(f"model name at {index} = {self.model_name}")

        return model_name

    def check_params(self, max_len, width, height):
        if not (max_len >= 32 and max_len <= 77):
            sys.exit("please specify max_len in the range [32, 77].")
        if not (width % 8 == 0 and width >= 128):
            sys.exit("width should be greater than 128 and multiple of 8")
        if not (height % 8 == 0 and height >= 128):
            sys.exit("height should be greater than 128 and multiple of 8")

    # Get the input info for a model i.e. "unet", "clip", "vae", etc.
    def get_input_info_for(self, model_info):
        dtype_config = {"f32": torch.float32, "i64": torch.int64}
        input_map = []
        for inp in model_info:
            shape = model_info[inp]["shape"]
            dtype = dtype_config[model_info[inp]["dtype"]]
            tensor = None
            if isinstance(shape, list):
                clean_shape = replace_shape_str(
                    shape,
                    self.max_len,
                    self.width,
                    self.height,
                    self.batch_size,
                )
                if dtype == torch.int64:
                    tensor = torch.randint(1, 3, tuple(clean_shape))
                else:
                    tensor = torch.randn(*clean_shape).to(dtype)
            elif isinstance(shape, int):
                tensor = torch.tensor(shape).to(dtype)
            else:
                sys.exit("shape isn't specified correctly.")
            input_map.append(tensor)
        return input_map

    def get_vae_encode(self):
        class VaeEncodeModel(torch.nn.Module):
            def __init__(
                self, model_id=self.model_id, low_cpu_mem_usage=False
            ):
                super().__init__()
                self.vae = AutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )

            def forward(self, input):
                latents = self.vae.encode(input).latent_dist.sample()
                return 0.18215 * latents

        vae_encode = VaeEncodeModel()
        inputs = tuple(self.inputs["vae_encode"])
        is_f16 = (
            True
            if not self.is_upscaler and self.precision == "fp16"
            else False
        )
        shark_vae_encode, vae_encode_mlir = compile_through_fx(
            vae_encode,
            inputs,
            is_f16=is_f16,
            use_tuned=self.use_tuned,
            extended_model_name=self.model_name["vae_encode"],
            extra_args=get_opt_flags("vae", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name="vae_encode",
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_vae_encode, vae_encode_mlir

    def get_vae(self):
        class VaeModel(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                base_vae=self.base_vae,
                custom_vae=self.custom_vae,
                low_cpu_mem_usage=False,
            ):
                super().__init__()
                self.vae = None
                if custom_vae == "":
                    self.vae = AutoencoderKL.from_pretrained(
                        model_id,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                elif not isinstance(custom_vae, dict):
                    self.vae = AutoencoderKL.from_pretrained(
                        custom_vae,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                else:
                    self.vae = AutoencoderKL.from_pretrained(
                        model_id,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.vae.load_state_dict(custom_vae)
                self.base_vae = base_vae

            def forward(self, input):
                if not self.base_vae:
                    input = 1 / 0.18215 * input
                x = self.vae.decode(input, return_dict=False)[0]
                x = (x / 2 + 0.5).clamp(0, 1)
                if self.base_vae:
                    return x
                x = x * 255.0
                return x.round()

        vae = VaeModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        inputs = tuple(self.inputs["vae"])
        is_f16 = (
            True
            if not self.is_upscaler and self.precision == "fp16"
            else False
        )
        save_dir = os.path.join(self.sharktank_dir, self.model_name["vae"])
        if self.debug:
            os.makedirs(save_dir, exist_ok=True)
        shark_vae, vae_mlir = compile_through_fx(
            vae,
            inputs,
            is_f16=is_f16,
            use_tuned=self.use_tuned,
            extended_model_name=self.model_name["vae"],
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("vae", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name="vae",
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_vae, vae_mlir

    def get_vae_sdxl(self):
        # TODO: Remove this after convergence with shark_tank. This should just be part of
        #       opt_params.py.
        shark_module_or_none = process_vmfb_ir_sdxl(
            self.model_name["vae"], "vae", args.device, self.precision
        )
        if shark_module_or_none[0]:
            return shark_module_or_none

        class VaeModel(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                base_vae=self.base_vae,
                custom_vae=self.custom_vae,
                low_cpu_mem_usage=False,
            ):
                super().__init__()
                self.vae = None
                if custom_vae == "":
                    print(f"Loading default vae, with target {model_id}")
                    self.vae = AutoencoderKL.from_pretrained(
                        model_id,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                elif not isinstance(custom_vae, dict):
                    precision = "fp16" if "fp16" in custom_vae else None
                    print(f"Loading custom vae, with target {custom_vae}")
                    if os.path.exists(custom_vae):
                        self.vae = AutoencoderKL.from_pretrained(
                            custom_vae,
                            low_cpu_mem_usage=low_cpu_mem_usage,
                        )
                    else:
                        custom_vae = "/".join(
                            [
                                custom_vae.split("/")[-2].split("\\")[-1],
                                custom_vae.split("/")[-1],
                            ]
                        )
                        print("Using hub to get custom vae")
                        try:
                            self.vae = AutoencoderKL.from_pretrained(
                                custom_vae,
                                low_cpu_mem_usage=low_cpu_mem_usage,
                                variant=precision,
                            )
                        except:
                            self.vae = AutoencoderKL.from_pretrained(
                                custom_vae,
                                low_cpu_mem_usage=low_cpu_mem_usage,
                            )
                else:
                    print(f"Loading custom vae, with state {custom_vae}")
                    self.vae = AutoencoderKL.from_pretrained(
                        model_id,
                        subfolder="vae",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                    self.vae.load_state_dict(custom_vae)
                self.base_vae = base_vae

            def forward(self, latents):
                image = self.vae.decode(latents / 0.13025, return_dict=False)[
                    0
                ]
                return image

        vae = VaeModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        inputs = tuple(self.inputs["vae"])
        # Make sure the VAE is in float32 mode, as it overflows in float16 as per SDXL
        # pipeline.
        if not self.custom_vae:
            is_f16 = False
        elif "16" in self.custom_vae:
            is_f16 = True
        else:
            is_f16 = False
        save_dir = os.path.join(self.sharktank_dir, self.model_name["vae"])
        if self.debug:
            os.makedirs(save_dir, exist_ok=True)
        shark_vae, vae_mlir = compile_through_fx(
            vae,
            inputs,
            is_f16=is_f16,
            use_tuned=self.use_tuned,
            extended_model_name=self.model_name["vae"],
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("vae", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name="vae",
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_vae, vae_mlir

    def get_controlled_unet(self, use_large=False):
        class ControlledUnetModel(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                low_cpu_mem_usage=False,
                use_lora=self.use_lora,
            ):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                if use_lora != "":
                    update_lora_weight(self.unet, use_lora, "unet")
                self.in_channels = self.unet.config.in_channels
                self.train(False)

            def forward(
                self,
                latent,
                timestep,
                text_embedding,
                guidance_scale,
                control1,
                control2,
                control3,
                control4,
                control5,
                control6,
                control7,
                control8,
                control9,
                control10,
                control11,
                control12,
                control13,
                scale1,
                scale2,
                scale3,
                scale4,
                scale5,
                scale6,
                scale7,
                scale8,
                scale9,
                scale10,
                scale11,
                scale12,
                scale13,
            ):
                # TODO: Average pooling
                db_res_samples = [
                    control1,
                    control2,
                    control3,
                    control4,
                    control5,
                    control6,
                    control7,
                    control8,
                    control9,
                    control10,
                    control11,
                    control12,
                ]

                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                db_res_samples = tuple(
                    [
                        control1 * scale1,
                        control2 * scale2,
                        control3 * scale3,
                        control4 * scale4,
                        control5 * scale5,
                        control6 * scale6,
                        control7 * scale7,
                        control8 * scale8,
                        control9 * scale9,
                        control10 * scale10,
                        control11 * scale11,
                        control12 * scale12,
                    ]
                )
                mb_res_samples = control13 * scale13
                latents = torch.cat([latent] * 2)
                unet_out = self.unet.forward(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embedding,
                    down_block_additional_residuals=db_res_samples,
                    mid_block_additional_residual=mb_res_samples,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                return noise_pred

        unet = ControlledUnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False

        inputs = tuple(self.inputs["unet"])
        model_name = "stencil_unet"
        if use_large:
            pad = (0, 0) * (len(inputs[2].shape) - 2)
            pad = pad + (0, 512 - inputs[2].shape[1])
            inputs = (
                inputs[:2]
                + (torch.nn.functional.pad(inputs[2], pad),)
                + inputs[3:]
            )
            model_name = "stencil_unet_512"
        input_mask = [
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        shark_controlled_unet, controlled_unet_mlir = compile_through_fx(
            unet,
            inputs,
            extended_model_name=self.model_name[model_name],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name=model_name,
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_controlled_unet, controlled_unet_mlir

    def get_control_net(self, stencil_id, use_large=False):
        stencil_id = get_stencil_model_id(stencil_id)
        adapter_id, base_model_safe_id, ext_model_name = (None, None, None)
        print(f"Importing ControlNet adapter from {stencil_id}")

        class StencilControlNetModel(torch.nn.Module):
            def __init__(self, model_id=stencil_id, low_cpu_mem_usage=False):
                super().__init__()
                self.cnet = ControlNetModel.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                self.in_channels = self.cnet.config.in_channels
                self.train(False)

            def forward(
                self,
                latent,
                timestep,
                text_embedding,
                stencil_image_input,
                acc1,
                acc2,
                acc3,
                acc4,
                acc5,
                acc6,
                acc7,
                acc8,
                acc9,
                acc10,
                acc11,
                acc12,
                acc13,
            ):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                # TODO: guidance NOT NEEDED change in `get_input_info` later
                latents = torch.cat(
                    [latent] * 2
                )  # needs to be same as controlledUNET latents
                stencil_image = torch.cat(
                    [stencil_image_input] * 2
                )  # needs to be same as controlledUNET latents
                (
                    down_block_res_samples,
                    mid_block_res_sample,
                ) = self.cnet.forward(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embedding,
                    controlnet_cond=stencil_image,
                    return_dict=False,
                )
                return tuple(
                    list(down_block_res_samples) + [mid_block_res_sample]
                ) + (
                    acc1 + down_block_res_samples[0],
                    acc2 + down_block_res_samples[1],
                    acc3 + down_block_res_samples[2],
                    acc4 + down_block_res_samples[3],
                    acc5 + down_block_res_samples[4],
                    acc6 + down_block_res_samples[5],
                    acc7 + down_block_res_samples[6],
                    acc8 + down_block_res_samples[7],
                    acc9 + down_block_res_samples[8],
                    acc10 + down_block_res_samples[9],
                    acc11 + down_block_res_samples[10],
                    acc12 + down_block_res_samples[11],
                    acc13 + mid_block_res_sample,
                )

        scnet = StencilControlNetModel(
            low_cpu_mem_usage=self.low_cpu_mem_usage
        )
        is_f16 = True if self.precision == "fp16" else False

        inputs = tuple(self.inputs["stencil_adapter"])
        model_name = "stencil_adapter_512" if use_large else "stencil_adapter"
        stencil_names = self.get_extended_name_for_all_model([model_name])
        ext_model_name = stencil_names[model_name]
        if isinstance(ext_model_name, list):
            desired_name = None
            print(ext_model_name)
            for i in ext_model_name:
                if stencil_id.split("_")[-1] in i:
                    desired_name = i
                else:
                    continue
            if desired_name:
                ext_model_name = desired_name
            else:
                raise Exception(
                    f"Could not find extended configuration for {stencil_id}"
                )

        if use_large:
            pad = (0, 0) * (len(inputs[2].shape) - 2)
            pad = pad + (0, 512 - inputs[2].shape[1])
            inputs = (
                inputs[0],
                inputs[1],
                torch.nn.functional.pad(inputs[2], pad),
                *inputs[3:],
            )
        save_dir = os.path.join(self.sharktank_dir, ext_model_name)
        input_mask = [True, True, True, True] + ([True] * 13)

        shark_cnet, cnet_mlir = compile_through_fx(
            scnet,
            inputs,
            extended_model_name=ext_model_name,
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name=model_name,
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_cnet, cnet_mlir

    def get_unet(self, use_large=False):
        class UnetModel(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                low_cpu_mem_usage=False,
                use_lora=self.use_lora,
            ):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                if use_lora != "":
                    update_lora_weight(self.unet, use_lora, "unet")
                self.in_channels = self.unet.config.in_channels
                self.train(False)
                if (
                    args.attention_slicing is not None
                    and args.attention_slicing != "none"
                ):
                    if args.attention_slicing.isdigit():
                        self.unet.set_attention_slice(
                            int(args.attention_slicing)
                        )
                    else:
                        self.unet.set_attention_slice(args.attention_slicing)

            # TODO: Instead of flattening the `control` try to use the list.
            def forward(
                self,
                latent,
                timestep,
                text_embedding,
                guidance_scale,
            ):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latents = torch.cat([latent] * 2)
                unet_out = self.unet.forward(
                    latents, timestep, text_embedding, return_dict=False
                )[0]
                noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                return noise_pred

        unet = UnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False
        inputs = tuple(self.inputs["unet"])
        if use_large:
            pad = (0, 0) * (len(inputs[2].shape) - 2)
            pad = pad + (0, 512 - inputs[2].shape[1])
            inputs = (
                inputs[0],
                inputs[1],
                torch.nn.functional.pad(inputs[2], pad),
                inputs[3],
            )
            save_dir = os.path.join(
                self.sharktank_dir, self.model_name["unet512"]
            )
        else:
            save_dir = os.path.join(
                self.sharktank_dir, self.model_name["unet"]
            )
        input_mask = [True, True, True, False]
        if self.debug:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
        model_name = "unet512" if use_large else "unet"
        shark_unet, unet_mlir = compile_through_fx(
            unet,
            inputs,
            extended_model_name=self.model_name[model_name],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name=model_name,
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_unet, unet_mlir

    def get_unet_upscaler(self, use_large=False):
        class UnetModel(torch.nn.Module):
            def __init__(
                self, model_id=self.model_id, low_cpu_mem_usage=False
            ):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                self.in_channels = self.unet.in_channels
                self.train(False)

            def forward(self, latent, timestep, text_embedding, noise_level):
                unet_out = self.unet.forward(
                    latent,
                    timestep,
                    text_embedding,
                    noise_level,
                    return_dict=False,
                )[0]
                return unet_out

        unet = UnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False
        inputs = tuple(self.inputs["unet"])
        if use_large:
            pad = (0, 0) * (len(inputs[2].shape) - 2)
            pad = pad + (0, 512 - inputs[2].shape[1])
            inputs = (
                inputs[0],
                inputs[1],
                torch.nn.functional.pad(inputs[2], pad),
                inputs[3],
            )
        input_mask = [True, True, True, False]
        model_name = "unet512" if use_large else "unet"
        shark_unet, unet_mlir = compile_through_fx(
            unet,
            inputs,
            extended_model_name=self.model_name[model_name],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name=model_name,
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_unet, unet_mlir

    def get_unet_sdxl(self):
        # TODO: Remove this after convergence with shark_tank. This should just be part of
        #       opt_params.py.
        shark_module_or_none = process_vmfb_ir_sdxl(
            self.model_name["unet"], "unet", args.device, self.precision
        )
        if shark_module_or_none[0]:
            return shark_module_or_none

        class UnetModel(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                low_cpu_mem_usage=False,
            ):
                super().__init__()
                try:
                    self.unet = UNet2DConditionModel.from_pretrained(
                        model_id,
                        subfolder="unet",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                        variant="fp16",
                    )
                except:
                    self.unet = UNet2DConditionModel.from_pretrained(
                        model_id,
                        subfolder="unet",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                if (
                    args.attention_slicing is not None
                    and args.attention_slicing != "none"
                ):
                    if args.attention_slicing.isdigit():
                        self.unet.set_attention_slice(
                            int(args.attention_slicing)
                        )
                    else:
                        self.unet.set_attention_slice(args.attention_slicing)

            def forward(
                self,
                latent,
                timestep,
                prompt_embeds,
                text_embeds,
                time_ids,
                guidance_scale,
            ):
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                noise_pred = self.unet.forward(
                    latent,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                return noise_pred

        unet = UnetModel(low_cpu_mem_usage=self.low_cpu_mem_usage)
        is_f16 = True if self.precision == "fp16" else False
        inputs = tuple(self.inputs["unet"])
        save_dir = os.path.join(self.sharktank_dir, self.model_name["unet"])
        input_mask = [True, True, True, True, True, True]
        if self.debug:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
        shark_unet, unet_mlir = compile_through_fx(
            unet,
            inputs,
            extended_model_name=self.model_name["unet"],
            is_f16=is_f16,
            f16_input_mask=input_mask,
            use_tuned=self.use_tuned,
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("unet", precision=self.precision),
            base_model_id=self.base_model_id,
            model_name="unet",
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_unet, unet_mlir

    def get_clip(self):
        class CLIPText(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                low_cpu_mem_usage=False,
                use_lora=self.use_lora,
            ):
                super().__init__()
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                if use_lora != "":
                    update_lora_weight(
                        self.text_encoder, use_lora, "text_encoder"
                    )

            def forward(self, input):
                return self.text_encoder(input)[0]

        clip_model = CLIPText(low_cpu_mem_usage=self.low_cpu_mem_usage)
        save_dir = ""
        if self.debug:
            save_dir = os.path.join(
                self.sharktank_dir, self.model_name["clip"]
            )
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
        shark_clip, clip_mlir = compile_through_fx(
            clip_model,
            tuple(self.inputs["clip"]),
            extended_model_name=self.model_name["clip"],
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("clip", precision="fp32"),
            base_model_id=self.base_model_id,
            model_name="clip",
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_clip, clip_mlir

    def get_clip_sdxl(self, clip_index=1):
        if clip_index == 1:
            extended_model_name = self.model_name["clip"]
            model_name = "clip"
        else:
            extended_model_name = self.model_name["clip2"]
            model_name = "clip2"
        # TODO: Remove this after convergence with shark_tank. This should just be part of
        #       opt_params.py.
        shark_module_or_none = process_vmfb_ir_sdxl(
            extended_model_name, f"clip", args.device, self.precision
        )
        if shark_module_or_none[0]:
            return shark_module_or_none

        class CLIPText(torch.nn.Module):
            def __init__(
                self,
                model_id=self.model_id,
                low_cpu_mem_usage=False,
                clip_index=1,
            ):
                super().__init__()
                if clip_index == 1:
                    self.text_encoder = CLIPTextModel.from_pretrained(
                        model_id,
                        subfolder="text_encoder",
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )
                else:
                    self.text_encoder = (
                        CLIPTextModelWithProjection.from_pretrained(
                            model_id,
                            subfolder="text_encoder_2",
                            low_cpu_mem_usage=low_cpu_mem_usage,
                        )
                    )

            def forward(self, input):
                prompt_embeds = self.text_encoder(
                    input,
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                return prompt_embeds, pooled_prompt_embeds

        clip_model = CLIPText(
            low_cpu_mem_usage=self.low_cpu_mem_usage, clip_index=clip_index
        )
        save_dir = os.path.join(self.sharktank_dir, extended_model_name)
        if self.debug:
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
        shark_clip, clip_mlir = compile_through_fx(
            clip_model,
            tuple(self.inputs["clip"]),
            extended_model_name=extended_model_name,
            debug=self.debug,
            generate_vmfb=self.generate_vmfb,
            save_dir=save_dir,
            extra_args=get_opt_flags("clip", precision="fp32"),
            base_model_id=self.base_model_id,
            model_name="clip",
            precision=self.precision,
            return_mlir=self.return_mlir,
        )
        return shark_clip, clip_mlir

    def process_custom_vae(self):
        custom_vae = self.custom_vae.lower()
        if not custom_vae.endswith((".ckpt", ".safetensors")):
            return self.custom_vae
        try:
            preprocessCKPT(self.custom_vae)
            return get_path_to_diffusers_checkpoint(self.custom_vae)
        except:
            print("Processing standalone Vae checkpoint")
            vae_checkpoint = None
            vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}
            if custom_vae.endswith(".ckpt"):
                vae_checkpoint = torch.load(
                    self.custom_vae, map_location="cpu"
                )
            else:
                vae_checkpoint = safetensors.torch.load_file(
                    self.custom_vae, device="cpu"
                )
            if "state_dict" in vae_checkpoint:
                vae_checkpoint = vae_checkpoint["state_dict"]

            try:
                vae_checkpoint = convert_original_vae(vae_checkpoint)
            finally:
                vae_dict = {
                    k: v
                    for k, v in vae_checkpoint.items()
                    if k[0:4] != "loss" and k not in vae_ignore_keys
                }
                return vae_dict

    def compile_unet_variants(self, model, use_large=False, base_model=""):
        if self.is_sdxl:
            return self.get_unet_sdxl()
        if model == "unet":
            if self.is_upscaler:
                return self.get_unet_upscaler(use_large=use_large)
            # TODO: Plug the experimental "int8" support at right place.
            elif self.use_quantize == "int8":
                from apps.stable_diffusion.src.models.opt_params import (
                    get_unet,
                )

                return get_unet()
            else:
                return self.get_unet(use_large=use_large)
        else:
            return self.get_controlled_unet(use_large=use_large)

    def vae_encode(self):
        try:
            self.inputs["vae_encode"] = self.get_input_info_for(
                base_models["vae_encode"]
            )
            compiled_vae_encode, vae_encode_mlir = self.get_vae_encode()

            check_compilation(compiled_vae_encode, "Vae Encode")
            if self.return_mlir:
                return vae_encode_mlir
            return compiled_vae_encode
        except Exception as e:
            sys.exit(e)

    def clip(self):
        try:
            self.inputs["clip"] = self.get_input_info_for(base_models["clip"])
            compiled_clip, clip_mlir = self.get_clip()

            check_compilation(compiled_clip, "Clip")
            if self.return_mlir:
                return clip_mlir
            return compiled_clip
        except Exception as e:
            sys.exit(e)

    def sdxl_clip(self):
        try:
            self.inputs["clip"] = self.get_input_info_for(
                base_models["sdxl_clip"]
            )
            compiled_clip, clip_mlir = self.get_clip_sdxl(clip_index=1)
            compiled_clip2, clip_mlir2 = self.get_clip_sdxl(clip_index=2)

            check_compilation(compiled_clip, "Clip")
            check_compilation(compiled_clip, "Clip2")
            if self.return_mlir:
                return clip_mlir, clip_mlir2
            return compiled_clip, compiled_clip2
        except Exception as e:
            sys.exit(e)

    def unet(self, use_large=False):
        try:
            stencil_count = 0
            for stencil in self.stencils:
                stencil_count += 1
            model = "stencil_unet" if stencil_count > 0 else "unet"
            compiled_unet = None
            unet_inputs = base_models[model]

            if self.base_model_id != "":
                self.inputs["unet"] = self.get_input_info_for(
                    unet_inputs[self.base_model_id]
                )
                compiled_unet, unet_mlir = self.compile_unet_variants(
                    model, use_large=use_large, base_model=self.base_model_id
                )
            else:
                for model_id in unet_inputs:
                    self.base_model_id = model_id
                    self.inputs["unet"] = self.get_input_info_for(
                        unet_inputs[model_id]
                    )

                    try:
                        compiled_unet, unet_mlir = self.compile_unet_variants(
                            model, use_large=use_large, base_model=model_id
                        )
                    except Exception as e:
                        print(e)
                        print(
                            "Retrying with a different base model configuration"
                        )
                        continue

                    # -- Once a successful compilation has taken place we'd want to store
                    #    the base model's configuration inferred.
                    fetch_and_update_base_model_id(self.model_to_run, model_id)
                    # This is done just because in main.py we are basing the choice of tokenizer and scheduler
                    # on `args.hf_model_id`. Since now, we don't maintain 1:1 mapping of variants and the base
                    # model and rely on retrying method to find the input configuration, we should also update
                    # the knowledge of base model id accordingly into `args.hf_model_id`.
                    if args.ckpt_loc != "":
                        args.hf_model_id = model_id
                    break

            check_compilation(compiled_unet, "Unet")
            if self.return_mlir:
                return unet_mlir
            return compiled_unet
        except Exception as e:
            sys.exit(e)

    def vae(self):
        try:
            vae_input = (
                base_models["vae"]["vae_upscaler"]
                if self.is_upscaler
                else base_models["vae"]["vae"]
            )
            self.inputs["vae"] = self.get_input_info_for(vae_input)

            is_base_vae = self.base_vae
            if self.is_upscaler:
                self.base_vae = True
            if self.is_sdxl:
                compiled_vae, vae_mlir = self.get_vae_sdxl()
            else:
                compiled_vae, vae_mlir = self.get_vae()
            self.base_vae = is_base_vae

            check_compilation(compiled_vae, "Vae")
            if self.return_mlir:
                return vae_mlir
            return compiled_vae
        except Exception as e:
            sys.exit(e)

    def controlnet(self, stencil_id, use_large=False):
        try:
            self.inputs["stencil_adapter"] = self.get_input_info_for(
                base_models["stencil_adapter"]
            )
            compiled_stencil_adapter, controlnet_mlir = self.get_control_net(
                stencil_id, use_large=use_large
            )

            check_compilation(compiled_stencil_adapter, "Stencil")
            if self.return_mlir:
                return controlnet_mlir
            return compiled_stencil_adapter
        except Exception as e:
            sys.exit(e)
