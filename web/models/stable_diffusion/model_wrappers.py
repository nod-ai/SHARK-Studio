from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
from models.stable_diffusion.utils import compile_through_fx
from models.stable_diffusion.resources import models_config
from models.stable_diffusion.stable_args import args
import numpy as np
import torch


# clip has 2 variants of max length 77 or 64.
model_clip_max_length = 64 if args.max_length == 64 else 77
if args.variant in ["anythingv3", "analogdiffusion", "dreamlike"]:
    model_clip_max_length = 77
elif args.variant == "openjourney":
    model_clip_max_length = 64

model_input = {
    "v2_1": {
        "clip": (torch.randint(1, 2, (2, model_clip_max_length)),),
        "vae_encode": (torch.randn(1, 768, 768, 3),),
        "vae": (torch.randn(1, 4, 96, 96),),
        "unet": (
            torch.randn(1, 4, 96, 96),  # latents
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, model_clip_max_length, 1024),  # embedding
            torch.tensor(1).to(torch.float32),  # guidance_scale
        ),
    },
    "v2_1base": {
        "clip": (torch.randint(1, 2, (2, model_clip_max_length)),),
        "vae_encode": (torch.randn(1, 512, 512, 3),),
        "vae": (torch.randn(1, 4, 64, 64),),
        "unet": (
            torch.randn(1, 4, 64, 64),  # latents
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, model_clip_max_length, 1024),  # embedding
            torch.tensor(1).to(torch.float32),  # guidance_scale
        ),
    },
    "v1_4": {
        "clip": (torch.randint(1, 2, (2, model_clip_max_length)),),
        "vae_encode": (torch.randn(1, 512, 512, 3),),
        "vae": (torch.randn(1, 4, 64, 64),),
        "unet": (
            torch.randn(1, 4, 64, 64),
            torch.tensor([1]).to(torch.float32),  # timestep
            torch.randn(2, model_clip_max_length, 768),
            torch.tensor(1).to(torch.float32),
        ),
    },
}

version = args.version if args.variant == "stablediffusion" else "v1_4"


def get_configs():
    model_id_key = f"{args.variant}/{version}"
    revision_key = f"{args.variant}/{args.precision}"
    try:
        model_id = models_config[0][model_id_key]
        revision = models_config[1][revision_key]
    except KeyError:
        raise Exception(
            f"No entry for {model_id_key} or {revision_key} in the models configuration"
        )

    return model_id, revision


def get_clip_mlir(model_name="clip_text", extra_args=[]):
    model_id, revision = get_configs()

    class CLIPText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                revision=revision,
            )

        def forward(self, input):
            return self.text_encoder(input)[0]

    clip_model = CLIPText()
    shark_clip = compile_through_fx(
        clip_model,
        model_input[version]["clip"],
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_clip


def get_shark_module(model_key, module, model_name, extra_args):
    if args.precision == "fp16":
        module = module.half().cuda()
        inputs = tuple(
            [
                inputs.half().cuda() if len(inputs.shape) != 0 else inputs
                for inputs in model_input[version][model_key]
            ]
        )
    else:
        inputs = model_input[version][model_key]

    shark_module = compile_through_fx(
        module,
        inputs,
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_module


def get_vae_encode_mlir(model_name="vae_encode", extra_args=[]):
    model_id, revision = get_configs()

    class VaeEncodeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
                revision=revision,
            )

        def forward(self, input):
            input_arr = np.stack([np.array(i) for i in input.cpu()], axis=0)
            input_arr = input_arr / 255.0
            input_arr = torch.from_numpy(input_arr).permute(0, 3, 1, 2)
            input_arr = 2 * (input_arr - 0.5)
            latent_dists = self.vae.encode(input_arr.cuda())["latent_dist"]
            latent_samples = latent_dists.sample()
            return latent_samples * 0.18215

    vae_encode = VaeEncodeModel()
    return get_shark_module("vae_encode", vae_encode, model_name, extra_args)


def get_base_vae_mlir(model_name="vae", extra_args=[]):
    model_id, revision = get_configs()

    class BaseVaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
                revision=revision,
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

    vae = BaseVaeModel()
    return get_shark_module("vae", vae, model_name, extra_args)


def get_vae_mlir(model_name="vae", extra_args=[]):
    model_id, revision = get_configs()

    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
                revision=revision,
            )

        def forward(self, input):
            input = 1 / 0.18215 * input
            x = self.vae.decode(input, return_dict=False)[0]
            x = (x / 2 + 0.5).clamp(0, 1)
            x = x * 255.0
            return x.round()

    vae = VaeModel()
    return get_shark_module("vae", vae, model_name, extra_args)


def get_unet_mlir(model_name="unet", extra_args=[]):
    model_id, revision = get_configs()

    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                revision=revision,
            )
            self.in_channels = self.unet.in_channels
            self.train(False)

        def forward(self, latent, timestep, text_embedding, guidance_scale):
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

    unet = UnetModel()
    return get_shark_module("unet", unet, model_name, extra_args)
