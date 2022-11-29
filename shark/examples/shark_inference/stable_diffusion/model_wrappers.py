from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel
from utils import compile_through_fx
from stable_args import args
import torch

YOUR_TOKEN = "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"


BATCH_SIZE = len(args.prompts)


def get_clipped_text(model_name="clip_text", extra_args=[]):
    class CLIPText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            )

        def forward(self, input):
            return self.text_encoder(input)[0]

    clip_model = CLIPText()
    clip_input = torch.randint(1, 2, (BATCH_SIZE, 77))
    shark_clip = compile_through_fx(
        clip_model,
        (clip_input,),
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_clip


def get_vae32(model_name="vae_fp32", extra_args=[]):
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                use_auth_token=YOUR_TOKEN,
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

    vae = VaeModel()
    vae_input = torch.rand(BATCH_SIZE, 4, 64, 64)
    shark_vae = compile_through_fx(
        vae,
        (vae_input,),
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_vae


def get_vae16(model_name="vae_fp16", extra_args=[]):
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                use_auth_token=YOUR_TOKEN,
                revision="fp16",
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return (x / 2 + 0.5).clamp(0, 1)

    vae = VaeModel()
    vae = vae.half().cuda()
    vae_input = torch.rand(BATCH_SIZE, 4, 64, 64, dtype=torch.half).cuda()
    shark_vae = compile_through_fx(
        vae,
        (vae_input,),
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_vae


def get_unet32(model_name="unet_fp32_wrapped", extra_args=[]):
    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="unet",
                use_auth_token=YOUR_TOKEN,
            )
            self.in_channels = self.unet.in_channels
            self.train(False)

        def forward(self, latent, timestep, text_embedding, guidance_scale):
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
    latent_model_input = torch.rand([BATCH_SIZE, 4, 64, 64])
    text_embeddings = torch.rand([2 * BATCH_SIZE, args.max_length, 768])
    guidance_scale = torch.tensor(1).to(torch.float32)
    shark_unet = compile_through_fx(
        unet,
        (
            latent_model_input,
            torch.tensor([1.0]),
            text_embeddings,
            guidance_scale,
        ),
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_unet


def get_unet16(model_name="unet_fp16", extra_args=[]):
    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="unet",
                use_auth_token=YOUR_TOKEN,
                revision="fp16",
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
    unet = unet.half().cuda()
    latent_model_input = torch.rand([BATCH_SIZE, 4, 64, 64]).half().cuda()
    text_embeddings = (
        torch.rand([2 * BATCH_SIZE, args.max_length, 768]).half().cuda()
    )
    guidance_scale = torch.tensor(1).to(torch.float32)
    shark_unet = compile_through_fx(
        unet,
        (
            latent_model_input,
            torch.tensor([1.0]).half().cuda(),
            text_embeddings,
            guidance_scale,
        ),
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_unet
