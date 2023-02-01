from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
from utils import compile_through_fx
import torch

model_id = "stabilityai/stable-diffusion-x4-upscaler"

model_input = {
    "clip": (torch.randint(1, 2, (1, 77)),),
    "vae": (torch.randn(1, 4, 128, 128),),
    "unet": (
        torch.randn(2, 7, 128, 128),  # latents
        torch.tensor([1]).to(torch.float32),  # timestep
        torch.randn(2, 77, 1024),  # embedding
        torch.randn(2).to(torch.int64),  # noise_level
    ),
}


def get_clip_mlir(model_name="clip_text", extra_args=[]):
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
    )

    class CLIPText(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, input):
            return self.text_encoder(input)[0]

    clip_model = CLIPText()
    shark_clip = compile_through_fx(
        clip_model,
        model_input["clip"],
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_clip


def get_vae_mlir(model_name="vae", extra_args=[]):
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
            )

        def forward(self, input):
            x = self.vae.decode(input, return_dict=False)[0]
            return x

    vae = VaeModel()
    shark_vae = compile_through_fx(
        vae,
        model_input["vae"],
        model_name=model_name,
        extra_args=extra_args,
    )
    return shark_vae


def get_unet_mlir(model_name="unet", extra_args=[]):
    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
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

    unet = UnetModel()
    f16_input_mask = (True, True, True, False)
    shark_unet = compile_through_fx(
        unet,
        model_input["unet"],
        model_name=model_name,
        is_f16=True,
        f16_input_mask=f16_input_mask,
        extra_args=extra_args,
    )
    return shark_unet
