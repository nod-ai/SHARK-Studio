import torch
from diffusers import (
    UNet2DConditionModel,
)
from torch.fx.experimental.proxy_tensor import make_fx


class UnetModel(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
        )

    def forward(self, sample, timestep, encoder_hidden_states, guidance_scale):
        samples = torch.cat([sample] * 2)
        unet_out = self.unet.forward(
            samples, timestep, encoder_hidden_states, return_dict=False
        )[0]
        noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred


if __name__ == "__main__":
    hf_model_name = "CompVis/stable-diffusion-v1-4"
    unet = UnetModel(hf_model_name)
    inputs = (torch.randn(1, 4, 64, 64), 1, torch.randn(2, 77, 768), 7.5)

    fx_g = make_fx(
        unet,
        decomposition_table={},
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        _allow_fake_constant=False,
    )(*inputs)

    print(fx_g)
