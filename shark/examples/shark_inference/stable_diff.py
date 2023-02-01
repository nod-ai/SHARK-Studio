from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from shark.shark_inference import SharkInference
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
import torch_mlir
import tempfile
import numpy as np

# pip install diffusers
# pip install scipy

############### Parsing args #####################
import argparse

p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

p.add_argument(
    "--prompt",
    type=str,
    default="a photograph of an astronaut riding a horse",
    help="the text prompt to use",
)
p.add_argument("--device", type=str, default="cpu", help="the device to use")
p.add_argument("--steps", type=int, default=10, help="the device to use")
p.add_argument("--mlir_loc", type=str, default=None, help="the device to use")
p.add_argument("--vae_loc", type=str, default=None, help="the device to use")
args = p.parse_args()

#####################################################


def load_mlir(mlir_loc):
    import os

    if mlir_loc == None:
        return None
    print(f"Trying to load the model from {mlir_loc}.")
    with open(os.path.join(mlir_loc)) as f:
        mlir_module = f.read()
    return mlir_module


def compile_through_fx(model, inputs, mlir_loc=None, extra_args=[]):
    module = load_mlir(mlir_loc)
    if mlir_loc == None:
        fx_g = make_fx(
            model,
            decomposition_table=get_decompositions(
                [
                    torch.ops.aten.embedding_dense_backward,
                    torch.ops.aten.native_layer_norm_backward,
                    torch.ops.aten.slice_backward,
                    torch.ops.aten.select_backward,
                    torch.ops.aten.norm.ScalarOpt_dim,
                    torch.ops.aten.native_group_norm,
                    torch.ops.aten.upsample_bilinear2d.vec,
                    torch.ops.aten.split.Tensor,
                    torch.ops.aten.split_with_sizes,
                ]
            ),
        )(*inputs)

        fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
        fx_g.recompile()

        def strip_overloads(gm):
            """
            Modifies the target of graph nodes in :attr:`gm` to strip overloads.
            Args:
                gm(fx.GraphModule): The input Fx graph module to be modified
            """
            for node in gm.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverload):
                    node.target = node.target.overloadpacket
            gm.recompile()

        strip_overloads(fx_g)

        ts_g = torch.jit.script(fx_g)

        module = torch_mlir.compile(
            ts_g,
            inputs,
            torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=False,
            verbose=False,
        )

    mlir_model = module
    func_name = "forward"

    shark_module = SharkInference(
        mlir_model,
        func_name,
        device=args.device,
        mlir_dialect="tm_tensor",
    )
    shark_module.compile(extra_args)

    return shark_module


if __name__ == "__main__":
    YOUR_TOKEN = "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"

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        use_auth_token=YOUR_TOKEN,
    )

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                use_auth_token=YOUR_TOKEN,
            )

        def forward(self, input):
            return self.vae.decode(input, return_dict=False)[0]

    vae = VaeModel()
    vae_input = torch.rand(1, 4, 64, 64)
    shark_vae = compile_through_fx(vae, (vae_input,), args.vae_loc)

    # Wrap the unet model to return tuples.
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

        def forward(self, x, y, z):
            return self.unet.forward(x, y, z, return_dict=False)[0]

    # 3. The UNet model for generating the latents.
    unet = UnetModel()
    latent_model_input = torch.rand([2, 4, 64, 64])
    text_embeddings = torch.rand([2, 77, 768])
    shark_unet = compile_through_fx(
        unet,
        (latent_model_input, torch.tensor([1.0]), text_embeddings),
        args.mlir_loc,
        ["--iree-flow-enable-conv-nchw-to-nhwc-transform"],
    )

    # torch.jit.script(unet)

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    prompt = [args.prompt]

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = args.steps  # Number of denoising steps

    guidance_scale = 7.5  # Scale for classifier-free guidance

    generator = torch.manual_seed(
        42
    )  # Seed generator to create the inital latent noise

    batch_size = len(prompt)

    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = text_encoder(text_input.input_ids)[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    # latents = latents.to(torch_device)

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.sigmas[0]
    # print(latents, latents.shape)

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        print(f"i = {i} t = {t}")
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual

        # with torch.no_grad():
        # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

        latent_model_input_numpy = latent_model_input.detach().numpy()
        text_embeddings_numpy = text_embeddings.detach().numpy()

        noise_pred = shark_unet.forward(
            (
                latent_model_input_numpy,
                np.array([t]).astype(np.float32),
                text_embeddings_numpy,
            )
        )
        noise_pred = torch.from_numpy(noise_pred)

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    # print("Latents shape : ", latents.shape)

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    latents_numpy = latents.detach().numpy()
    image = shark_vae.forward((latents_numpy,))
    image = torch.from_numpy(image)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save("astro.jpg")
