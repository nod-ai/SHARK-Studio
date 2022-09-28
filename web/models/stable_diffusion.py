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
import os

##############################################################################


def load_mlir(mlir_loc):
    if mlir_loc == None:
        return None
    with open(os.path.join(mlir_loc)) as f:
        mlir_module = f.read()
    return mlir_module


def compile_through_fx(model, inputs, device, mlir_loc=None):

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
        mlir_model, func_name, device=device, mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    return shark_module


##############################################################################

DEBUG = False
compiled_module = {}


def stable_diff_inf(prompt: str, steps, device: str):

    args = {}
    args["prompt"] = [prompt]
    args["steps"] = steps
    args["device"] = device
    args["mlir_loc"] = "./stable_diffusion.mlir"
    output_loc = (
        f"stored_results/stable_diffusion/{prompt}_{int(steps)}_{device}.jpg"
    )

    global DEBUG
    global compiled_module

    DEBUG = False
    log_write = open(r"logs/stable_diffusion_log.txt", "w")
    if log_write:
        DEBUG = True

    if args["device"] not in compiled_module.keys():
        YOUR_TOKEN = "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"

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        compiled_module["vae"] = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            use_auth_token=YOUR_TOKEN,
        )

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        compiled_module["tokenizer"] = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        compiled_module["text_encoder"] = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        if DEBUG:
            log_write.write("Compiling the Unet module.\n")

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
            args["device"],
            args["mlir_loc"],
        )
        compiled_module[args["device"]] = shark_unet
        if DEBUG:
            log_write.write("Compilation successful.\n")

        compiled_module["unet"] = unet
        compiled_module["scheduler"] = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

    shark_unet = compiled_module[args["device"]]
    vae = compiled_module["vae"]
    unet = compiled_module["unet"]
    tokenizer = compiled_module["tokenizer"]
    text_encoder = compiled_module["text_encoder"]
    scheduler = compiled_module["scheduler"]

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion

    num_inference_steps = int(args["steps"])  # Number of denoising steps

    guidance_scale = 7.5  # Scale for classifier-free guidance

    generator = torch.manual_seed(
        42
    )  # Seed generator to create the inital latent noise

    batch_size = len(args["prompt"])

    text_input = tokenizer(
        args["prompt"],
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
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    for i, t in tqdm(enumerate(scheduler.timesteps)):

        if DEBUG:
            log_write.write(f"i = {i} t = {t}\n")
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
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

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    output = pil_images[0]
    # save the output image with the prompt name.
    output.save(os.path.join(output_loc))
    log_write.close()

    std_output = ""
    with open(r"logs/stable_diffusion_log.txt", "r") as log_read:
        std_output = log_read.read()

    return output, std_output
