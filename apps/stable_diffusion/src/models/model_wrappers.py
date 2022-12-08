from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel
from collections import defaultdict
import torch
import sys
import traceback
import re
from ..utils import compile_through_fx, get_opt_flags, base_models, args


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
            if "batch_size" in shape[i]:
                mul_val = int(shape[i].split("*")[0])
                new_shape.append(batch_size * mul_val)
        else:
            new_shape.append(shape[i])
    return new_shape


# Get the input info for various models i.e. "unet", "clip", "vae".
def get_input_info(model_info, max_len, width, height, batch_size):
    dtype_config = {"f32": torch.float32, "i64": torch.int64}
    input_map = defaultdict(list)
    for k in model_info:
        for inp in model_info[k]:
            shape = model_info[k][inp]["shape"]
            dtype = dtype_config[model_info[k][inp]["dtype"]]
            tensor = None
            if isinstance(shape, list):
                clean_shape = replace_shape_str(
                    shape, max_len, width, height, batch_size
                )
                if dtype == torch.int64:
                    tensor = torch.randint(1, 3, tuple(clean_shape))
                else:
                    tensor = torch.randn(*clean_shape).to(dtype)
            elif isinstance(shape, int):
                tensor = torch.tensor(shape).to(dtype)
            else:
                sys.exit("shape isn't specified correctly.")
            input_map[k].append(tensor)
    return input_map


class SharkifyStableDiffusionModel:
    def __init__(
        self,
        model_id: str,
        custom_weights: str,
        precision: str,
        max_len: int = 64,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        use_base_vae: bool = False,
    ):
        self.check_params(max_len, width, height)
        self.max_len = max_len
        self.height = height // 8
        self.width = width // 8
        self.batch_size = batch_size
        self.model_id = model_id if custom_weights == "" else custom_weights
        self.precision = precision
        self.base_vae = use_base_vae
        self.model_name = (
            str(batch_size)
            + "_"
            + str(max_len)
            + "_"
            + str(height)
            + "_"
            + str(width)
            + "_"
            + precision
        )
        # We need a better naming convention for the .vmfbs because despite
        # using the custom model variant the .vmfb names remain the same and
        # it'll always pick up the compiled .vmfb instead of compiling the
        # custom model.
        # So, currently, we add `self.model_id` in the `self.model_name` of
        # .vmfb file.
        # TODO: Have a better way of naming the vmfbs using self.model_name.

        model_name = re.sub(r"\W+", "_", self.model_id)
        if model_name[0] == "_":
            model_name = model_name[1:]
        self.model_name = self.model_name + "_" + model_name

    def check_params(self, max_len, width, height):
        if not (max_len >= 32 and max_len <= 77):
            sys.exit("please specify max_len in the range [32, 77].")
        if not (width % 8 == 0 and width >= 384):
            sys.exit("width should be greater than 384 and multiple of 8")
        if not (height % 8 == 0 and height >= 384):
            sys.exit("height should be greater than 384 and multiple of 8")

    def get_vae(self):
        class VaeModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id, base_vae=self.base_vae):
                super().__init__()
                self.vae = AutoencoderKL.from_pretrained(
                    model_id,
                    subfolder="vae",
                )
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

        vae = VaeModel()
        inputs = tuple(self.inputs["vae"])
        is_f16 = True if self.precision == "fp16" else False
        vae_name = "base_vae" if self.base_vae else "vae"
        shark_vae = compile_through_fx(
            vae,
            inputs,
            is_f16=is_f16,
            model_name=vae_name + self.model_name,
            extra_args=get_opt_flags("vae", precision=self.precision),
        )
        return shark_vae

    def get_unet(self):
        class UnetModel(torch.nn.Module):
            def __init__(self, model_id=self.model_id):
                super().__init__()
                self.unet = UNet2DConditionModel.from_pretrained(
                    model_id,
                    subfolder="unet",
                )
                self.in_channels = self.unet.in_channels
                self.train(False)

            def forward(
                self, latent, timestep, text_embedding, guidance_scale
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

        unet = UnetModel()
        is_f16 = True if self.precision == "fp16" else False
        inputs = tuple(self.inputs["unet"])
        input_mask = [True, True, True, False]
        shark_unet = compile_through_fx(
            unet,
            inputs,
            model_name="unet" + self.model_name,
            is_f16=is_f16,
            f16_input_mask=input_mask,
            extra_args=get_opt_flags("unet", precision=self.precision),
        )
        return shark_unet

    def get_clip(self):
        class CLIPText(torch.nn.Module):
            def __init__(self, model_id=self.model_id):
                super().__init__()
                self.text_encoder = CLIPTextModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                )

            def forward(self, input):
                return self.text_encoder(input)[0]

        clip_model = CLIPText()

        shark_clip = compile_through_fx(
            clip_model,
            tuple(self.inputs["clip"]),
            model_name="clip" + self.model_name,
            extra_args=get_opt_flags("clip", precision="fp32"),
        )
        return shark_clip

    def __call__(self):

        for model_id in base_models:
            self.inputs = get_input_info(
                base_models[model_id],
                self.max_len,
                self.width,
                self.height,
                self.batch_size,
            )
            try:
                compiled_clip = self.get_clip()
                compiled_unet = self.get_unet()
                compiled_vae = self.get_vae()
            except Exception as e:
                if args.enable_stack_trace:
                    traceback.print_exc()
                print("Retrying with a different base model configuration")
                continue
            # This is done just because in main.py we are basing the choice of tokenizer and scheduler
            # on `args.hf_model_id`. Since now, we don't maintain 1:1 mapping of variants and the base
            # model and rely on retrying method to find the input configuration, we should also update
            # the knowledge of base model id accordingly into `args.hf_model_id`.
            if args.ckpt_loc != "":
                args.hf_model_id = model_id
            return compiled_clip, compiled_unet, compiled_vae
        sys.exit(
            "Cannot compile the model. Please use `enable_stack_trace` and create an issue at https://github.com/nod-ai/SHARK/issues"
        )
