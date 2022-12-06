import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras_cv.models.generative.stable_diffusion.clip_tokenizer import (
    SimpleTokenizer,
)
from keras_cv.models.generative.stable_diffusion.constants import (
    _ALPHAS_CUMPROD,
)
from keras_cv.models.generative.stable_diffusion.constants import (
    _UNCONDITIONAL_TOKENS,
)
from keras_cv.models.generative.stable_diffusion.decoder import Decoder
from keras_cv.models.generative.stable_diffusion.text_encoder import (
    TextEncoder,
)

from shark.shark_inference import SharkInference
from shark.shark_downloader import download_tf_model
from PIL import Image

# pip install "git+https://github.com/keras-team/keras-cv.git"
# pip install tensorflow_dataset

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
p.add_argument(
    "--steps", type=int, default=10, help="the number of steps to use"
)
p.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="the file to save the resulting image to. (default to <input prompt>.jpg)",
)
args = p.parse_args()

#####################################################

MAX_PROMPT_LENGTH = 77


class SharkStableDiffusion:
    """Shark implementation of Stable Diffusion based on model from keras_cv.
    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text description
    (called a "prompt").
    Arguments:
        device: Device to use with SHARK. Default: cpu
        jit_compile: Whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Default: False.
    References:
    - [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
    - [Original implementation](https://github.com/CompVis/stable-diffusion)
    """

    def __init__(self, device="cpu", jit_compile=True):
        self.img_height = 512
        self.img_width = 512
        self.tokenizer = SimpleTokenizer()

        # Create models
        self.text_encoder = TextEncoder(MAX_PROMPT_LENGTH)

        mlir_model, func_name, inputs, golden_out = download_tf_model(
            "stable_diff", tank_url="gs://shark_tank/quinn"
        )
        shark_module = SharkInference(
            mlir_model, func_name, device=device, mlir_dialect="mhlo"
        )
        shark_module.compile()
        self.diffusion_model = shark_module
        self.decoder = Decoder(self.img_height, self.img_width)
        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)

        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL-M license at "
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE"
        )
        # Load weights
        text_encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
            file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
        )
        decoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
            file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
        )
        self.text_encoder.load_weights(text_encoder_weights_fpath)
        self.decoder.load_weights(decoder_weights_fpath)

    def text_to_image(
        self,
        prompt,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        seed=None,
    ):
        encoded_text = self.encode_text(prompt)

        return self.generate_image(
            encoded_text,
            batch_size=batch_size,
            num_steps=num_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            seed=seed,
        )

    def encode_text(self, prompt):
        """Encodes a prompt into a latent text encoding.
        The encoding produced by this method should be used as the
        `encoded_text` parameter of `StableDiffusion.generate_image`. Encoding
        text separately from generating an image can be used to arbitrarily
        modify the text encoding priot to image generation, e.g. for walking
        between two prompts.
        Args:
            prompt: a string to encode, must be 77 tokens or shorter.
        Example:
        ```python
        from keras_cv.models import StableDiffusion
        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        encoded_text  = model.encode_text("Tacos at dawn")
        img = model.generate_image(encoded_text)
        ```
        """
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

        context = self.text_encoder.predict_on_batch(
            [phrase, self._get_pos_ids()]
        )

        return context

    def generate_image(
        self,
        encoded_text,
        batch_size=1,
        num_steps=25,
        unconditional_guidance_scale=7.5,
        diffusion_noise=None,
        seed=None,
    ):
        """Generates an image based on encoded text.
        The encoding passed to this method should be derived from
        `StableDiffusion.encode_text`.
        Args:
            encoded_text: Tensor of shape (`batch_size`, 77, 768), or a Tensor
            of shape (77, 768). When the batch axis is omitted, the same encoded
            text will be used to produce every generated image.
            batch_size: number of images to generate. Default: 1.
            num_steps: number of diffusion steps (controls image quality).
                Default: 25.
            unconditional_guidance_scale: float controling how closely the image
                should adhere to the prompt. Larger values result in more
                closely adhering to the prompt, but will make the image noisier.
                Default: 7.5.
            diffusion_noise: Tensor of shape (`batch_size`, img_height // 8,
                img_width // 8, 4), or a Tensor of shape (img_height // 8,
                img_width // 8, 4). Optional custom noise to seed the diffusion
                process. When the batch axis is omitted, the same noise will be
                used to seed diffusion for every generated image.
            seed: integer which is used to seed the random generation of
                diffusion noise, only to be specified if `diffusion_noise` is
                None.
        Example:
        ```python
        from keras_cv.models import StableDiffusion
        batch_size = 8
        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        e_tacos = model.encode_text("Tacos at dawn")
        e_watermelons = model.encode_text("Watermelons at dusk")
        e_interpolated = tf.linspace(e_tacos, e_watermelons, batch_size)
        images = model.generate_image(e_interpolated, batch_size=batch_size)
        ```
        """
        if diffusion_noise is not None and seed is not None:
            raise ValueError(
                "`diffusion_noise` and `seed` should not both be passed to "
                "`generate_image`. `seed` is only used to generate diffusion "
                "noise when it's not already user-specified."
            )

        encoded_text = tf.squeeze(encoded_text)
        if encoded_text.shape.rank == 2:
            encoded_text = tf.repeat(
                tf.expand_dims(encoded_text, axis=0), batch_size, axis=0
            )

        context = encoded_text
        unconditional_context = tf.repeat(
            self._get_unconditional_context(), batch_size, axis=0
        )
        context = tf.concat([context, unconditional_context], 0)

        if diffusion_noise is not None:
            diffusion_noise = tf.squeeze(diffusion_noise)
            if diffusion_noise.shape.rank == 3:
                diffusion_noise = tf.repeat(
                    tf.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Iterative reverse diffusion stage
        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)

            # Prepare the latent and unconditional latent to be run with a single forward call
            latent = tf.concat([latent, latent], 0)
            t_emb = tf.concat([t_emb, t_emb], 0)
            latent_numpy = self.diffusion_model.forward(
                [latent.numpy(), t_emb.numpy(), context.numpy()]
            )
            latent = tf.convert_to_tensor(latent_numpy, dtype=tf.float32)
            latent, unconditional_latent = tf.split(latent, 2)

            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(
                a_t
            )
            latent = (
                latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            )
            iteration += 1
            progbar.update(iteration)

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def _get_unconditional_context(self):
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, self._get_pos_ids()]
        )

        return unconditional_context

    def _get_timestep_embedding(
        self, timestep, batch_size, dim=320, max_period=10000
    ):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size, seed):
        return tf.random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4),
            seed=seed,
        )

    @staticmethod
    def _get_pos_ids():
        return tf.convert_to_tensor(
            [list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32
        )


if __name__ == "__main__":
    SD = SharkStableDiffusion(device=args.device)
    images = SD.text_to_image(args.prompt, num_steps=args.steps)
    pil_images = [Image.fromarray(image) for image in images]
    save_fname = args.prompt + ".jpg"
    if args.save_path is not None:
        save_fname = args.save_path
    pil_images[0].save(save_fname)
