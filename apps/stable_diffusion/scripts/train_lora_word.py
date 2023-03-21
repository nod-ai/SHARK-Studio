# Install the required libs
# pip install -U git+https://github.com/huggingface/diffusers.git
# pip install accelerate transformers ftfy

# HuggingFace Token
# YOUR_TOKEN = "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"


# Import required libraries
import itertools
import math
import os
from typing import List
import random
import torch_mlir

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
import logging

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor

import torch_mlir
from torch_mlir.dynamo import make_simple_dynamo_backend
import torch._dynamo as dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from shark.shark_inference import SharkInference

torch._dynamo.config.verbose = True

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
)

from io import BytesIO

from dataclasses import dataclass
from apps.stable_diffusion.src import (
    args,
    get_schedulers,
    set_init_device_flags,
    clear_all,
)


# Setup the dataset
class LoraDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        repeats=100,
        interpolation="bicubic",
        set="train",
        prompt="myloraprompt",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.prompt = prompt

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["input_ids"] = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2,
                (w - crop) // 2 : (w + crop) // 2,
            ]

        image = Image.fromarray(img)
        image = image.resize(
            (self.size, self.size), resample=self.interpolation
        )

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


########## Setting up the model ##########
def lora_train(
    prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    custom_model: str,
    hf_model_id: str,
    precision: str,
    device: str,
    max_length: int,
    training_images_dir: str,
    lora_save_dir: str,
):
    from apps.stable_diffusion.web.ui.utils import (
        get_custom_model_pathfile,
        Config,
    )
    import apps.stable_diffusion.web.utils.global_obj as global_obj

    print(
        "Note LoRA training is not compatible with the latest torch-mlir branch"
    )
    print(
        "To run LoRA training you'll need this to follow this guide for the torch-mlir branch: https://github.com/nod-ai/SHARK/tree/main/shark/examples/shark_training/stable_diffusion"
    )
    torch.manual_seed(seed)

    args.prompts = [prompt]
    args.steps = steps

    # set ckpt_loc and hf_model_id.
    types = (
        ".ckpt",
        ".safetensors",
    )  # the tuple of file types
    args.ckpt_loc = ""
    args.hf_model_id = ""
    if custom_model == "None":
        if not hf_model_id:
            return (
                None,
                "Please provide either custom model or huggingface model ID, both must not be empty",
            )
        args.hf_model_id = hf_model_id
    elif ".ckpt" in custom_model or ".safetensors" in custom_model:
        args.ckpt_loc = custom_model
    else:
        args.hf_model_id = custom_model

    args.training_images_dir = training_images_dir
    args.lora_save_dir = lora_save_dir

    args.precision = precision
    args.batch_size = batch_size
    args.max_length = max_length
    args.height = height
    args.width = width
    args.device = device

    # Load the Stable Diffusion model
    text_encoder = CLIPTextModel.from_pretrained(
        args.hf_model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(args.hf_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.hf_model_id, subfolder="unet"
    )

    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    # Freeze everything but LoRA
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    freeze_params(text_encoder.parameters())

    # Move vae and unet to device
    vae.to(args.device)
    unet.to(args.device)
    text_encoder.to(args.device)

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id
            ]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae

        def forward(self, input):
            x = self.vae.encode(input, return_dict=False)[0]
            return x

    class UnetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = unet

        def forward(self, x, y, z):
            return self.unet.forward(x, y, z, return_dict=False)[0]

    shark_vae = VaeModel()
    shark_unet = UnetModel()

    ####### Creating our training data ########

    tokenizer = CLIPTokenizer.from_pretrained(
        args.hf_model_id,
        subfolder="tokenizer",
    )

    # Let's create the Dataset and Dataloader
    train_dataset = LoraDataset(
        data_root=args.training_images_dir,
        tokenizer=tokenizer,
        size=vae.sample_size,
        prompt=args.prompts[0],
        repeats=100,
        center_crop=False,
        set="train",
    )

    def create_dataloader(train_batch_size=1):
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

    # Create noise_scheduler for training
    noise_scheduler = DDPMScheduler.from_config(
        args.hf_model_id, subfolder="scheduler"
    )

    ######## Training ###########

    # Define hyperparameters for our training. If you are not happy with your results,
    # you can tune the `learning_rate` and the `max_train_steps`

    # Setting up all training args
    hyperparameters = {
        "learning_rate": 5e-04,
        "scale_lr": True,
        "max_train_steps": steps,
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "seed": 42,
        "output_dir": "sd-concept-output",
    }
    # creating output directory
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, hyperparameters["output_dir"])
    while not os.path.exists(str(out_dir)):
        try:
            os.mkdir(out_dir)
        except OSError as error:
            print("Output directory not created")

    ###### Torch-MLIR Compilation ######

    def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
        removed_indexes = []
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, (list, tuple)):
                    node_arg = list(node_arg)
                    node_args_len = len(node_arg)
                    for i in range(node_args_len):
                        curr_index = node_args_len - (i + 1)
                        if node_arg[curr_index] is None:
                            removed_indexes.append(curr_index)
                            node_arg.pop(curr_index)
                    node.args = (tuple(node_arg),)
                    break

        if len(removed_indexes) > 0:
            fx_g.graph.lint()
            fx_g.graph.eliminate_dead_code()
            fx_g.recompile()
        removed_indexes.sort()
        return removed_indexes

    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
        """
        Replace tuple with tuple element in functions that return one-element tuples.
        Returns true if an unwrapping took place, and false otherwise.
        """
        unwrapped_tuple = False
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    if len(node_arg) == 1:
                        node.args = (node_arg[0],)
                        unwrapped_tuple = True
                        break

        if unwrapped_tuple:
            fx_g.graph.lint()
            fx_g.recompile()
        return unwrapped_tuple

    def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    return len(node_arg) == 0
        return False

    def transform_fx(fx_g):
        for node in fx_g.graph.nodes:
            if node.op == "call_function":
                if node.target in [
                    torch.ops.aten.empty,
                ]:
                    # aten.empty should be filled with zeros.
                    if node.target in [torch.ops.aten.empty]:
                        with fx_g.graph.inserting_after(node):
                            new_node = fx_g.graph.call_function(
                                torch.ops.aten.zero_,
                                args=(node,),
                            )
                            node.append(new_node)
                            node.replace_all_uses_with(new_node)
                            new_node.args = (node,)

        fx_g.graph.lint()

    @make_simple_dynamo_backend
    def refbackend_torchdynamo_backend(
        fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ):
        # handling usage of empty tensor without initializing
        transform_fx(fx_graph)
        fx_graph.recompile()
        if _returns_nothing(fx_graph):
            return fx_graph
        removed_none_indexes = _remove_nones(fx_graph)
        was_unwrapped = _unwrap_single_tuple_return(fx_graph)

        mlir_module = torch_mlir.compile(
            fx_graph, example_inputs, output_type="linalg-on-tensors"
        )

        bytecode_stream = BytesIO()
        mlir_module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()

        shark_module = SharkInference(
            mlir_module=bytecode, device=args.device, mlir_dialect="tm_tensor"
        )
        shark_module.compile()

        def compiled_callable(*inputs):
            inputs = [x.numpy() for x in inputs]
            result = shark_module("forward", inputs)
            if was_unwrapped:
                result = [
                    result,
                ]
            if not isinstance(result, list):
                result = torch.from_numpy(result)
            else:
                result = tuple(torch.from_numpy(x) for x in result)
                result = list(result)
                for removed_index in removed_none_indexes:
                    result.insert(removed_index, None)
                result = tuple(result)
            return result

        return compiled_callable

    def predictions(torch_func, jit_func, batchA, batchB):
        res = jit_func(batchA.numpy(), batchB.numpy())
        if res is not None:
            # prediction = torch.from_numpy(res)
            prediction = res
        else:
            prediction = None
        return prediction

    logger = logging.getLogger(__name__)

    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters[
        "gradient_accumulation_steps"
    ]
    learning_rate = hyperparameters["learning_rate"]
    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            # * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    # Training function
    def train_func(batch_pixel_values, batch_input_ids):
        # Convert images to latent space
        latents = shark_vae(batch_pixel_values).sample().detach()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch_input_ids)[0]

        # Predict the noise residual
        noise_pred = shark_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
        )

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        loss = (
            F.mse_loss(noise_pred, target, reduction="none")
            .mean([1, 2, 3])
            .mean()
        )
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss

    def training_function():
        max_train_steps = hyperparameters["max_train_steps"]
        output_dir = hyperparameters["output_dir"]
        gradient_checkpointing = hyperparameters["gradient_checkpointing"]

        train_dataloader = create_dataloader(train_batch_size)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(
            max_train_steps / num_update_steps_per_epoch
        )

        # Train!
        total_batch_size = (
            train_batch_size
            * gradient_accumulation_steps
            # train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(
            f"  Instantaneous batch size per device = {train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            # range(max_train_steps), disable=not accelerator.is_local_main_process
            range(max_train_steps)
        )
        progress_bar.set_description("Steps")
        global_step = 0

        params__ = [
            i for i in text_encoder.get_input_embeddings().parameters()
        ]

        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                dynamo_callable = dynamo.optimize(
                    refbackend_torchdynamo_backend
                )(train_func)
                lam_func = lambda x, y: dynamo_callable(
                    torch.from_numpy(x), torch.from_numpy(y)
                )
                loss = predictions(
                    train_func,
                    lam_func,
                    batch["pixel_values"],
                    batch["input_ids"],
                )

                # Checks if the accelerator has performed an optimization step behind the scenes
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

    training_function()

    # Save the lora weights
    unet.save_attn_procs(args.lora_save_dir)

    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad  # free some memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if args.clear_all:
        clear_all()

    dtype = torch.float32 if args.precision == "fp32" else torch.half
    cpu_scheduling = not args.scheduler.startswith("Shark")
    set_init_device_flags()
    schedulers = get_schedulers(args.hf_model_id)
    scheduler_obj = schedulers[args.scheduler]
    seed = args.seed
    if len(args.prompts) != 1:
        print("Need exactly one prompt for the LoRA word")
    lora_train(
        args.prompts[0],
        args.height,
        args.width,
        args.training_steps,
        args.guidance_scale,
        args.seed,
        args.batch_count,
        args.batch_size,
        args.scheduler,
        "None",
        args.hf_model_id,
        args.precision,
        args.device,
        args.max_length,
        args.training_images_dir,
        args.lora_save_dir,
    )
