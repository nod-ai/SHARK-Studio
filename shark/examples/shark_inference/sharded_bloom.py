####################################################################################
# Please make sure you have transformers 4.21.2 installed before running this demo
#
# -p --model_path: the directory in which you want to store the bloom files.
# -dl --device_list: the list of device indices you want to use.  if you want to only use the first device, or you are running on cpu leave this blank.
#                     Otherwise, please give this argument in this format: "[0, 1, 2]"
# -de --device: the device you want to run bloom on.  E.G. cpu, cuda
# -c, --recompile: set to true if you want to recompile to vmfb.
# -d, --download: set to true if you want to redownload the mlir files
# -t --token_count: the number of tokens you want to generate
# -pr --prompt: the prompt you want to feed to the model
#####################################################################################

import os
import torch
import torch.nn as nn
from collections import OrderedDict
import torch_mlir
from torch_mlir import TensorPlaceholder
import re
from transformers.models.bloom.configuration_bloom import BloomConfig
import json
import sys
import argparse
from cuda.cudart import cudaSetDevice
import json

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_public_file

from transformers.models.bloom.modeling_bloom import (
    BloomBlock,
    build_alibi_tensor,
)

IS_CUDA = False


class ShardedBloom:
    def __init__(self, src_folder):
        f = open(f"{src_folder}/config.json")
        config = json.load(f)
        f.close()

        self.layers_initialized = False

        self.src_folder = src_folder
        self.n_embed = config["n_embed"]
        self.vocab_size = config["vocab_size"]
        self.n_layer = config["n_layer"]
        self.n_head = config["num_attention_heads"]

    def _init_layer(self, layer_name, device, replace, device_idx):
        if replace or not os.path.exists(
            f"{self.src_folder}/{layer_name}.vmfb"
        ):
            f_ = open(f"{self.src_folder}/{layer_name}.mlir")
            module = f_.read()
            f_.close()
            module = bytes(module, "utf-8")
            shark_module = SharkInference(
                module,
                device=device,
                mlir_dialect="tm_tensor",
                device_idx=device_idx,
            )
            shark_module.save_module(
                module_name=f"{self.src_folder}/{layer_name}",
                extra_args=[
                    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    "--iree-stream-resource-max-allocation-size=1000000000",
                    "--iree-codegen-check-ir-before-llvm-conversion=false",
                ],
            )
        else:
            shark_module = SharkInference(
                "",
                device=device,
                mlir_dialect="tm_tensor",
                device_idx=device_idx,
            )

        return shark_module

    def init_layers(self, device, replace=False, device_idx=[0]):
        if device_idx is not None:
            n_devices = len(device_idx)

        self.word_embeddings_module = self._init_layer(
            "word_embeddings",
            device,
            replace,
            device_idx if device_idx is None else device_idx[0 % n_devices],
        )
        self.word_embeddings_layernorm_module = self._init_layer(
            "word_embeddings_layernorm",
            device,
            replace,
            device_idx if device_idx is None else device_idx[1 % n_devices],
        )
        self.ln_f_module = self._init_layer(
            "ln_f",
            device,
            replace,
            device_idx if device_idx is None else device_idx[2 % n_devices],
        )
        self.lm_head_module = self._init_layer(
            "lm_head",
            device,
            replace,
            device_idx if device_idx is None else device_idx[3 % n_devices],
        )
        self.block_modules = [
            self._init_layer(
                f"bloom_block_{i}",
                device,
                replace,
                device_idx
                if device_idx is None
                else device_idx[(i + 4) % n_devices],
            )
            for i in range(self.n_layer)
        ]

        self.layers_initialized = True

    def load_layers(self):
        assert self.layers_initialized

        self.word_embeddings_module.load_module(
            f"{self.src_folder}/word_embeddings.vmfb"
        )
        self.word_embeddings_layernorm_module.load_module(
            f"{self.src_folder}/word_embeddings_layernorm.vmfb"
        )
        for block_module, i in zip(self.block_modules, range(self.n_layer)):
            block_module.load_module(f"{self.src_folder}/bloom_block_{i}.vmfb")
        self.ln_f_module.load_module(f"{self.src_folder}/ln_f.vmfb")
        self.lm_head_module.load_module(f"{self.src_folder}/lm_head.vmfb")

    def forward_pass(self, input_ids, device):
        if IS_CUDA:
            cudaSetDevice(self.word_embeddings_module.device_idx)

        input_embeds = self.word_embeddings_module(
            inputs=(input_ids,), function_name="forward"
        )

        input_embeds = torch.tensor(input_embeds).float()
        if IS_CUDA:
            cudaSetDevice(self.word_embeddings_layernorm_module.device_idx)
        hidden_states = self.word_embeddings_layernorm_module(
            inputs=(input_embeds,), function_name="forward"
        )

        hidden_states = torch.tensor(hidden_states).float()

        attention_mask = torch.ones(
            [hidden_states.shape[0], len(input_ids[0])]
        )
        alibi = build_alibi_tensor(
            attention_mask,
            self.n_head,
            hidden_states.dtype,
            hidden_states.device,
        )

        causal_mask = _prepare_attn_mask(
            attention_mask, input_ids.size(), input_embeds, 0
        )
        causal_mask = torch.tensor(causal_mask).float()

        presents = ()
        all_hidden_states = tuple(hidden_states)

        for block_module, i in zip(self.block_modules, range(self.n_layer)):
            if IS_CUDA:
                cudaSetDevice(block_module.device_idx)

            output = block_module(
                inputs=(
                    hidden_states.detach().numpy(),
                    alibi.detach().numpy(),
                    causal_mask.detach().numpy(),
                ),
                function_name="forward",
            )
            hidden_states = torch.tensor(output[0]).float()
            all_hidden_states = all_hidden_states + (hidden_states,)
            presents = presents + (
                tuple(
                    (
                        output[1],
                        output[2],
                    )
                ),
            )
        if IS_CUDA:
            cudaSetDevice(self.ln_f_module.device_idx)

        hidden_states = self.ln_f_module(
            inputs=(hidden_states,), function_name="forward"
        )
        if IS_CUDA:
            cudaSetDevice(self.lm_head_module.device_idx)

        logits = self.lm_head_module(
            inputs=(hidden_states,), function_name="forward"
        )
        logits = torch.tensor(logits).float()

        return torch.argmax(logits[:, -1, :], dim=-1)


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.full((target_length, target_length), torch.finfo(dtype).min)
    mask_cond = torch.arange(mask.size(-1))
    intermediate_mask = mask_cond < (mask_cond + 1).view(mask.size(-1), 1)
    mask.masked_fill_(intermediate_mask, 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    target_length, past_key_values_length, dtype=dtype
                ),
                mask,
            ],
            dim=-1,
        )
    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    batch_size, source_length = mask.size()
    tgt_len = tgt_len if tgt_len is not None else source_length

    expanded_mask = (
        mask[:, None, None, :]
        .expand(batch_size, 1, tgt_len, source_length)
        .to(dtype)
    )

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _prepare_attn_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
        ).to(attention_mask.device)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def download_560m(destination_folder):
    download_public_file(
        "https://bloom-560m/bloom_block_0.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_1.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_2.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_3.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_4.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_5.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_6.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_7.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_8.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_9.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_10.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_11.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_12.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_13.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_14.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_15.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_16.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_17.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_18.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_19.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_20.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_21.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_22.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/bloom_block_23.mlir", destination_folder
    )
    download_public_file("https://bloom-560m/config.json", destination_folder)
    download_public_file("https://bloom-560m/lm_head.mlir", destination_folder)
    download_public_file("https://bloom-560m/ln_f.mlir", destination_folder)
    download_public_file(
        "https://bloom-560m/word_embeddings.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/word_embeddings_layernorm.mlir", destination_folder
    )
    download_public_file(
        "https://bloom-560m/tokenizer.json", destination_folder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Bloom-560m")
    parser.add_argument("-p", "--model_path")
    parser.add_argument("-dl", "--device_list", default=None)
    parser.add_argument("-de", "--device", default="cpu")
    parser.add_argument("-c", "--recompile", default=False, type=bool)
    parser.add_argument("-d", "--download", default=False, type=bool)
    parser.add_argument("-t", "--token_count", default=10, type=int)
    parser.add_argument(
        "-pr",
        "--prompt",
        default="The SQL command to extract all the users whose name starts with A is: ",
    )
    args = parser.parse_args()

    if args.device_list is not None:
        args.device_list = json.loads(args.device_list)

    if args.device == "cuda" and args.device_list is not None:
        IS_CUDA = True
    if args.download:
        download_560m(args.model_path)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BloomConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")

    shardedbloom = ShardedBloom(args.model_path)
    shardedbloom.init_layers(
        device=args.device, replace=args.recompile, device_idx=args.device_list
    )
    shardedbloom.load_layers()

    for _ in range(args.token_count):
        next_token = shardedbloom.forward_pass(
            torch.tensor(input_ids), device=args.device
        )
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

    print(tokenizer.decode(input_ids.squeeze()))
