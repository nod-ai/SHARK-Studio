import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomConfig
import re
from shark.shark_inference import SharkInference
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers.models.bloom.modeling_bloom import (
    BloomBlock,
    build_alibi_tensor,
)
import time
import json


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


if __name__ == "__main__":
    working_dir = sys.argv[1]
    layer_name = sys.argv[2]
    will_compile = sys.argv[3]
    device = sys.argv[4]
    device_idx = sys.argv[5]
    prompt = sys.argv[6]

    if device_idx.lower().strip() == "none":
        device_idx = None
    else:
        device_idx = int(device_idx)

    if will_compile.lower().strip() == "true":
        will_compile = True
    else:
        will_compile = False

    f = open(f"{working_dir}/config.json")
    config = json.load(f)
    f.close()

    layers_initialized = False
    try:
        n_embed = config["n_embed"]
    except KeyError:
        n_embed = config["hidden_size"]
    vocab_size = config["vocab_size"]
    n_layer = config["n_layer"]
    try:
        n_head = config["num_attention_heads"]
    except KeyError:
        n_head = config["n_head"]

    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    if layer_name == "start":
        tokenizer = AutoTokenizer.from_pretrained(working_dir)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        mlir_str = ""

        if will_compile:
            f = open(f"{working_dir}/word_embeddings.mlir", encoding="utf-8")
            mlir_str = f.read()
            f.close()

            mlir_str = bytes(mlir_str, "utf-8")

        shark_module = SharkInference(
            mlir_str,
            device="cpu",
            mlir_dialect="tm_tensor",
            device_idx=None,
        )

        if will_compile:
            shark_module.save_module(
                module_name=f"{working_dir}/word_embeddings",
                extra_args=[
                    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    "--iree-stream-resource-max-allocation-size=1000000000",
                    "--iree-codegen-check-ir-before-llvm-conversion=false",
                ],
            )

        shark_module.load_module(f"{working_dir}/word_embeddings.vmfb")
        input_embeds = shark_module(
            inputs=(input_ids,), function_name="forward"
        )
        input_embeds = torch.tensor(input_embeds).float()

        mlir_str = ""

        if will_compile:
            f = open(
                f"{working_dir}/word_embeddings_layernorm.mlir",
                encoding="utf-8",
            )
            mlir_str = f.read()
            f.close()

        shark_module = SharkInference(
            mlir_str,
            device="cpu",
            mlir_dialect="tm_tensor",
            device_idx=None,
        )

        if will_compile:
            shark_module.save_module(
                module_name=f"{working_dir}/word_embeddings_layernorm",
                extra_args=[
                    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    "--iree-stream-resource-max-allocation-size=1000000000",
                    "--iree-codegen-check-ir-before-llvm-conversion=false",
                ],
            )

        shark_module.load_module(
            f"{working_dir}/word_embeddings_layernorm.vmfb"
        )
        hidden_states = shark_module(
            inputs=(input_embeds,), function_name="forward"
        )
        hidden_states = torch.tensor(hidden_states).float()

        torch.save(hidden_states, f"{working_dir}/hidden_states_0.pt")

        attention_mask = torch.ones(
            [hidden_states.shape[0], len(input_ids[0])]
        )

        attention_mask = torch.tensor(attention_mask).float()

        alibi = build_alibi_tensor(
            attention_mask,
            n_head,
            hidden_states.dtype,
            device="cpu",
        )

        torch.save(alibi, f"{working_dir}/alibi.pt")

        causal_mask = _prepare_attn_mask(
            attention_mask, input_ids.size(), input_embeds, 0
        )
        causal_mask = torch.tensor(causal_mask).float()

        torch.save(causal_mask, f"{working_dir}/causal_mask.pt")

    elif layer_name in [str(x) for x in range(n_layer)]:
        hidden_states = torch.load(
            f"{working_dir}/hidden_states_{layer_name}.pt"
        )
        alibi = torch.load(f"{working_dir}/alibi.pt")
        causal_mask = torch.load(f"{working_dir}/causal_mask.pt")

        mlir_str = ""

        if will_compile:
            f = open(
                f"{working_dir}/bloom_block_{layer_name}.mlir",
                encoding="utf-8",
            )
            mlir_str = f.read()
            f.close()

            mlir_str = bytes(mlir_str, "utf-8")

        shark_module = SharkInference(
            mlir_str,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
        )

        if will_compile:
            shark_module.save_module(
                module_name=f"{working_dir}/bloom_block_{layer_name}",
                extra_args=[
                    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    "--iree-stream-resource-max-allocation-size=1000000000",
                    "--iree-codegen-check-ir-before-llvm-conversion=false",
                ],
            )

        shark_module.load_module(
            f"{working_dir}/bloom_block_{layer_name}.vmfb"
        )

        output = shark_module(
            inputs=(
                hidden_states.detach().numpy(),
                alibi.detach().numpy(),
                causal_mask.detach().numpy(),
            ),
            function_name="forward",
        )

        hidden_states = torch.tensor(output[0]).float()

        torch.save(
            hidden_states,
            f"{working_dir}/hidden_states_{int(layer_name) + 1}.pt",
        )

    elif layer_name == "end":
        mlir_str = ""

        if will_compile:
            f = open(f"{working_dir}/ln_f.mlir", encoding="utf-8")
            mlir_str = f.read()
            f.close()

            mlir_str = bytes(mlir_str, "utf-8")

        shark_module = SharkInference(
            mlir_str,
            device="cpu",
            mlir_dialect="tm_tensor",
            device_idx=None,
        )

        if will_compile:
            shark_module.save_module(
                module_name=f"{working_dir}/ln_f",
                extra_args=[
                    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    "--iree-stream-resource-max-allocation-size=1000000000",
                    "--iree-codegen-check-ir-before-llvm-conversion=false",
                ],
            )

        shark_module.load_module(f"{working_dir}/ln_f.vmfb")

        hidden_states = torch.load(f"{working_dir}/hidden_states_{n_layer}.pt")

        hidden_states = shark_module(
            inputs=(hidden_states,), function_name="forward"
        )

        mlir_str = ""

        if will_compile:
            f = open(f"{working_dir}/lm_head.mlir", encoding="utf-8")
            mlir_str = f.read()
            f.close()

            mlir_str = bytes(mlir_str, "utf-8")

        if config["n_embed"] == 14336:

            def get_state_dict():
                d = torch.load(
                    f"{working_dir}/pytorch_model_00001-of-00072.bin"
                )
                return OrderedDict(
                    (k.replace("word_embeddings.", ""), v)
                    for k, v in d.items()
                )

            def load_causal_lm_head():
                linear = nn.utils.skip_init(
                    nn.Linear, 14336, 250880, bias=False, dtype=torch.float
                )
                linear.load_state_dict(get_state_dict(), strict=False)
                return linear.float()

            lm_head = load_causal_lm_head()

            logits = lm_head(torch.tensor(hidden_states).float())

        else:
            shark_module = SharkInference(
                mlir_str,
                device="cpu",
                mlir_dialect="tm_tensor",
                device_idx=None,
            )

            if will_compile:
                shark_module.save_module(
                    module_name=f"{working_dir}/lm_head",
                    extra_args=[
                        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                        "--iree-stream-resource-max-allocation-size=1000000000",
                        "--iree-codegen-check-ir-before-llvm-conversion=false",
                    ],
                )

            shark_module.load_module(f"{working_dir}/lm_head.vmfb")

            logits = shark_module(
                inputs=(hidden_states,), function_name="forward"
            )

        logits = torch.tensor(logits).float()

        tokenizer = AutoTokenizer.from_pretrained(working_dir)

        next_token = tokenizer.decode(torch.argmax(logits[:, -1, :], dim=-1))

        f = open(f"{working_dir}/prompt.txt", "w+")
        f.write(prompt + next_token)
        f.close()