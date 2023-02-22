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
# -m --model_namme: the name of the model, e.g. bloom-560m
#####################################################################################

import os
import io
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
import json
import urllib.request

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_public_file
from transformers import (
    BloomTokenizerFast,
    BloomForSequenceClassification,
    BloomForCausalLM,
)
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
        try:
            self.n_embed = config["n_embed"]
        except KeyError:
            self.n_embed = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.n_layer = config["n_layer"]
        try:
            self.n_head = config["num_attention_heads"]
        except KeyError:
            self.n_head = config["n_head"]

    def _init_layer(self, layer_name, device, replace, device_idx):
        if replace or not os.path.exists(
            f"{self.src_folder}/{layer_name}.vmfb"
        ):
            f_ = open(f"{self.src_folder}/{layer_name}.mlir", encoding="utf-8")
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


def download_model(destination_folder, model_name):
    download_public_file(
        f"https://{model_name}/config.json", destination_folder
    )
    f = open(f"{destination_folder}/config.json")
    config = json.load(f)
    f.close()
    n_blocks = config["n_layer"]
    download_public_file(
        f"https://{model_name}/lm_head.mlir", destination_folder
    )
    download_public_file(f"https://{model_name}/ln_f.mlir", destination_folder)
    download_public_file(
        f"https://{model_name}/word_embeddings.mlir", destination_folder
    )
    download_public_file(
        f"https://{model_name}/word_embeddings_layernorm.mlir",
        destination_folder,
    )
    download_public_file(
        f"https://{model_name}/tokenizer.json", destination_folder
    )

    for i in range(n_blocks):
        download_public_file(
            f"https://{model_name}/bloom_block_{i}.mlir", destination_folder
        )


def compile_embeddings(embeddings_layer, input_ids, path):
    input_ids_placeholder = torch_mlir.TensorPlaceholder.like(
        input_ids, dynamic_axes=[1]
    )
    module = torch_mlir.compile(
        embeddings_layer,
        (input_ids_placeholder),
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    bytecode_stream = io.BytesIO()
    module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    f_ = open(path, "w+")
    f_.write(str(module))
    f_.close()
    return


def compile_word_embeddings_layernorm(
    embeddings_layer_layernorm, embeds, path
):
    embeds_placeholder = torch_mlir.TensorPlaceholder.like(
        embeds, dynamic_axes=[1]
    )
    module = torch_mlir.compile(
        embeddings_layer_layernorm,
        (embeds_placeholder),
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    bytecode_stream = io.BytesIO()
    module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    f_ = open(path, "w+")
    f_.write(str(module))
    f_.close()
    return


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


def compile_to_mlir(
    bblock,
    hidden_states,
    layer_past=None,
    attention_mask=None,
    head_mask=None,
    use_cache=None,
    output_attentions=False,
    alibi=None,
    block_index=0,
    path=".",
):
    fx_g = make_fx(
        bblock,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
            ]
        ),
        tracing_mode="real",
        _allow_non_fake_inputs=False,
    )(hidden_states, alibi, attention_mask)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    strip_overloads(fx_g)

    hidden_states_placeholder = TensorPlaceholder.like(
        hidden_states, dynamic_axes=[1]
    )
    attention_mask_placeholder = TensorPlaceholder.like(
        attention_mask, dynamic_axes=[2, 3]
    )
    alibi_placeholder = TensorPlaceholder.like(alibi, dynamic_axes=[2])

    ts_g = torch.jit.script(fx_g)

    module = torch_mlir.compile(
        ts_g,
        (
            hidden_states_placeholder,
            alibi_placeholder,
            attention_mask_placeholder,
        ),
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    module_placeholder = module
    module_context = module_placeholder.context

    def check_valid_line(line, line_n, mlir_file_len):
        if "private" in line:
            return False
        if "attributes" in line:
            return False
        if mlir_file_len - line_n == 2:
            return False

        return True

    mlir_file_len = len(str(module).split("\n"))

    def remove_constant_dim(line):
        if "17x" in line:
            line = re.sub("17x", "?x", line)
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
        if "tensor.empty" in line and "?x?" in line:
            line = re.sub(
                "tensor.empty\(%dim\)", "tensor.empty(%dim, %dim)", line
            )
        if "arith.cmpi eq" in line:
            line = re.sub("c17", "dim", line)
        if " 17," in line:
            line = re.sub(" 17,", " %dim,", line)
        return line

    module = "\n".join(
        [
            remove_constant_dim(line)
            for line, line_n in zip(
                str(module).split("\n"), range(mlir_file_len)
            )
            if check_valid_line(line, line_n, mlir_file_len)
        ]
    )

    module = module_placeholder.parse(module, context=module_context)
    bytecode_stream = io.BytesIO()
    module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    f_ = open(path, "w+")
    f_.write(str(module))
    f_.close()
    return


def compile_ln_f(ln_f, hidden_layers, path):
    hidden_layers_placeholder = torch_mlir.TensorPlaceholder.like(
        hidden_layers, dynamic_axes=[1]
    )
    module = torch_mlir.compile(
        ln_f,
        (hidden_layers_placeholder),
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    bytecode_stream = io.BytesIO()
    module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    f_ = open(path, "w+")
    f_.write(str(module))
    f_.close()
    return


def compile_lm_head(lm_head, hidden_layers, path):
    hidden_layers_placeholder = torch_mlir.TensorPlaceholder.like(
        hidden_layers, dynamic_axes=[1]
    )
    module = torch_mlir.compile(
        lm_head,
        (hidden_layers_placeholder),
        torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=False,
        verbose=False,
    )

    bytecode_stream = io.BytesIO()
    module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    f_ = open(path, "w+")
    f_.write(str(module))
    f_.close()
    return


def create_mlirs(destination_folder, model_name):
    model_config = "bigscience/" + model_name
    sample_input_ids = torch.ones([1, 17], dtype=torch.int64)

    urllib.request.urlretrieve(
        f"https://huggingface.co/bigscience/{model_name}/resolve/main/config.json",
        filename=f"{destination_folder}/config.json",
    )
    urllib.request.urlretrieve(
        f"https://huggingface.co/bigscience/bloom/resolve/main/tokenizer.json",
        filename=f"{destination_folder}/tokenizer.json",
    )

    class HuggingFaceLanguage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = BloomForCausalLM.from_pretrained(model_config)

        def forward(self, tokens):
            return self.model.forward(tokens)[0]

    class HuggingFaceBlock(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.model = block

        def forward(self, tokens, alibi, attention_mask):
            output = self.model(
                hidden_states=tokens,
                alibi=alibi,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
            )
            return (output[0], output[1][0], output[1][1])

    model = HuggingFaceLanguage()

    compile_embeddings(
        model.model.transformer.word_embeddings,
        sample_input_ids,
        f"{destination_folder}/word_embeddings.mlir",
    )

    inputs_embeds = model.model.transformer.word_embeddings(sample_input_ids)

    compile_word_embeddings_layernorm(
        model.model.transformer.word_embeddings_layernorm,
        inputs_embeds,
        f"{destination_folder}/word_embeddings_layernorm.mlir",
    )

    hidden_states = model.model.transformer.word_embeddings_layernorm(
        inputs_embeds
    )

    input_shape = sample_input_ids.size()

    current_sequence_length = hidden_states.shape[1]
    past_key_values_length = 0
    past_key_values = tuple([None] * len(model.model.transformer.h))

    attention_mask = torch.ones(
        (hidden_states.shape[0], current_sequence_length), device="cpu"
    )

    alibi = build_alibi_tensor(
        attention_mask,
        model.model.transformer.n_head,
        hidden_states.dtype,
        "cpu",
    )

    causal_mask = _prepare_attn_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    head_mask = model.model.transformer.get_head_mask(
        None, model.model.transformer.config.n_layer
    )
    output_attentions = model.model.transformer.config.output_attentions

    all_hidden_states = ()

    for i, (block, layer_past) in enumerate(
        zip(model.model.transformer.h, past_key_values)
    ):
        all_hidden_states = all_hidden_states + (hidden_states,)

        proxy_model = HuggingFaceBlock(block)

        compile_to_mlir(
            proxy_model,
            hidden_states,
            layer_past=layer_past,
            attention_mask=causal_mask,
            head_mask=head_mask[i],
            use_cache=True,
            output_attentions=output_attentions,
            alibi=alibi,
            block_index=i,
            path=f"{destination_folder}/bloom_block_{i}.mlir",
        )

    compile_ln_f(
        model.model.transformer.ln_f,
        hidden_states,
        f"{destination_folder}/ln_f.mlir",
    )
    hidden_states = model.model.transformer.ln_f(hidden_states)
    compile_lm_head(
        model.model.lm_head,
        hidden_states,
        f"{destination_folder}/lm_head.mlir",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Bloom-560m")
    parser.add_argument("-p", "--model_path")
    parser.add_argument("-dl", "--device_list", default=None)
    parser.add_argument("-de", "--device", default="cpu")
    parser.add_argument("-c", "--recompile", default=False, type=bool)
    parser.add_argument("-d", "--download", default=False, type=bool)
    parser.add_argument("-t", "--token_count", default=10, type=int)
    parser.add_argument("-m", "--model_name", default="bloom-560m")
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
        from cuda.cudart import cudaSetDevice
    if args.download:
        # download_model(args.model_path, args.model_name)
        create_mlirs(args.model_path, args.model_name)
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
