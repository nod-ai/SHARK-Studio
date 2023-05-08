import torch
import argparse
import torch_mlir
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from io import BytesIO
from pathlib import Path
from shark.shark_downloader import download_public_file
from shark.shark_importer import transform_fx as transform_fx_
import re
from shark.shark_inference import SharkInference
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

parser.add_argument("--precision", "-p", default="fp32", help="fp32, fp16")
parser.add_argument(
    "--device", "-d", default="vulkan", help="vulkan, cpu, cuda"
)


class VicunaLayer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, attention_mask, position_ids):
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        next_hidden_states = outputs[0]
        return next_hidden_states


class CompiledVicunaLayer(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        hidden_states = hidden_states.detach()
        attention_mask = attention_mask.detach()
        position_ids = position_ids.detach()
        output = self.model(
            "forward",
            (
                hidden_states,
                attention_mask,
                position_ids,
            ),
        )

        print(output)

        output = torch.tensor(output)

        return (output,)


class ShardedVicunaModel(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        assert len(layers) == len(model.model.layers)
        self.model.model.layers = torch.nn.modules.container.ModuleList(layers)
        self.model.model.config.use_cache = False
        self.model.model.config.output_attentions = False

    def forward(self, input_ids, attention_mask=None):
        return self.model.forward(input_ids, attention_mask=attention_mask)


def compile_vicuna_layer(
    vicuna_layer, hidden_states, attention_mask, position_ids
):
    fx_g = make_fx(
        vicuna_layer,
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
    )(hidden_states, attention_mask, position_ids)

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

    transform_fx(fx_g)
    if args.precision == "fp16":
        fx_g = fx_g.half()
    fx_g.recompile()
    removed_none_indexes = _remove_nones(fx_g)
    was_unwrapped = _unwrap_single_tuple_return(fx_g)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    print("FX_G recompile")

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
    return ts_g


path = "TheBloke/vicuna-7B-1.1-HF"
kwargs = {"torch_dtype": torch.float32}
vicuna_model = AutoModelForCausalLM.from_pretrained(
    path, low_cpu_mem_usage=True, **kwargs
)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

print(type(vicuna_model.model.layers))


def compile_to_vmfb(inputs, layers):
    mlirs, modules = [], []
    for idx, layer in tqdm(enumerate(layers), desc="Getting mlirs"):
        mlir_path = Path(f"{idx}.mlir")
        if mlir_path.exists():
            # print(f"Found layer {idx} mlir")
            f_ = open(mlir_path, "rb")
            bytecode = f_.read()
            f_.close()
        else:
            print(f"Compiling layer {idx} mlir")
            ts_g = compile_vicuna_layer(layer, inputs[0], inputs[1], inputs[2])
            module = torch_mlir.compile(
                ts_g,
                inputs,
                torch_mlir.OutputType.LINALG_ON_TENSORS,
                use_tracing=False,
                verbose=False,
            )
            bytecode_stream = BytesIO()
            module.operation.write_bytecode(bytecode_stream)
            bytecode = bytecode_stream.getvalue()
            f_ = open(mlir_path, "wb")
            f_.write(bytecode)
            f_.close()
        mlirs.append(bytecode)

    for idx, layer in tqdm(enumerate(layers), desc="compiling modules"):
        device = args.device if idx < 25 else "cpu"
        vmfb_path = Path(f"{idx}.vmfb")
        if vmfb_path.exists():
            # print(f"Found layer {idx} vmfb")
            module = SharkInference(
                None, device=device, mlir_dialect="tm_tensor"
            )
            module.load_module(vmfb_path)
        else:
            print(f"Compiling layer {idx} vmfb")
            module = SharkInference(
                mlirs[idx], device=device, mlir_dialect="tm_tensor"
            )
            module.save_module("", f"{idx}")
            module.load_module(vmfb_path)
        modules.append(module)
    return mlirs, modules


if __name__ == "__main__":
    args = parser.parse_args()
    # prompt = input("Enter Prompt: ")
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    placeholder_input = (
        torch.zeros([1, 256, 4096], dtype=dtype),
        torch.zeros([1, 1, 256, 256], dtype=dtype),
        torch.zeros([1, 256], dtype=torch.int64),
    )

    _, modules = compile_to_vmfb(placeholder_input, vicuna_model.model.layers)

    shark_layers = [CompiledVicunaLayer(m) for m in modules]

    sharded_model = ShardedVicunaModel(vicuna_model, shark_layers)
    prompt = "It was a dark and stormy"
    prompt = prompt.strip()
    input_ids = tokenizer(prompt).input_ids
    original_input_ids = input_ids
    input_id_len = len(input_ids)
    pad_len = 256 - input_id_len
    attention_mask = torch.ones([1, input_id_len], dtype=torch.int64)
    input_ids = torch.nn.functional.pad(
        torch.tensor(input_ids), (0, pad_len), mode="constant", value=259
    )
    input_ids = input_ids.reshape([1, 256])
    attention_mask = torch.nn.functional.pad(
        torch.tensor(attention_mask),
        (0, pad_len),
        mode="constant",
        value=0,
    )

    # print(input_ids)
    if args.precision == "fp16":
        input_ids = input_ids.to(torch.float16)
    print(attention_mask)

    logits = sharded_model.forward(input_ids, attention_mask=attention_mask)[
        "logits"
    ]
    print(logits)
