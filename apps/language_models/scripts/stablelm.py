import torch
import torch_mlir
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from io import BytesIO
from pathlib import Path
from shark.shark_downloader import download_public_file

from shark.shark_inference import SharkInference
from pathlib import Path


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def shouldStop(tokens):
    stop_ids = [50278, 50279, 50277, 1, 0]
    for stop_id in stop_ids:
        if tokens[0][-1] == stop_id:
            return True
    return False


MAX_SEQUENCE_LENGTH = 256


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def get_torch_mlir_module_bytecode(model, model_inputs):
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
        # tracing_mode='symbolic',
    )(*model_inputs)
    print("Got FX_G")

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
    print("Got TS_G")
    return ts_g


def compile_stableLM(model, model_inputs, model_name, model_vmfb_name):
    # ADD Device Arg
    from shark.shark_inference import SharkInference

    device = "cuda"  # 'cpu'
    vmfb_path = (
        Path(model_name + f"_{device}.vmfb")
        if model_vmfb_name is None
        else Path(model_vmfb_name)
    )
    if vmfb_path.exists():
        print("Loading vmfb from: ", vmfb_path)
        shark_module = SharkInference(
            None, device=device, mlir_dialect="tm_tensor"
        )
        shark_module.load_module(vmfb_path)
        print("Successfully loaded vmfb")
        return shark_module

    mlir_path = Path(model_name + ".mlir")
    print(
        f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
    )
    if mlir_path.exists():
        with open(mlir_path, "rb") as f:
            bytecode = f.read()
    else:
        ts_graph = get_torch_mlir_module_bytecode(model, model_inputs)
        module = torch_mlir.compile(
            ts_graph,
            [*model_inputs],
            torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=False,
            verbose=False,
        )
        bytecode_stream = BytesIO()
        module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()
    f_ = open(model_name + ".mlir", "wb")
    f_.write(bytecode)
    print("Saved mlir")
    f_.close()

    shark_module = SharkInference(
        mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    import os

    path = shark_module.save_module(os.getcwd(), model_vmfb_name, [])
    print("Saved vmfb at ", str(path))

    return shark_module


class StableLMModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        combine_input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        output = self.model(**combine_input_dict)
        return output.logits


# Initialize a StopOnTokens object
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


def get_tokenizer():
    model_path = "stabilityai/stablelm-tuned-alpha-3b"
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.add_special_tokens({"pad_token": "<PAD>"})
    print(f"Sucessfully loaded the tokenizer to the memory")
    return tok


# sharkStableLM = compile_stableLM(None, tuple([input_ids, attention_mask]), "stableLM_linalg_f32_seqLen256", "/home/shark/vivek/stableLM_shark_f32_seqLen256")
def generate(
    new_text,
    # streamer,
    max_new_tokens,
    do_sample,
    top_p,
    top_k,
    temperature,
    num_beams,
    stopping_criteria,
    sharkStableLM,
    tok=None,
    input_ids=torch.randint(3, (1, 256)),
    attention_mask=torch.randint(3, (1, 256)),
):
    if tok == None:
        tok = get_tokenizer()
    # Construct the input message string for the model by concatenating the current system message and conversation history
    # Tokenize the messages string
    # sharkStableLM = compile_stableLM(None, tuple([input_ids, attention_mask]), "stableLM_linalg_f32_seqLen256", "/home/shark/vivek/stableLM_shark_f32_seqLen256")
    words_list = []
    for i in range(max_new_tokens):
        numWords = len(new_text.split())
        # if(numWords>220):
        #  break
        model_inputs = tok(
            [new_text],
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        sum_attentionmask = torch.sum(model_inputs.attention_mask)
        # sharkStableLM = compile_stableLM(None, tuple([input_ids, attention_mask]), "stableLM_linalg_f32_seqLen256", "/home/shark/vivek/stableLM_shark_f32_seqLen256")
        output = sharkStableLM(
            "forward", [model_inputs.input_ids, model_inputs.attention_mask]
        )
        output = torch.from_numpy(output)
        next_toks = torch.topk(output, 1)
        if shouldStop(next_toks.indices):
            break
        #        streamer.put(next_toks.indices[0][int(sum_attentionmask)-1])
        new_word = tok.decode(
            next_toks.indices[0][int(sum_attentionmask) - 1],
            skip_special_tokens=True,
        )
        print(new_word, end="", flush=True)
        words_list.append(new_word)
        if new_word == "":
            break
        new_text = new_text + new_word
    return words_list
