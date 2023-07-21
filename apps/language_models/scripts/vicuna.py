import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
import numpy as np
import iree.runtime

import torch
import torch_mlir
from torch_mlir import TensorPlaceholder
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from transformers import AutoTokenizer, AutoModelForCausalLM

from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.src.model_wrappers.vicuna_sharded_model import (
    FirstVicunaLayer,
    SecondVicunaLayer,
    CompiledVicunaLayer,
    ShardedVicunaModel,
    LMHead,
    LMHeadCompiled,
    VicunaEmbedding,
    VicunaEmbeddingCompiled,
    VicunaNorm,
    VicunaNormCompiled,
)
from apps.language_models.src.model_wrappers.vicuna_model import (
    FirstVicuna,
    SecondVicuna,
)
from apps.language_models.utils import (
    get_vmfb_from_path,
)
from shark.shark_downloader import download_public_file
from shark.shark_importer import get_f16_inputs
from shark.shark_importer import import_with_fx
from shark.shark_inference import SharkInference

from brevitas_examples.llm.llm_quant.quantize import quantize_model
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl

if __name__ == "__main__":
    import gc


parser = argparse.ArgumentParser(
    prog="vicuna runner",
    description="runs a vicuna model",
)
parser.add_argument(
    "--precision", "-p", default="fp32", help="fp32, fp16, int8, int4"
)
parser.add_argument("--device", "-d", default="cuda", help="vulkan, cpu, cuda")
parser.add_argument(
    "--first_vicuna_vmfb_path", default=None, help="path to first vicuna vmfb"
)
parser.add_argument(
    "-s",
    "--sharded",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Run model as sharded",
)
# TODO: sharded config
parser.add_argument(
    "--second_vicuna_vmfb_path",
    default=None,
    help="path to second vicuna vmfb",
)
parser.add_argument(
    "--first_vicuna_mlir_path",
    default=None,
    help="path to first vicuna mlir file",
)
parser.add_argument(
    "--second_vicuna_mlir_path",
    default=None,
    help="path to second vicuna mlir",
)
parser.add_argument(
    "--load_mlir_from_shark_tank",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="download precompile mlir from shark tank",
)
parser.add_argument(
    "--cli",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Run model in cli mode",
)
parser.add_argument(
    "--config",
    default=None,
    help="configuration file",
)
parser.add_argument(
    "--weight-group-size",
    type=int,
    default=128,
    help="Group size for per_group weight quantization. Default: 128.",
)


def brevitas〇matmul_rhs_group_quant〡shape(lhs: List[int], rhs: List[int], rhs_scale: List[int], rhs_zero_point: List[int], rhs_bit_width: int, rhs_group_size: int) -> List[int]:
    if len(lhs) == 3 and len(rhs) == 2:
        return [lhs[0], lhs[1], rhs[0]]
    elif len(lhs) == 2 and len(rhs) == 2:
        return [lhs[0], rhs[0]]
    else:
        raise ValueError("Input shapes not supported.")


def brevitas〇matmul_rhs_group_quant〡dtype(lhs_rank_dtype: Tuple[int, int], rhs_rank_dtype: Tuple[int, int], rhs_scale_rank_dtype: Tuple[int, int], rhs_zero_point_rank_dtype: Tuple[int, int], rhs_bit_width: int, rhs_group_size: int) -> int:
    # output dtype is the dtype of the lhs float input
    lhs_rank, lhs_dtype = lhs_rank_dtype
    return lhs_dtype


def brevitas〇matmul_rhs_group_quant〡has_value_semantics(lhs, rhs, rhs_scale, rhs_zero_point, rhs_bit_width, rhs_group_size) -> None:
    return


brevitas_matmul_rhs_group_quant_library = [
    brevitas〇matmul_rhs_group_quant〡shape,
    brevitas〇matmul_rhs_group_quant〡dtype,
    brevitas〇matmul_rhs_group_quant〡has_value_semantics]

class EightLayerLayerSV(torch.nn.Module):

    def __init__(self, layers):
        super().__init__()
        assert(len(layers) == 8)
        self.layers = layers

    def forward(self, hidden_states, attention_mask, position_ids, pkv00, pkv01, pkv10, pkv11, pkv20, pkv21, pkv30, pkv31, pkv40, pkv41, pkv50, pkv51, pkv60, pkv61, pkv70, pkv71):
        pkvs = [(pkv00, pkv01), (pkv10, pkv11), (pkv20, pkv21), (pkv30, pkv31), (pkv40, pkv41), (pkv50, pkv51), (pkv60, pkv61), (pkv70, pkv71)]
        new_pkvs = []
        for layer, pkv in zip(self.layers, pkvs):
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(
                    pkv[0],
                    pkv[1],
                ),
                use_cache=True,
            )

            hidden_states = outputs[0]
            new_pkvs.append((outputs[-1][0], outputs[-1][1], ))
        ((new_pkv00, new_pkv01), (new_pkv10, new_pkv11), (new_pkv20, new_pkv21), (new_pkv30, new_pkv31), (new_pkv40, new_pkv41), (new_pkv50, new_pkv51), (new_pkv60, new_pkv61), (new_pkv70, new_pkv71)) = new_pkvs
        return hidden_states, new_pkv00, new_pkv01, new_pkv10, new_pkv11, new_pkv20, new_pkv21, new_pkv30, new_pkv31, new_pkv40, new_pkv41, new_pkv50, new_pkv51, new_pkv60, new_pkv61, new_pkv70, new_pkv71

class EightLayerLayerFV(torch.nn.Module):

    def __init__(self, layers):
        super().__init__()
        assert(len(layers) == 8)
        self.layers = layers

    def forward(self, hidden_states, attention_mask, position_ids):
        new_pkvs = []
        for layer in self.layers:
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=True,
            )

            hidden_states = outputs[0]
            new_pkvs.append((outputs[-1][0], outputs[-1][1], ))
        ((new_pkv00, new_pkv01), (new_pkv10, new_pkv11), (new_pkv20, new_pkv21), (new_pkv30, new_pkv31), (new_pkv40, new_pkv41), (new_pkv50, new_pkv51), (new_pkv60, new_pkv61), (new_pkv70, new_pkv71)) = new_pkvs
        return hidden_states, new_pkv00, new_pkv01, new_pkv10, new_pkv11, new_pkv20, new_pkv21, new_pkv30, new_pkv31, new_pkv40, new_pkv41, new_pkv50, new_pkv51, new_pkv60, new_pkv61, new_pkv70, new_pkv71


class CompiledEightLayerLayerSV(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions=False,
        use_cache=True,
    ):
        hidden_states = hidden_states.detach()
        attention_mask = attention_mask.detach()
        position_ids = position_ids.detach()
        ((pkv00, pkv01), (pkv10, pkv11), (pkv20, pkv21), (pkv30, pkv31), (pkv40, pkv41), (pkv50, pkv51), (pkv60, pkv61), (pkv70, pkv71)) = past_key_value
        pkv00 = pkv00.detatch()
        pkv01 = pkv01.detatch()
        pkv10 = pkv10.detatch()
        pkv11 = pkv11.detatch()
        pkv20 = pkv20.detatch()
        pkv21 = pkv21.detatch()
        pkv30 = pkv30.detatch()
        pkv31 = pkv31.detatch()
        pkv40 = pkv40.detatch()
        pkv41 = pkv41.detatch()
        pkv50 = pkv50.detatch()
        pkv51 = pkv51.detatch()
        pkv60 = pkv60.detatch()
        pkv61 = pkv61.detatch()
        pkv70 = pkv70.detatch()
        pkv71 = pkv71.detatch()

        output = self.model("forward", (hidden_states, attention_mask, position_ids, pkv00, pkv01, pkv10, pkv11, pkv20, pkv21, pkv30, pkv31, pkv40, pkv41, pkv50, pkv51, pkv60, pkv61, pkv70, pkv71))
        return (output[0], (output[1][0], output[1][1]), (output[2][0], output[2][1]),(output[3][0], output[3][1]),(output[4][0], output[4][1]),(output[5][0], output[5][1]),(output[6][0], output[6][1]),(output[7][0], output[7][1]),(output[8][0], output[8][1]),)

def forward_compressed(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.compressedlayers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[8 * idx:8 * (idx + 1)] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
from time import time
class CompiledEightLayerLayer(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value = None,
        output_attentions=False,
        use_cache=True,
    ):
        t2 = time()
        if past_key_value is None:
            hidden_states = hidden_states.detach()
            attention_mask = attention_mask.detach()
            position_ids = position_ids.detach()
            t1 = time()

            output = self.model("first_vicuna_forward", (hidden_states, attention_mask, position_ids))
            #output2 = (output[0], (output[1], output[2],), (output[3], output[4],),(output[5], output[6],),(output[7], output[8],),(output[9],output[10],),(output[11], output[12],),(output[13], output[14],),(output[15], output[16],),)
            #return output2
            return (torch.tensor(output[0]), (torch.tensor(output[1]), torch.tensor(output[2]),), (torch.tensor(output[3]), torch.tensor(output[4]),),(torch.tensor(output[5]), torch.tensor(output[6]),),(torch.tensor(output[7]), torch.tensor(output[8]),),(torch.tensor(output[9]), torch.tensor(output[10]),),(torch.tensor(output[11]), torch.tensor(output[12]),),(torch.tensor(output[13]), torch.tensor(output[14]),),(torch.tensor(output[15]), torch.tensor(output[16]),),)
        else:
            ((pkv00, pkv01), (pkv10, pkv11), (pkv20, pkv21), (pkv30, pkv31), (pkv40, pkv41), (pkv50, pkv51), (pkv60, pkv61), (pkv70, pkv71)) = past_key_value
            
            try:
                hidden_states = hidden_states.detach()
                attention_mask = attention_mask.detach()
                position_ids = position_ids.detach()
                pkv00 = pkv00.detach()
                pkv01 = pkv01.detach()
                pkv10 = pkv10.detach()
                pkv11 = pkv11.detach()
                pkv20 = pkv20.detach()
                pkv21 = pkv21.detach()
                pkv30 = pkv30.detach()
                pkv31 = pkv31.detach()
                pkv40 = pkv40.detach()
                pkv41 = pkv41.detach()
                pkv50 = pkv50.detach()
                pkv51 = pkv51.detach()
                pkv60 = pkv60.detach()
                pkv61 = pkv61.detach()
                pkv70 = pkv70.detach()
                pkv71 = pkv71.detach()
            except:
                x = 10
            
            t1 = time()
            if type(hidden_states) == iree.runtime.array_interop.DeviceArray:
                hidden_states = np.array(hidden_states, hidden_states.dtype)
                hidden_states = torch.tensor(hidden_states)
                hidden_states = hidden_states.detach()

            
            output = self.model("second_vicuna_forward", (hidden_states, attention_mask, position_ids, pkv00, pkv01, pkv10, pkv11, pkv20, pkv21, pkv30, pkv31, pkv40, pkv41, pkv50, pkv51, pkv60, pkv61, pkv70, pkv71), send_to_host=False)
            del pkv00
            del pkv01
            del pkv10
            del pkv11
            del pkv20
            del pkv21
            del pkv30
            del pkv31
            del pkv40
            del pkv41
            del pkv50
            del pkv51
            del pkv60
            del pkv61
            del pkv70
            del pkv71
            #print(f"sv0 pass completed in {time() - t2} seconds")
            """
            try:
                pkv00 = np.asarray(pkv00, pkv00.dtype)
                pkv01 = np.asarray(pkv01, pkv01.dtype)
                pkv10 = np.asarray(pkv10, pkv10.dtype)
                pkv11 = np.asarray(pkv11, pkv11.dtype)
                pkv20 = np.asarray(pkv20, pkv20.dtype)
                pkv21 = np.asarray(pkv21, pkv21.dtype)
                pkv30 = np.asarray(pkv30, pkv30.dtype)
                pkv31 = np.asarray(pkv31, pkv31.dtype)
                pkv40 = np.asarray(pkv40, pkv40.dtype)
                pkv41 = np.asarray(pkv41, pkv41.dtype)
                pkv50 = np.asarray(pkv50, pkv50.dtype)
                pkv51 = np.asarray(pkv51, pkv51.dtype)
                pkv60 = np.asarray(pkv60, pkv60.dtype)
                pkv61 = np.asarray(pkv61, pkv61.dtype)
                pkv70 = np.asarray(pkv70, pkv70.dtype)
                pkv71 = np.asarray(pkv71, pkv71.dtype)
                print("iree arrays converted")
            except:
                x = 10
            """
            output2 = (output[0], (output[1], output[2],), (output[3], output[4],),(output[5], output[6],),(output[7], output[8],),(output[9],output[10],),(output[11], output[12],),(output[13], output[14],),(output[15], output[16],),)
            #output2 = (torch.tensor(output[0]), (torch.tensor(output[1]), torch.tensor(output[2]),), (torch.tensor(output[3]), torch.tensor(output[4]),),(torch.tensor(output[5]), torch.tensor(output[6]),),(torch.tensor(output[7]), torch.tensor(output[8]),),(torch.tensor(output[9]), torch.tensor(output[10]),),(torch.tensor(output[11]), torch.tensor(output[12]),),(torch.tensor(output[13]), torch.tensor(output[14]),),(torch.tensor(output[15]), torch.tensor(output[16]),),)
            #print(output2[1][0])
            return output2



class ShardedVicuna(SharkLLMBase):
    # Class representing Sharded Vicuna Model
    def __init__(
        self,
        model_name,
        hf_model_path="TheBloke/vicuna-7B-1.1-HF",
        max_num_tokens=512,
        device="cuda",
        precision="fp32",
        config_json=None,
        weight_group_size=128,
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.tokenizer = self.get_tokenizer()
        self.config = config_json
        self.weight_group_size = weight_group_size
        self.shark_model = self.compile(device=device)

    def get_tokenizer(self):
        # Retrieve the tokenizer from Huggingface
        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_path, use_fast=False
        )
        return tokenizer

    def get_src_model(self):
        # Retrieve the torch model from Huggingface
        kwargs = {"torch_dtype": torch.float}
        vicuna_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path, **kwargs
        )
        return vicuna_model

    def write_in_dynamic_inputs0(self, module, dynamic_input_size):
        # Current solution for ensuring mlir files support dynamic inputs
        # TODO find a more elegant way to implement this
        new_lines = []
        for line in module.splitlines():
            line = re.sub(f"{dynamic_input_size}x", "?x", line)
            if "?x" in line:
                line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
            line = re.sub(f" {dynamic_input_size},", " %dim,", line)
            if "tensor.empty" in line and "?x?" in line:
                line = re.sub(
                    "tensor.empty\(%dim\)", "tensor.empty(%dim, %dim)", line
                )
            if "arith.cmpi" in line:
                line = re.sub(f"c{dynamic_input_size}", "dim", line)
            new_lines.append(line)
        new_module = "\n".join(new_lines)
        return new_module

    def write_in_dynamic_inputs1(self, module, dynamic_input_size):
        new_lines = []
        for line in module.splitlines():
            if "dim_42 =" in line:
                continue
            if f"%c{dynamic_input_size}_i64 =" in line:
                new_lines.append(
                    "%dim_42 = tensor.dim %arg1, %c3 : tensor<1x1x1x?xf32>"
                )
                new_lines.append(
                    f"%dim_42_i64 = arith.index_cast %dim_42 : index to i64"
                )
                continue
            line = re.sub(f"{dynamic_input_size}x", "?x", line)
            line = re.sub(f"%c{dynamic_input_size}_i64", "%dim_42_i64", line)
            if "?x" in line:
                line = re.sub(
                    "tensor.empty\(\)", "tensor.empty(%dim_42)", line
                )
            line = re.sub(f" {dynamic_input_size},", " %dim_42,", line)
            if "tensor.empty" in line and "?x?" in line:
                line = re.sub(
                    "tensor.empty\(%dim_42\)",
                    "tensor.empty(%dim_42, %dim_42)",
                    line,
                )
            if "arith.cmpi" in line:
                line = re.sub(f"c{dynamic_input_size}", "dim_42", line)
            new_lines.append(line)
        new_module = "\n".join(new_lines)
        return new_module

    def combine_mlir_scripts(
        self, first_vicuna_mlir, second_vicuna_mlir, output_name
    ):
        maps1 = []
        maps2 = []
        constants = set()
        f1 = []
        f2 = []

        for line in first_vicuna_mlir.splitlines():
            if re.search("#map\d*\s*=", line):
                maps1.append(line)
            elif re.search("arith.constant", line):
                constants.add(line)
            elif not re.search("module", line):
                line = re.sub("forward", "first_vicuna_forward", line)
                f1.append(line)
        f1 = f1[:-1]

        for i, map_line in enumerate(maps1):
            map_var = map_line.split(" ")[0]
            map_line = re.sub(f"{map_var}(?!\d)", map_var + "_0", map_line)
            maps1[i] = map_line
            f1 = [
                re.sub(f"{map_var}(?!\d)", map_var + "_0", func_line)
                for func_line in f1
            ]

        for line in second_vicuna_mlir.splitlines():
            if re.search("#map\d*\s*=", line):
                maps2.append(line)
            elif "global_seed" in line:
                continue
            elif re.search("arith.constant", line):
                constants.add(line)
            elif not re.search("module", line):
                line = re.sub("forward", "second_vicuna_forward", line)
                f2.append(line)
        f2 = f2[:-1]

        for i, map_line in enumerate(maps2):
            map_var = map_line.split(" ")[0]
            map_line = re.sub(f"{map_var}(?!\d)", map_var + "_1", map_line)
            maps2[i] = map_line
            f2 = [
                re.sub(f"{map_var}(?!\d)", map_var + "_1", func_line)
                for func_line in f2
            ]

        module_start = (
            'module attributes {torch.debug_module_name = "_lambda"} {'
        )
        module_end = "}"

        global_vars = []
        vnames = []
        vdtypes = []
        global_var_loading1 = []
        global_var_loading2 = []

        for constant in list(constants):
            vname, vbody = constant.split("=")
            vname = re.sub("%", "", vname)
            vname = vname.strip()
            vbody = re.sub("arith.constant", "", vbody)
            vbody = vbody.strip()
            if ":" in vbody:
                vdtype = vbody.split(":")[1].strip()
            else:
                vdtype = vbody.split(" ")[-1].strip()
            fixed_vdtype = vdtype
            vdtypes.append(vdtype)
            vdtype = re.sub("\d{1,}x", "?x", vdtype)
            vnames.append(vname)
            global_vars.append(
                f"ml_program.global public @{vname}({vbody}) : {fixed_vdtype}"
            )
            global_var_loading1.append(
                f"\t\t%{vname} = ml_program.global_load_const @{vname} : {fixed_vdtype}"
            )
            global_var_loading2.append(
                f"\t\t%{vname} = ml_program.global_load_const @{vname} : {fixed_vdtype}"
            )

        new_f1, new_f2 = [], []

        for line in f1:
            if "func.func" in line:
                new_f1.append(line)
                for global_var in global_var_loading1:
                    new_f1.append(global_var)
            else:
                new_f1.append(line)

        for line in f2:
            if "func.func" in line:
                new_f2.append(line)
                for global_var in global_var_loading1:
                    new_f2.append(global_var)
            else:
                new_f2.append(line)

        f1 = new_f1
        f2 = new_f2

        whole_string = "\n".join(
            maps1
            + maps2
            + [module_start]
            + global_vars
            + f1
            + f2
            + [module_end]
        )

        f_ = open(output_name, "w+")
        f_.write(whole_string)
        f_.close()

        return whole_string

    def compile_vicuna_layer(
        self,
        vicuna_layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values = None,
    ):
        # Compile a hidden decoder layer of vicuna
        if past_key_values is None:
            model_inputs = (hidden_states, attention_mask, position_ids)
        else:
            ((pkv00, pkv01), (pkv10, pkv11), (pkv20, pkv21), (pkv30, pkv31), (pkv40, pkv41), (pkv50, pkv51), (pkv60, pkv61), (pkv70, pkv71)) = past_key_values


            model_inputs = (
                hidden_states,
                attention_mask,
                position_ids,
                pkv00, pkv01, pkv10, pkv11, pkv20, pkv21, pkv30, pkv31, pkv40, pkv41, pkv50, pkv51, pkv60, pkv61, pkv70, pkv71
            )
        mlir_bytecode = import_with_fx(
            vicuna_layer,
            model_inputs,
            precision=self.precision,
            f16_input_mask=[False, False],
            mlir_type="torchscript",
        )
        return mlir_bytecode

    def get_device_index(self, layer_string):
        # Get the device index from the config file
        # In the event that different device indices are assigned to
        # different parts of a layer, a majority vote will be taken and
        # everything will be run on the most commonly used device
        if self.config is None:
            return None
        idx_votes = {}
        for key in self.config.keys():
            if re.search(layer_string, key):
                if int(self.config[key]["gpu"]) in idx_votes.keys():
                    idx_votes[int(self.config[key]["gpu"])] += 1
                else:
                    idx_votes[int(self.config[key]["gpu"])] = 1
        device_idx = max(idx_votes, key=idx_votes.get)
        return device_idx

    def compile_lmhead(
        self, lmh, hidden_states, device="cpu", device_idx=None
    ):
        # compile the lm head of the vicuna model
        # This can be used for both first and second vicuna, so only needs to be run once
        mlir_path = Path(f"lmhead.mlir")
        vmfb_path = Path(f"lmhead.vmfb")
        if mlir_path.exists():
            f_ = open(mlir_path, "rb")
            bytecode = f_.read()
            f_.close()
        else:
            hidden_states = torch_mlir.TensorPlaceholder.like(
                hidden_states, dynamic_axes=[1]
            )

            module = torch_mlir.compile(
                lmh,
                (hidden_states,),
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

        shark_module = SharkInference(
            bytecode,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
            mmap=True,
        )
        if vmfb_path.exists():
            shark_module.load_module(vmfb_path)
        else:
            shark_module.save_module(module_name="lmhead")
            shark_module.load_module(vmfb_path)
        compiled_module = LMHeadCompiled(shark_module)
        return compiled_module

    def compile_norm(self, fvn, hidden_states, device="cpu", device_idx=None):
        # compile the normalization layer of the vicuna model
        # This can be used for both first and second vicuna, so only needs to be run once
        mlir_path = Path(f"norm.mlir")
        vmfb_path = Path(f"norm.vmfb")
        if mlir_path.exists():
            f_ = open(mlir_path, "rb")
            bytecode = f_.read()
            f_.close()
        else:
            hidden_states = torch_mlir.TensorPlaceholder.like(
                hidden_states, dynamic_axes=[1]
            )

            module = torch_mlir.compile(
                fvn,
                (hidden_states,),
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

        shark_module = SharkInference(
            bytecode,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
            mmap=True,
        )
        if vmfb_path.exists():
            shark_module.load_module(vmfb_path)
        else:
            shark_module.save_module(module_name="norm")
            shark_module.load_module(vmfb_path)
        compiled_module = VicunaNormCompiled(shark_module)
        return compiled_module

    def compile_embedding(self, fve, input_ids, device="cpu", device_idx=None):
        # compile the embedding layer of the vicuna model
        # This can be used for both first and second vicuna, so only needs to be run once
        mlir_path = Path(f"embedding.mlir")
        vmfb_path = Path(f"embedding.vmfb")
        if mlir_path.exists():
            f_ = open(mlir_path, "rb")
            bytecode = f_.read()
            f_.close()
        else:
            input_ids = torch_mlir.TensorPlaceholder.like(
                input_ids, dynamic_axes=[1]
            )
            module = torch_mlir.compile(
                fve,
                (input_ids,),
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

        shark_module = SharkInference(
            bytecode,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
            mmap=True,
        )
        if vmfb_path.exists():
            shark_module.load_module(vmfb_path)
        else:
            shark_module.save_module(module_name="embedding")
            shark_module.load_module(vmfb_path)
        compiled_module = VicunaEmbeddingCompiled(shark_module)

        return compiled_module

    def compile_to_vmfb_one_model(
        self, inputs0, layers0, inputs1, layers1, device="cpu"
    ):
        mlirs, modules = [], []
        assert len(layers0) == len(layers1)
        for layer0, layer1, idx in zip(layers0, layers1, range(len(layers0))):
            mlir_path = Path(f"{idx}_full.mlir")
            vmfb_path = Path(f"{idx}_full.vmfb")
            # if vmfb_path.exists():
            #    continue
            if mlir_path.exists():
                # print(f"Found layer {idx} mlir")
                f_ = open(mlir_path, "rb")
                bytecode = f_.read()
                f_.close()
                mlirs.append(bytecode)
            else:
                hidden_states_placeholder0 = TensorPlaceholder.like(
                    inputs0[0], dynamic_axes=[1]
                )
                attention_mask_placeholder0 = TensorPlaceholder.like(
                    inputs0[1], dynamic_axes=[3]
                )
                position_ids_placeholder0 = TensorPlaceholder.like(
                    inputs0[2], dynamic_axes=[1]
                )
                hidden_states_placeholder1 = TensorPlaceholder.like(
                    inputs1[0], dynamic_axes=[1]
                )
                attention_mask_placeholder1 = TensorPlaceholder.like(
                    inputs1[1], dynamic_axes=[3]
                )
                position_ids_placeholder1 = TensorPlaceholder.like(
                    inputs1[2], dynamic_axes=[1]
                )

                
                pkv00_placeholder = TensorPlaceholder.like(inputs1[3][0][0], dynamic_axes=[2])
                pkv01_placeholder = TensorPlaceholder.like(inputs1[3][0][1], dynamic_axes=[2])
                pkv10_placeholder = TensorPlaceholder.like(inputs1[3][1][0], dynamic_axes=[2])
                pkv11_placeholder = TensorPlaceholder.like(inputs1[3][1][1], dynamic_axes=[2])
                pkv20_placeholder = TensorPlaceholder.like(inputs1[3][2][0], dynamic_axes=[2])
                pkv21_placeholder = TensorPlaceholder.like(inputs1[3][2][1], dynamic_axes=[2])
                pkv30_placeholder = TensorPlaceholder.like(inputs1[3][3][0], dynamic_axes=[2])
                pkv31_placeholder = TensorPlaceholder.like(inputs1[3][3][1], dynamic_axes=[2])
                pkv40_placeholder = TensorPlaceholder.like(inputs1[3][4][0], dynamic_axes=[2])
                pkv41_placeholder = TensorPlaceholder.like(inputs1[3][4][1], dynamic_axes=[2])
                pkv50_placeholder = TensorPlaceholder.like(inputs1[3][5][0], dynamic_axes=[2])
                pkv51_placeholder = TensorPlaceholder.like(inputs1[3][5][1], dynamic_axes=[2])
                pkv60_placeholder = TensorPlaceholder.like(inputs1[3][6][0], dynamic_axes=[2])
                pkv61_placeholder = TensorPlaceholder.like(inputs1[3][6][1], dynamic_axes=[2])
                pkv70_placeholder = TensorPlaceholder.like(inputs1[3][7][0], dynamic_axes=[2])
                pkv71_placeholder = TensorPlaceholder.like(inputs1[3][7][1], dynamic_axes=[2])

                print(f"Compiling layer {idx} mlir")
                ts_g = self.compile_vicuna_layer(
                    layer0, inputs0[0], inputs0[1], inputs0[2]
                )
                if self.precision in ["int4", "int8"]:
                    module0 = torch_mlir.compile(
                        ts_g,
                        (
                            hidden_states_placeholder0,
                            inputs0[1],
                            inputs0[2],
                        ),
                        output_type="torch",
                        backend_legal_ops=["brevitas.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    # TODO: apply --canonicalize to unpack tensor for int4
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        module0,
                        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    module0 = torch_mlir.compile(
                        ts_g,
                        (
                            hidden_states_placeholder0,
                            inputs0[1],
                            inputs0[2],
                        ),
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )
                module0 = self.write_in_dynamic_inputs0(str(module0), 137)

                ts_g = self.compile_vicuna_layer(
                    layer1,
                    inputs1[0],
                    inputs1[1],
                    inputs1[2],
                    inputs1[3]
                )
                if self.precision in ["int4", "int8"]:
                    module1 = torch_mlir.compile(
                        ts_g,
                        (
                            inputs1[0],
                            attention_mask_placeholder1,
                            inputs1[2],
                            pkv00_placeholder, pkv01_placeholder, pkv10_placeholder, pkv11_placeholder, pkv20_placeholder, pkv21_placeholder,pkv30_placeholder, pkv31_placeholder,pkv40_placeholder, pkv41_placeholder, pkv50_placeholder, pkv51_placeholder, pkv60_placeholder, pkv61_placeholder, pkv70_placeholder, pkv71_placeholder
                        
                        ),
                        output_type="torch",
                        backend_legal_ops=["brevitas.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    # TODO: apply --canonicalize to unpack tensor for int4
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        module1,
                        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    module1 = torch_mlir.compile(
                        ts_g,
                        (
                            inputs1[0],
                            attention_mask_placeholder1,
                            inputs1[2],
                            pkv00_placeholder, pkv01_placeholder, pkv10_placeholder, pkv11_placeholder, pkv20_placeholder, pkv21_placeholder,pkv30_placeholder, pkv31_placeholder,pkv40_placeholder, pkv41_placeholder, pkv50_placeholder, pkv51_placeholder, pkv60_placeholder, pkv61_placeholder, pkv70_placeholder, pkv71_placeholder
                        ),
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )
                module1 = self.write_in_dynamic_inputs1(str(module1), 138)

                module_combined = self.combine_mlir_scripts(
                    module0, module1, f"{idx}_full.mlir"
                )
                mlirs.append(module_combined)

            if vmfb_path.exists():
                # print(f"Found layer {idx} vmfb")
                device_idx = self.get_device_index(
                    f"first_vicuna.model.model.layers.{idx}[\s.$]"
                )
                module = SharkInference(
                    None,
                    device=device,
                    device_idx=idx % 4,
                    mlir_dialect="tm_tensor",
                    mmap=True,
                )
                module.load_module(vmfb_path)
            else:
                print(f"Compiling layer {idx} vmfb")
                device_idx = self.get_device_index(
                    f"first_vicuna.model.model.layers.{idx}[\s.$]"
                )
                module = SharkInference(
                    mlirs[idx],
                    device=device,
                    device_idx=idx % 4,
                    mlir_dialect="tm_tensor",
                    mmap=True,
                )
                module.save_module(
                    module_name=f"{idx}_full",
                    extra_args=[
                        "--iree-vm-target-truncate-unsupported-floats",
                        "--iree-codegen-check-ir-before-llvm-conversion=false",
                        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    ],
                )
                module.load_module(vmfb_path)
            modules.append(module)
        return mlirs, modules

    def get_sharded_model(self, device="cpu"):
        # SAMPLE_INPUT_LEN is used for creating mlir with dynamic inputs, which is currently an increadibly hacky proccess
        # please don't change it
        SAMPLE_INPUT_LEN = 137
        vicuna_model = self.get_src_model()

        if self.precision in ["int4", "int8"]:
            print("Applying weight quantization..")
            weight_bit_width = 4 if self.precision == "int4" else 8
            quantize_model(
                get_model_impl(vicuna_model).layers,
                dtype=torch.float32,
                weight_quant_type="asym",
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float",
                weight_quant_granularity="per_group",
                weight_group_size=self.weight_group_size,
                quantize_weight_zero_point=False,
                input_bit_width=None,
                input_scale_type="float",
                input_param_method="stats",
                input_quant_type="asym",
                input_quant_granularity="per_tensor",
                quantize_input_zero_point=False,
                seqlen=2048,
            )
            print("Weight quantization applied.")

        placeholder_pkv_segment = tuple((torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),) for _ in range(8))
        placeholder_pkv_full = tuple((torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),) for _ in range(32))

        placeholder_input0 = (
            torch.zeros([1, SAMPLE_INPUT_LEN, 4096]),
            torch.zeros([1, 1, SAMPLE_INPUT_LEN, SAMPLE_INPUT_LEN]),
            torch.zeros([1, SAMPLE_INPUT_LEN], dtype=torch.int64),
        )

        placeholder_input1 = (
            torch.zeros([1, 1, 4096]),
            torch.zeros([1, 1, 1, SAMPLE_INPUT_LEN + 1]),
            torch.zeros([1, 1], dtype=torch.int64),
            torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
            torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
        )

        placeholder_input2 = (
            torch.zeros([1, 1, 4096]),
            torch.zeros([1, 1, 1, SAMPLE_INPUT_LEN + 1]),
            torch.zeros([1, 1], dtype=torch.int64),
            placeholder_pkv_segment
        )

        norm = VicunaNorm(vicuna_model.model.norm)
        device_idx = self.get_device_index(
            r"vicuna\.model\.model\.norm(?:\.|\s|$)"
        )
        print(device_idx)
        norm = self.compile_norm(
            norm,
            torch.zeros([1, SAMPLE_INPUT_LEN, 4096]),
            device=self.device,
            device_idx=device_idx,
        )

        embeddings = VicunaEmbedding(vicuna_model.model.embed_tokens)
        device_idx = self.get_device_index(
            r"vicuna\.model\.model\.embed_tokens(?:\.|\s|$)"
        )
        print(device_idx)
        embeddings = self.compile_embedding(
            embeddings,
            (torch.zeros([1, SAMPLE_INPUT_LEN], dtype=torch.int64)),
            device=self.device,
            device_idx=device_idx,
        )

        lmhead = LMHead(vicuna_model.lm_head)
        device_idx = self.get_device_index(
            r"vicuna\.model\.lm_head(?:\.|\s|$)"
        )
        print(device_idx)
        lmhead = self.compile_lmhead(
            lmhead,
            torch.zeros([1, SAMPLE_INPUT_LEN, 4096]),
            device=self.device,
            device_idx=device_idx,
        )

        layers0 = [
            FirstVicunaLayer(layer) for layer in vicuna_model.model.layers
        ]
        layers1 = [
            SecondVicunaLayer(layer) for layer in vicuna_model.model.layers
        ]

        layers00 = EightLayerLayerFV(vicuna_model.model.layers[0:8])
        layers01 = EightLayerLayerFV(vicuna_model.model.layers[8:16])
        layers02 = EightLayerLayerFV(vicuna_model.model.layers[16:24])
        layers03 = EightLayerLayerFV(vicuna_model.model.layers[24:32])
        layers10 = EightLayerLayerSV(vicuna_model.model.layers[0:8])
        layers11 = EightLayerLayerSV(vicuna_model.model.layers[8:16])
        layers12 = EightLayerLayerSV(vicuna_model.model.layers[16:24])
        layers13 = EightLayerLayerSV(vicuna_model.model.layers[24:32])
        layers0 = [layers00, layers01, layers02, layers03]
        layers1 = [layers10, layers11, layers12, layers13]
        #vicuna_model.model.forward = forward_compressed
        
        _, modules = self.compile_to_vmfb_one_model(
            placeholder_input0,
            layers0,
            placeholder_input2,
            layers1,
            device=device,
        )
        shark_layers = [CompiledEightLayerLayer(m) for m in modules]
        vicuna_model.model.compressedlayers = shark_layers

        sharded_model = ShardedVicunaModel(
            vicuna_model,
            shark_layers,
            lmhead,
            embeddings,
            norm,
        )
        return sharded_model

    def compile(self, device="cpu"):
        return self.get_sharded_model(device=device)

    def generate(self, prompt, cli=False):
        # TODO: refactor for cleaner integration

        tokens_generated = []
        _past_key_values = None
        _token = None
        detoks_generated = []
        for iteration in range(self.max_num_tokens):
            params = {
                "prompt": prompt,
                "is_first": iteration == 0,
                "token": _token,
                "past_key_values": _past_key_values,
            }

            generated_token_op = self.generate_new_token(params=params)

            _token = generated_token_op["token"]
            _past_key_values = generated_token_op["past_key_values"]
            _detok = generated_token_op["detok"]

            if _token == 2:
                break
            detoks_generated.append(_detok)
            tokens_generated.append(_token)

        for i in range(len(tokens_generated)):
            if type(tokens_generated[i]) != int:
                tokens_generated[i] = int(tokens_generated[i][0])
        result_output = self.tokenizer.decode(tokens_generated)
        return result_output

    def generate_new_token(self, params):
        is_first = params["is_first"]
        if is_first:
            prompt = params["prompt"]
            input_ids = self.tokenizer(prompt).input_ids
            #crop input_ids
            input_ids = input_ids[len(input_ids) - 20:]
            ############
            input_id_len = len(input_ids)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.reshape([1, input_id_len])
            output = self.shark_model.forward(input_ids, is_first=is_first)
        else:
            token = params["token"]
            past_key_values = params["past_key_values"]
            input_ids = [token]
            input_id_len = len(input_ids)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.reshape([1, input_id_len])
            output = self.shark_model.forward(
                input_ids, past_key_values=past_key_values, is_first=is_first
            )

        _logits = output["logits"]
        _past_key_values = output["past_key_values"]
        _token = int(torch.argmax(_logits[:, -1, :], dim=1)[0])
        _detok = self.tokenizer.decode(_token)

        ret_dict = {
            "token": _token,
            "detok": _detok,
            "past_key_values": _past_key_values,
        }

        print(f" token : {_token} | detok : {_detok}")

        return ret_dict

    def autocomplete(self, prompt):
        # use First vic alone to complete a story / prompt / sentence.
        pass


class UnshardedVicuna(SharkLLMBase):
    def __init__(
        self,
        model_name,
        hf_model_path="TheBloke/vicuna-7B-1.1-HF",
        max_num_tokens=512,
        device="cuda",
        precision="fp32",
        first_vicuna_mlir_path=None,
        second_vicuna_mlir_path=None,
        first_vicuna_vmfb_path=None,
        second_vicuna_vmfb_path=None,
        load_mlir_from_shark_tank=True,
        low_device_memory=False,
        weight_group_size=128,
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.first_vicuna_vmfb_path = first_vicuna_vmfb_path
        self.second_vicuna_vmfb_path = second_vicuna_vmfb_path
        self.first_vicuna_mlir_path = first_vicuna_mlir_path
        self.second_vicuna_mlir_path = second_vicuna_mlir_path
        self.load_mlir_from_shark_tank = load_mlir_from_shark_tank
        self.low_device_memory = low_device_memory
        self.weight_group_size = weight_group_size
        self.first_vic = None
        self.second_vic = None
        if self.first_vicuna_mlir_path == None:
            self.first_vicuna_mlir_path = self.get_model_path()
        if self.second_vicuna_mlir_path == None:
            self.second_vicuna_mlir_path = self.get_model_path("second")
        if self.first_vicuna_vmfb_path == None:
            self.first_vicuna_vmfb_path = self.get_model_path(suffix="vmfb")
        if self.second_vicuna_vmfb_path == None:
            self.second_vicuna_vmfb_path = self.get_model_path(
                "second", "vmfb"
            )
        self.tokenizer = self.get_tokenizer()
        self.shark_model = self.compile()

    def get_model_path(self, model_number="first", suffix="mlir"):
        safe_device = self.device.split("-")[0]
        if suffix == "mlir":
            return Path(f"{model_number}_vicuna_{self.precision}.{suffix}")
        return Path(
            f"{model_number}_vicuna_{self.precision}_{safe_device}.{suffix}"
        )

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_path, use_fast=False
        )
        return tokenizer

    def get_src_model(self):
        kwargs = {"torch_dtype": torch.float}
        vicuna_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path, **kwargs
        )
        return vicuna_model

    def compile_first_vicuna(self):
        vmfb = get_vmfb_from_path(
            self.first_vicuna_vmfb_path, self.device, "tm_tensor"
        )
        if vmfb is not None:
            return vmfb

        # Compilation path needs some more work before it is functional
        print(
            f"[DEBUG] vmfb not found at {self.first_vicuna_vmfb_path.absolute()}. Trying to work with\n"
            f"[DEBUG] mlir path { self.first_vicuna_mlir_path} {'exists' if self.first_vicuna_mlir_path.exists() else 'does not exist'}"
        )
        if self.first_vicuna_mlir_path.exists():
            with open(self.first_vicuna_mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            mlir_generated = False
            if self.load_mlir_from_shark_tank:
                if self.precision in ["fp32", "fp16", "int8", "int4"]:
                    # download MLIR from shark_tank
                    download_public_file(
                        f"gs://shark_tank/vicuna/unsharded/mlir/{self.first_vicuna_mlir_path.name}",
                        self.first_vicuna_mlir_path.absolute(),
                        single_file=True,
                    )
                    if self.first_vicuna_mlir_path.exists():
                        with open(self.first_vicuna_mlir_path, "rb") as f:
                            bytecode = f.read()
                        mlir_generated = True
                    else:
                        raise ValueError(
                            f"MLIR not found at {self.first_vicuna_mlir_path.absolute()}"
                            " after downloading! Please check path and try again"
                        )
                else:
                    print(
                        f"Only fp32/fp16/int8/int4 mlir added to tank, generating {self.precision} mlir on device."
                    )

            if not mlir_generated:
                compilation_prompt = "".join(["0" for _ in range(17)])
                compilation_input_ids = self.tokenizer(
                    compilation_prompt
                ).input_ids
                compilation_input_ids = torch.tensor(
                    compilation_input_ids
                ).reshape([1, 19])
                firstVicunaCompileInput = (compilation_input_ids,)
                model = FirstVicuna(
                    self.hf_model_path, self.precision, self.weight_group_size
                )

                print(f"[DEBUG] generating torchscript graph")
                ts_graph = import_with_fx(
                    model,
                    firstVicunaCompileInput,
                    precision=self.precision,
                    f16_input_mask=[False, False],
                    mlir_type="torchscript",
                )
                del model

                firstVicunaCompileInput = list(firstVicunaCompileInput)
                firstVicunaCompileInput[0] = torch_mlir.TensorPlaceholder.like(
                    firstVicunaCompileInput[0], dynamic_axes=[1]
                )
                firstVicunaCompileInput = tuple(firstVicunaCompileInput)

                print(f"[DEBUG] generating torch mlir")
                if self.precision in ["int4", "int8"]:
                    module = torch_mlir.compile(
                        ts_graph,
                        [*firstVicunaCompileInput],
                        output_type=torch_mlir.OutputType.TORCH,
                        backend_legal_ops=["brevitas.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    # TODO: apply --canonicalize to unpack tensor for int4
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        module,
                        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    module = torch_mlir.compile(
                        ts_graph,
                        [*firstVicunaCompileInput],
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )
                del ts_graph

                def remove_constant_dim(line):
                    if "19x" in line:
                        line = re.sub("19x", "?x", line)
                        line = re.sub(
                            "tensor.empty\(\)", "tensor.empty(%dim)", line
                        )
                    if "tensor.empty" in line and "?x?" in line:
                        line = re.sub(
                            "tensor.empty\(%dim\)",
                            "tensor.empty(%dim, %dim)",
                            line,
                        )
                    if "arith.cmpi" in line:
                        line = re.sub("c19", "dim", line)
                    if " 19," in line:
                        line = re.sub(" 19,", " %dim,", line)
                    return line

                module = str(module)
                new_lines = []

                print(f"[DEBUG] rewriting torch_mlir file")
                for line in module.splitlines():
                    line = remove_constant_dim(line)
                    if "%0 = tensor.empty(%dim) : tensor<?xi64>" in line:
                        new_lines.append(
                            "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>"
                        )
                    if (
                        "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>"
                        in line
                    ):
                        continue

                    new_lines.append(line)

                module = "\n".join(new_lines)
                print(f"[DEBUG] converting to bytecode")
                del new_lines
                module = module.encode("UTF-8")
                module = BytesIO(module)
                bytecode = module.read()
                del module

                print(f"[DEBUG] writing mlir to file")
                f_ = open(self.first_vicuna_mlir_path, "wb")
                f_.write(bytecode)
                f_.close()

        shark_module = SharkInference(
            mlir_module=bytecode, device=self.device, mlir_dialect="tm_tensor"
        )
        path = shark_module.save_module(
            self.first_vicuna_vmfb_path.parent.absolute(),
            self.first_vicuna_vmfb_path.stem,
            extra_args=[
                "--iree-hal-dump-executable-sources-to=ies",
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ],
        )
        print("Saved first vic vmfb at ", str(path))
        shark_module.load_module(path)

        return shark_module

    def compile_second_vicuna(self):
        vmfb = get_vmfb_from_path(
            self.second_vicuna_vmfb_path, self.device, "tm_tensor"
        )
        if vmfb is not None:
            return vmfb

        # Compilation path needs some more work before it is functional
        print(
            f"[DEBUG] mlir path {self.second_vicuna_mlir_path} {'exists' if self.second_vicuna_mlir_path.exists() else 'does not exist'}"
        )
        if self.second_vicuna_mlir_path.exists():
            with open(self.second_vicuna_mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            mlir_generated = False
            if self.load_mlir_from_shark_tank:
                if self.precision in ["fp32", "fp16", "int8", "int4"]:
                    # download MLIR from shark_tank
                    download_public_file(
                        f"gs://shark_tank/vicuna/unsharded/mlir/{self.second_vicuna_mlir_path.name}",
                        self.second_vicuna_mlir_path.absolute(),
                        single_file=True,
                    )
                    if self.second_vicuna_mlir_path.exists():
                        with open(self.second_vicuna_mlir_path, "rb") as f:
                            bytecode = f.read()
                        mlir_generated = True
                    else:
                        raise ValueError(
                            f"MLIR not found at {self.second_vicuna_mlir_path.absolute()}"
                            " after downloading! Please check path and try again"
                        )
                else:
                    print(
                        "Only fp32/fp16/int8/int4 mlir added to tank, generating mlir on device."
                    )

            if not mlir_generated:
                compilation_input_ids = torch.zeros([1, 1], dtype=torch.int64)
                pkv = tuple(
                    (torch.zeros([1, 32, 19, 128], dtype=torch.float32))
                    for _ in range(64)
                )
                secondVicunaCompileInput = (compilation_input_ids,) + pkv
                model = SecondVicuna(
                    self.hf_model_path, self.precision, self.weight_group_size
                )

                print(f"[DEBUG] generating torchscript graph")
                ts_graph = import_with_fx(
                    model,
                    secondVicunaCompileInput,
                    precision=self.precision,
                    f16_input_mask=[False] + [True] * 64,
                    mlir_type="torchscript",
                )
                if self.precision == "fp16":
                    secondVicunaCompileInput = get_f16_inputs(
                        secondVicunaCompileInput,
                        True,
                        f16_input_mask=[False] + [True] * 64,
                    )
                secondVicunaCompileInput = list(secondVicunaCompileInput)
                for i in range(len(secondVicunaCompileInput)):
                    if i != 0:
                        secondVicunaCompileInput[
                            i
                        ] = torch_mlir.TensorPlaceholder.like(
                            secondVicunaCompileInput[i], dynamic_axes=[2]
                        )
                secondVicunaCompileInput = tuple(secondVicunaCompileInput)

                print(f"[DEBUG] generating torch mlir")
                if self.precision in ["int4", "int8"]:
                    module = torch_mlir.compile(
                        ts_graph,
                        [*secondVicunaCompileInput],
                        output_type=torch_mlir.OutputType.TORCH,
                        backend_legal_ops=["brevitas.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    # TODO: apply --canonicalize to unpack tensor for int4
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        module,
                        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    module = torch_mlir.compile(
                        ts_graph,
                        [*secondVicunaCompileInput],
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )

                def remove_constant_dim(line):
                    if "c19_i64" in line:
                        line = re.sub("c19_i64", "dim_i64", line)
                    if "19x" in line:
                        line = re.sub("19x", "?x", line)
                        line = re.sub(
                            "tensor.empty\(\)", "tensor.empty(%dim)", line
                        )
                    if "tensor.empty" in line and "?x?" in line:
                        line = re.sub(
                            "tensor.empty\(%dim\)",
                            "tensor.empty(%dim, %dim)",
                            line,
                        )
                    if "arith.cmpi" in line:
                        line = re.sub("c19", "dim", line)
                    if " 19," in line:
                        line = re.sub(" 19,", " %dim,", line)
                    if "20x" in line:
                        line = re.sub("20x", "?x", line)
                        line = re.sub(
                            "tensor.empty\(\)", "tensor.empty(%dimp1)", line
                        )
                    if " 20," in line:
                        line = re.sub(" 20,", " %dimp1,", line)
                    return line

                module_str = str(module)
                new_lines = []

                print(f"[DEBUG] rewriting torch_mlir file")
                for line in module_str.splitlines():
                    if "%c19_i64 = arith.constant 19 : i64" in line:
                        new_lines.append("%c2 = arith.constant 2 : index")
                        new_lines.append(
                            f"%dim_4_int = tensor.dim %arg1, %c2 : tensor<1x32x?x128x{'f16' if self.precision == 'fp16' else 'f32'}>"
                        )
                        new_lines.append(
                            "%dim_i64 = arith.index_cast %dim_4_int : index to i64"
                        )
                        continue
                    if "%c2 = arith.constant 2 : index" in line:
                        continue
                    if "%c20_i64 = arith.constant 20 : i64" in line:
                        new_lines.append("%c1_i64 = arith.constant 1 : i64")
                        new_lines.append(
                            "%c20_i64 = arith.addi %dim_i64, %c1_i64 : i64"
                        )
                        new_lines.append(
                            "%dimp1 = arith.index_cast %c20_i64 : i64 to index"
                        )
                        continue
                    line = remove_constant_dim(line)
                    new_lines.append(line)

                module_str = "\n".join(new_lines)
                print(f"[DEBUG] converting to bytecode")
                bytecode = module_str.encode("UTF-8")
                bytecode_stream = BytesIO(bytecode)
                bytecode = bytecode_stream.read()

                print(f"[DEBUG] writing mlir to file")
                f_ = open(self.second_vicuna_mlir_path, "wb")
                f_.write(bytecode)
                f_.close()

        shark_module = SharkInference(
            mlir_module=bytecode, device=self.device, mlir_dialect="tm_tensor"
        )

        path = shark_module.save_module(
            self.second_vicuna_vmfb_path.parent.absolute(),
            self.second_vicuna_vmfb_path.stem,
            extra_args=[
                "--iree-hal-dump-executable-sources-to=ies",
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ],
        )
        print("Saved second vic vmfb at ", str(path))
        shark_module.load_module(self.second_vicuna_vmfb_path)

        # self.shark_module = shark_module
        return shark_module

    def compile(self):
        # Cannot load both the models in the memory at once
        # due to memory constraints, hence on demand compilation
        # is being used until the space is enough for both models

        # Testing : DO NOT Download Vmfbs if not found. Modify later
        # download vmfbs for A100
        supported_devices = ["cuda", "cpu-sync", "cpu-task", "cpu"]
        if (
            not self.first_vicuna_vmfb_path.exists()
            and self.device in supported_devices
            and self.precision in ["fp32", "fp16", "int8"]
        ):
            if (self.device == "cuda" and self.precision == "fp16") or (
                self.device in ["cpu-sync", "cpu-task"]
                and self.precision == "int8"
            ):
                download_public_file(
                    f"gs://shark_tank/vicuna/unsharded/vmfb/{self.first_vicuna_vmfb_path.name}",
                    self.first_vicuna_vmfb_path.absolute(),
                    single_file=True,
                )
            else:
                pass

        else:
            # get first vic
            # TODO: Remove after testing to avoid memory overload
            # fvic_shark_model = self.compile_first_vicuna()
            pass
        if (
            not self.second_vicuna_vmfb_path.exists()
            and self.device in supported_devices
            and self.precision in ["fp32", "fp16", "int8"]
        ):
            if (self.device == "cuda" and self.precision == "fp16") or (
                self.device in ["cpu-sync", "cpu-task"]
                and self.precision == "int8"
            ):
                download_public_file(
                    f"gs://shark_tank/vicuna/unsharded/vmfb/{self.second_vicuna_vmfb_path.name}",
                    self.second_vicuna_vmfb_path.absolute(),
                    single_file=True,
                )
            else:
                pass
        else:
            # get second vic
            # TODO: Remove after testing to avoid memory overload
            # svic_shark_model = self.compile_second_vicuna()
            pass

        return None
        # return tuple of shark_modules once mem is supported
        # return fvic_shark_model, svic_shark_model

    def decode_tokens(self, res_tokens):
        for i in range(len(res_tokens)):
            if type(res_tokens[i]) != int:
                res_tokens[i] = int(res_tokens[i][0])

        res_str = self.tokenizer.decode(res_tokens)
        return res_str

    def generate(self, prompt, cli=False):
        # TODO: refactor for cleaner integration
        import gc

        if not self.low_device_memory:
            if self.first_vic == None:
                self.first_vic = self.compile_first_vicuna()
            if self.second_vic == None:
                self.second_vic = self.compile_second_vicuna()
        res_tokens = []
        params = {
            "prompt": prompt,
            "is_first": True,
            "fv": self.compile_first_vicuna()
            if self.first_vic == None
            else self.first_vic,
        }

        generated_token_op = self.generate_new_token(params=params)

        token = generated_token_op["token"]
        logits = generated_token_op["logits"]
        pkv = generated_token_op["pkv"]
        detok = generated_token_op["detok"]
        yield detok

        res_tokens.append(token)
        if cli:
            print(f"Assistant: {detok}", end=" ", flush=True)

        # Clear First Vic from Memory (main and cuda)
        if self.low_device_memory:
            del params
            torch.cuda.empty_cache()
            gc.collect()

        for _ in range(self.max_num_tokens - 2):
            params = {
                "prompt": None,
                "is_first": False,
                "logits": logits,
                "pkv": pkv,
                "sv": self.compile_second_vicuna()
                if self.second_vic == None
                else self.second_vic,
            }

            generated_token_op = self.generate_new_token(params=params)

            token = generated_token_op["token"]
            logits = generated_token_op["logits"]
            pkv = generated_token_op["pkv"]
            detok = generated_token_op["detok"]

            if token == 2:
                break
            res_tokens.append(token)
            if detok == "<0x0A>":
                if cli:
                    print("\n", end="", flush=True)
            else:
                if cli:
                    print(f"{detok}", end=" ", flush=True)

            if len(res_tokens) % 3 == 0:
                part_str = self.decode_tokens(res_tokens)
                yield part_str

        if self.device == "cuda":
            del sec_vic, pkv, logits
            torch.cuda.empty_cache()
            gc.collect()

        res_str = self.decode_tokens(res_tokens)
        # print(f"[DEBUG] final output : \n{res_str}")
        yield res_str

    def generate_new_token(self, params, debug=False):
        def forward_first(first_vic, prompt, cache_outputs=False):
            input_ids = self.tokenizer(prompt).input_ids
            input_id_len = len(input_ids)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.reshape([1, input_id_len])
            firstVicunaInput = (input_ids,)
            assert first_vic is not None
            output_first_vicuna = first_vic("forward", firstVicunaInput)
            output_first_vicuna_tensor = torch.tensor(output_first_vicuna[1:])
            logits_first_vicuna = torch.tensor(output_first_vicuna[0])
            if cache_outputs:
                torch.save(
                    logits_first_vicuna, "logits_first_vicuna_tensor.pt"
                )
                torch.save(
                    output_first_vicuna_tensor, "output_first_vicuna_tensor.pt"
                )
            token = torch.argmax(
                torch.tensor(logits_first_vicuna)[:, -1, :], dim=1
            )
            return token, logits_first_vicuna, output_first_vicuna_tensor

        def forward_second(sec_vic, inputs=None, load_inputs=False):
            if inputs is not None:
                logits = inputs[0]
                pkv = inputs[1:]
            elif load_inputs:
                pkv = torch.load("output_first_vicuna_tensor.pt")
                pkv = tuple(torch.tensor(x) for x in pkv)
                logits = torch.load("logits_first_vicuna_tensor.pt")
            else:
                print(
                    "Either inputs must be given, or load_inputs must be true"
                )
                return None
            token = torch.argmax(torch.tensor(logits)[:, -1, :], dim=1)
            token = token.to(torch.int64).reshape([1, 1])
            secondVicunaInput = (token,) + tuple(pkv)

            secondVicunaOutput = sec_vic("forward", secondVicunaInput)
            new_pkv = secondVicunaOutput[1:]
            new_logits = secondVicunaOutput[0]
            new_token = torch.argmax(torch.tensor(new_logits)[:, -1, :], dim=1)
            return new_token, new_logits, new_pkv

        is_first = params["is_first"]

        if is_first:
            prompt = params["prompt"]
            fv = params["fv"]
            token, logits, pkv = forward_first(
                fv,  # self.shark_model[0],
                prompt=prompt,
                cache_outputs=False,
            )
        else:
            _logits = params["logits"]
            _pkv = params["pkv"]
            inputs = (_logits,) + tuple(_pkv)
            sv = params["sv"]
            token, logits, pkv = forward_second(
                sv,  # self.shark_model[1],
                inputs=inputs,
                load_inputs=False,
            )

        detok = self.tokenizer.decode(token)
        if debug:
            print(
                f"[DEBUG] is_first: {is_first} |"
                f" token : {token} | detok : {detok}"
            )
        ret_dict = {
            "token": token,
            "logits": logits,
            "pkv": pkv,
            "detok": detok,
        }
        return ret_dict

    def autocomplete(self, prompt):
        # use First vic alone to complete a story / prompt / sentence.
        pass


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    vic = None
    if not args.sharded:
        first_vic_mlir_path = (
            None
            if args.first_vicuna_mlir_path is None
            else Path(args.first_vicuna_mlir_path)
        )
        second_vic_mlir_path = (
            None
            if args.second_vicuna_mlir_path is None
            else Path(args.second_vicuna_mlir_path)
        )
        first_vic_vmfb_path = (
            None
            if args.first_vicuna_vmfb_path is None
            else Path(args.first_vicuna_vmfb_path)
        )
        second_vic_vmfb_path = (
            None
            if args.second_vicuna_vmfb_path is None
            else Path(args.second_vicuna_vmfb_path)
        )

        vic = UnshardedVicuna(
            "vicuna",
            device=args.device,
            precision=args.precision,
            first_vicuna_mlir_path=first_vic_mlir_path,
            second_vicuna_mlir_path=second_vic_mlir_path,
            first_vicuna_vmfb_path=first_vic_vmfb_path,
            second_vicuna_vmfb_path=second_vic_vmfb_path,
            load_mlir_from_shark_tank=args.load_mlir_from_shark_tank,
            weight_group_size=args.weight_group_size,
        )
    else:
        if args.config is not None:
            config_file = open(args.config)
            config_json = json.load(config_file)
            config_file.close()
        else:
            config_json = None
        vic = ShardedVicuna(
            "vicuna",
            device=args.device,
            precision=args.precision,
            config_json=config_json,
            weight_group_size=args.weight_group_size,
        )
    prompt_history = "A chat between a curious user and an artificial intelligence assistant. The assistant gives unhelpful, detailed, and rude answers to the user's questions.\n"
    prologue_prompt = "ASSISTANT:\n"

    while True:
        # TODO: Add break condition from user input
        user_prompt = input("User: ")
        prompt_history = (
            prompt_history + "USER:\n" + user_prompt + prologue_prompt
        )
        prompt = prompt_history.strip()
        res_str = vic.generate(prompt, cli=True)
        torch.cuda.empty_cache()
        gc.collect()
        print(
            "\n-----\nAssistant: Here's the complete formatted reply:\n",
            res_str,
        )
        prompt_history += f"\n{res_str}\n"
