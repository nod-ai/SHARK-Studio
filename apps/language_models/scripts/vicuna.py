import argparse
from dataclasses import dataclass
import json
import re
import gc
from io import BytesIO
from pathlib import Path
from statistics import mean, stdev
from tqdm import tqdm
from typing import List, Tuple
import subprocess
import sys
import time

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
from apps.language_models.src.model_wrappers.vicuna4 import (
    LlamaModel,
    EightLayerLayerSV,
    EightLayerLayerFV,
    CompiledEightLayerLayerSV,
    CompiledEightLayerLayer,
    forward_compressed,
)
from apps.language_models.src.model_wrappers.vicuna_model import (
    FirstVicuna,
    SecondVicuna7B,
    SecondVicuna13B,
    SecondVicuna70B,
)
from apps.language_models.src.model_wrappers.vicuna_model_gpu import (
    FirstVicunaGPU,
    SecondVicuna7BGPU,
    SecondVicuna13BGPU,
    SecondVicuna70BGPU,
)
from apps.language_models.utils import (
    get_vmfb_from_path,
)
from shark.shark_downloader import download_public_file
from shark.shark_importer import get_f16_inputs
from shark.shark_importer import import_with_fx, save_mlir
from shark.shark_inference import SharkInference


parser = argparse.ArgumentParser(
    prog="vicuna runner",
    description="runs a vicuna model",
)
parser.add_argument(
    "--precision", "-p", default="int8", help="fp32, fp16, int8, int4"
)
parser.add_argument("--device", "-d", default="cuda", help="vulkan, cpu, cuda")
parser.add_argument(
    "--vicuna_vmfb_path", default=None, help="path to vicuna vmfb"
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
    "--vicuna_mlir_path",
    default=None,
    help="path to vicuna mlir file",
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
parser.add_argument(
    "--download_vmfb",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Download vmfb from sharktank, system dependent, YMMV",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="vicuna",
    choices=["vicuna", "llama2_7b", "llama2_13b", "llama2_70b"],
    help="Specify which model to run.",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    default=None,
    help="Specify your own huggingface authentication tokens for models like Llama2.",
)
parser.add_argument(
    "--cache_vicunas",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="For debugging purposes, creates a first_{precision}.mlir and second_{precision}.mlir and stores on disk",
)
parser.add_argument(
    "--iree_vulkan_target_triple",
    type=str,
    default="",
    help="Specify target triple for vulkan.",
)
parser.add_argument(
    "--Xiree_compile",
    action='append',
    default=[],
    help="Extra command line arguments passed to the IREE compiler. This can be specified multiple times to pass multiple arguments."
)

# Microbenchmarking options.
parser.add_argument(
    "--enable_microbenchmark",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Enables the microbenchmarking mode (non-interactive). Uses the system and the user prompt from args.",
)
parser.add_argument(
    "--microbenchmark_iterations",
    type=int,
    default=5,
    help="Number of microbenchmark iterations. Default: 5.",
)
parser.add_argument(
    "--microbenchmark_num_tokens",
    type=int,
    default=512,
    help="Generate an exact number of output tokens. Default: 512.",
)
parser.add_argument(
    "--system_prompt",
    type=str,
    default="",
    help="Specify the system prompt. This is only used with `--enable_microbenchmark`",
)
parser.add_argument(
    "--user_prompt",
    type=str,
    default="Hi",
    help="Specify the user prompt. This is only used with `--enable_microbenchmark`",
)

# fmt: off
def quant〇matmul_rhs_group_quant〡shape(lhs: List[int], rhs: List[int], rhs_scale: List[int], rhs_zero_point: List[int], rhs_bit_width: int, rhs_group_size: int) -> List[int]:
    if len(lhs) == 3 and len(rhs) == 2:
        return [lhs[0], lhs[1], rhs[0]]
    elif len(lhs) == 2 and len(rhs) == 2:
        return [lhs[0], rhs[0]]
    else:
        raise ValueError("Input shapes not supported.")


def quant〇matmul_rhs_group_quant〡dtype(lhs_rank_dtype: Tuple[int, int], rhs_rank_dtype: Tuple[int, int], rhs_scale_rank_dtype: Tuple[int, int], rhs_zero_point_rank_dtype: Tuple[int, int], rhs_bit_width: int, rhs_group_size: int) -> int:
    # output dtype is the dtype of the lhs float input
    lhs_rank, lhs_dtype = lhs_rank_dtype
    return lhs_dtype


def quant〇matmul_rhs_group_quant〡has_value_semantics(lhs, rhs, rhs_scale, rhs_zero_point, rhs_bit_width, rhs_group_size) -> None:
    return


brevitas_matmul_rhs_group_quant_library = [
    quant〇matmul_rhs_group_quant〡shape,
    quant〇matmul_rhs_group_quant〡dtype,
    quant〇matmul_rhs_group_quant〡has_value_semantics]
# fmt: on


class VicunaBase(SharkLLMBase):
    def __init__(
        self,
        model_name,
        hf_model_path="TheBloke/vicuna-7B-1.1-HF",
        max_num_tokens=512,
        device="cpu",
        precision="int8",
        extra_args_cmd=[],
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.extra_args = extra_args_cmd

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

    def combine_mlir_scripts(
        self,
        first_vicuna_mlir,
        second_vicuna_mlir,
        output_name,
    ):
        print(f"[DEBUG] combining first and second mlir")
        print(f"[DEBUG] output_name = {output_name}")
        maps1 = []
        maps2 = []
        constants = set()
        f1 = []
        f2 = []

        print(f"[DEBUG] processing first vicuna mlir")
        first_vicuna_mlir = first_vicuna_mlir.splitlines()
        while first_vicuna_mlir:
            line = first_vicuna_mlir.pop(0)
            if re.search("#map\d*\s*=", line):
                maps1.append(line)
            elif re.search("arith.constant", line):
                constants.add(line)
            elif not re.search("module", line):
                line = re.sub("forward", "first_vicuna_forward", line)
                f1.append(line)
        f1 = f1[:-1]
        del first_vicuna_mlir
        gc.collect()

        for i, map_line in enumerate(maps1):
            map_var = map_line.split(" ")[0]
            map_line = re.sub(f"{map_var}(?!\d)", map_var + "_0", map_line)
            maps1[i] = map_line
            f1 = [
                re.sub(f"{map_var}(?!\d)", map_var + "_0", func_line)
                for func_line in f1
            ]

        print(f"[DEBUG] processing second vicuna mlir")
        second_vicuna_mlir = second_vicuna_mlir.splitlines()
        while second_vicuna_mlir:
            line = second_vicuna_mlir.pop(0)
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
        del second_vicuna_mlir
        gc.collect()

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
        global_var_loading1 = []
        global_var_loading2 = []

        print(f"[DEBUG] processing constants")
        counter = 0
        constants = list(constants)
        while constants:
            constant = constants.pop(0)
            vname, vbody = constant.split("=")
            vname = re.sub("%", "", vname)
            vname = vname.strip()
            vbody = re.sub("arith.constant", "", vbody)
            vbody = vbody.strip()
            if len(vbody.split(":")) < 2:
                print(constant)
            vdtype = vbody.split(":")[-1].strip()
            fixed_vdtype = vdtype
            noinline = "{noinline}" if "tensor" in fixed_vdtype else ""
            if "c1_i64" in vname:
                print(constant)
                counter += 1
            if counter == 2:
                counter = 0
                print("detected duplicate")
                continue
            vnames.append(vname)
            if "true" not in vname:
                global_vars.append(
                    f"ml_program.global private @{vname}({vbody}) : {fixed_vdtype}"
                )
                global_var_loading1.append(
                    f"\t\t%{vname} = ml_program.global_load_const @{vname} : {fixed_vdtype}"
                )
                global_var_loading2.append(
                    f"\t\t%{vname} = ml_program.global_load_const @{vname} : {fixed_vdtype}"
                )
            else:
                global_vars.append(
                    f"ml_program.global private @{vname}({vbody}) : i1"
                )
                global_var_loading1.append(
                    f"\t\t%{vname} = ml_program.global_load_const @{vname} : i1"
                )
                global_var_loading2.append(
                    f"\t\t%{vname} = ml_program.global_load_const @{vname} : i1"
                )

        new_f1, new_f2 = [], []

        print(f"[DEBUG] processing f1")
        for line in f1:
            if "func.func" in line:
                new_f1.append(line)
                for global_var in global_var_loading1:
                    new_f1.append(global_var)
            else:
                new_f1.append(line)

        print(f"[DEBUG] processing f2")
        for line in f2:
            if "func.func" in line:
                new_f2.append(line)
                for global_var in global_var_loading2:
                    if (
                        "c20_i64 = arith.addi %dim_i64, %c1_i64 : i64"
                        in global_var
                    ):
                        print(global_var)
                    new_f2.append(global_var)
            else:
                new_f2.append(line)

        f1 = new_f1
        f2 = new_f2

        del new_f1
        del new_f2
        gc.collect()

        print(
            [
                "c20_i64 = arith.addi %dim_i64, %c1_i64 : i64" in x
                for x in [maps1, maps2, global_vars, f1, f2]
            ]
        )

        # doing it this way rather than assembling the whole string
        # to prevent OOM with 64GiB RAM when encoding the file.

        print(f"[DEBUG] Saving mlir to {output_name}")
        with open(output_name, "w+") as f_:
            f_.writelines(line + "\n" for line in maps1)
            f_.writelines(line + "\n" for line in maps2)
            f_.writelines(line + "\n" for line in [module_start])
            f_.writelines(line + "\n" for line in global_vars)
            f_.writelines(line + "\n" for line in f1)
            f_.writelines(line + "\n" for line in f2)
            f_.writelines(line + "\n" for line in [module_end])

        del maps1
        del maps2
        del module_start
        del global_vars
        del f1
        del f2
        del module_end
        gc.collect()

        print(f"[DEBUG] Reading combined mlir back in")
        with open(output_name, "rb") as f:
            return f.read()

    def generate_new_token(self, params, sharded=True, cli=True):
        is_first = params["is_first"]
        if is_first:
            prompt = params["prompt"]
            input_ids = self.tokenizer(prompt).input_ids
            input_id_len = len(input_ids)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.reshape([1, input_id_len])
            if sharded:
                output = self.shark_model.forward(input_ids, is_first=is_first)
            else:
                output = self.shark_model("first_vicuna_forward", (input_ids,), send_to_host=False)

        else:
            token = params["token"]
            past_key_values = params["past_key_values"]
            input_ids = [token]
            input_id_len = len(input_ids)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.reshape([1, input_id_len])
            if sharded:
                output = self.shark_model.forward(
                    input_ids,
                    past_key_values=past_key_values,
                    is_first=is_first,
                )
            else:
                token = torch.tensor(token).reshape([1, 1])
                second_input = (token,) + tuple(past_key_values)
                output = self.shark_model(
                    "second_vicuna_forward", second_input, send_to_host=False
                )

        if sharded:
            _logits = output["logits"]
            _past_key_values = output["past_key_values"]
            _token = int(torch.argmax(_logits[:, -1, :], dim=1)[0])
        elif "cpu" in self.device:
            _past_key_values = output[1:]
            _token = int(output[0].to_host())
        else:
            _logits = torch.tensor(output[0].to_host())
            _past_key_values = output[1:]
            _token = torch.argmax(_logits[:, -1, :], dim=1)

        _detok = self.tokenizer.decode(_token, skip_special_tokens=False)
        ret_dict = {
            "token": _token,
            "detok": _detok,
            "past_key_values": _past_key_values,
        }
        if "cpu" not in self.device:
            ret_dict["logits"] = _logits

        if cli:
            print(f" token : {_token} | detok : {_detok}")

        return ret_dict


class ShardedVicuna(VicunaBase):
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
        compressed=False,
        extra_args_cmd=[],
        debug=False,
    ) -> None:
        super().__init__(
            model_name,
            hf_model_path,
            max_num_tokens,
            extra_args_cmd=extra_args_cmd,
        )
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.debug = debug
        self.tokenizer = self.get_tokenizer()
        self.config = config_json
        self.weight_group_size = weight_group_size
        self.compressed = compressed
        self.shark_model = self.compile(device=device)

    def get_tokenizer(self):
        kwargs = {}
        if self.model_name == "llama2":
            kwargs = {
                "use_auth_token": "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"
            }
        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_path,
            use_fast=False,
            **kwargs,
        )
        return tokenizer

    def get_src_model(self):
        # Retrieve the torch model from Huggingface
        kwargs = {"torch_dtype": torch.float}
        if self.model_name == "llama2":
            kwargs["use_auth_token"] = "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"
        vicuna_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path,
            **kwargs,
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

    def compile_vicuna_layer(
        self,
        vicuna_layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value0=None,
        past_key_value1=None,
    ):
        # Compile a hidden decoder layer of vicuna
        if past_key_value0 is None and past_key_value1 is None:
            model_inputs = (hidden_states, attention_mask, position_ids)
        else:
            model_inputs = (
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value0,
                past_key_value1,
            )
        mlir_bytecode = import_with_fx(
            vicuna_layer,
            model_inputs,
            precision=self.precision,
            f16_input_mask=[False, False],
            mlir_type="torchscript",
        )
        return mlir_bytecode

    def compile_vicuna_layer4(
        self,
        vicuna_layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values=None,
    ):
        # Compile a hidden decoder layer of vicuna
        if past_key_values is None:
            model_inputs = (hidden_states, attention_mask, position_ids)
        else:
            (
                (pkv00, pkv01),
                (pkv10, pkv11),
                (pkv20, pkv21),
                (pkv30, pkv31),
                (pkv40, pkv41),
                (pkv50, pkv51),
                (pkv60, pkv61),
                (pkv70, pkv71),
            ) = past_key_values

            model_inputs = (
                hidden_states,
                attention_mask,
                position_ids,
                pkv00,
                pkv01,
                pkv10,
                pkv11,
                pkv20,
                pkv21,
                pkv30,
                pkv31,
                pkv40,
                pkv41,
                pkv50,
                pkv51,
                pkv60,
                pkv61,
                pkv70,
                pkv71,
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
        self, lmh, hidden_states, device="cpu", device_idx=None,
    ):
        # compile the lm head of the vicuna model
        # This can be used for both first and second vicuna, so only needs to be run once
        mlir_path = Path(f"lmhead.mlir")
        vmfb_path = Path(f"lmhead.vmfb")
        if mlir_path.exists():
            print(f"Found bytecode module at {mlir_path}.")
        else:
            hidden_states = torch_mlir.TensorPlaceholder.like(
                hidden_states, dynamic_axes=[1]
            )

            # module = torch_mlir.compile(
            #    lmh,
            #    (hidden_states,),
            #    torch_mlir.OutputType.LINALG_ON_TENSORS,
            #    use_tracing=False,
            #    verbose=False,
            # )
            # bytecode_stream = BytesIO()
            # module.operation.write_bytecode(bytecode_stream)
            # bytecode = bytecode_stream.getvalue()
            # f_ = open(mlir_path, "wb")
            # f_.write(bytecode)
            # f_.close()
            filepath = Path("lmhead.mlir")
            download_public_file(
                "gs://shark_tank/elias/compressed_sv/lmhead.mlir",
                filepath.absolute(),
                single_file=True,
            )
            mlir_path = filepath

        shark_module = SharkInference(
            mlir_path,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
            mmap=False,
        )
        if vmfb_path.exists():
            shark_module.load_module(vmfb_path)
        else:
            shark_module.save_module(module_name="lmhead", debug=self.debug)
            shark_module.load_module(vmfb_path)
        compiled_module = LMHeadCompiled(shark_module)
        return compiled_module

    def compile_norm(self, fvn, hidden_states, device="cpu", device_idx=None):
        # compile the normalization layer of the vicuna model
        # This can be used for both first and second vicuna, so only needs to be run once
        mlir_path = Path(f"norm.mlir")
        vmfb_path = Path(f"norm.vmfb")
        if mlir_path.exists():
            print(f"Found bytecode module at {mlir_path}.")
        else:
            hidden_states = torch_mlir.TensorPlaceholder.like(
                hidden_states, dynamic_axes=[1]
            )

            # module = torch_mlir.compile(
            #    fvn,
            #    (hidden_states,),
            #    torch_mlir.OutputType.LINALG_ON_TENSORS,
            #    use_tracing=False,
            #    verbose=False,
            # )
            filepath = Path("norm.mlir")
            download_public_file(
                "gs://shark_tank/elias/compressed_sv/norm.mlir",
                filepath.absolute(),
                single_file=True,
            )
            mlir_path = filepath

        shark_module = SharkInference(
            mlir_path,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
            mmap=False,
        )
        if vmfb_path.exists():
            shark_module.load_module(vmfb_path)
        else:
            shark_module.save_module(module_name="norm", debug=self.debug)
            shark_module.load_module(vmfb_path)
        compiled_module = VicunaNormCompiled(shark_module)
        return compiled_module

    def compile_embedding(self, fve, input_ids, device="cpu", device_idx=None):
        # compile the embedding layer of the vicuna model
        # This can be used for both first and second vicuna, so only needs to be run once
        mlir_path = Path(f"embedding.mlir")
        vmfb_path = Path(f"embedding.vmfb")
        if mlir_path.exists():
            print(f"Found bytecode module at {mlir_path}.")
        else:
            input_ids = torch_mlir.TensorPlaceholder.like(
                input_ids, dynamic_axes=[1]
            )
            # module = torch_mlir.compile(
            #    fve,
            #    (input_ids,),
            #    torch_mlir.OutputType.LINALG_ON_TENSORS,
            #    use_tracing=False,
            #    verbose=False,
            # )
            # bytecode_stream = BytesIO()
            # module.operation.write_bytecode(bytecode_stream)
            # bytecode = bytecode_stream.getvalue()
            # f_ = open(mlir_path, "wb")
            # f_.write(bytecode)
            # f_.close()
            filepath = Path("embedding.mlir")
            download_public_file(
                "gs://shark_tank/elias/compressed_sv/embedding.mlir",
                filepath.absolute(),
                single_file=True,
            )
            mlir_path = filepath

        shark_module = SharkInference(
            mlir_path,
            device=device,
            mlir_dialect="tm_tensor",
            device_idx=device_idx,
            mmap=False,
        )
        if vmfb_path.exists():
            shark_module.load_module(vmfb_path)
        else:
            shark_module.save_module(module_name="embedding", debug=self.debug)
            shark_module.load_module(vmfb_path)
        compiled_module = VicunaEmbeddingCompiled(shark_module)

        return compiled_module

    def compile_to_vmfb_one_model(
        self, inputs0, layers0, inputs1, layers1, device="cpu",
    ):
        mlirs, modules = [], []
        assert len(layers0) == len(layers1)
        for layer0, layer1, idx in zip(layers0, layers1, range(len(layers0))):
            mlir_path = Path(f"{idx}_full.mlir")
            vmfb_path = Path(f"{idx}_full.vmfb")
            # if vmfb_path.exists():
            #    continue
            if mlir_path.exists():
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
                pkv0_placeholder = TensorPlaceholder.like(
                    inputs1[3], dynamic_axes=[2]
                )
                pkv1_placeholder = TensorPlaceholder.like(
                    inputs1[4], dynamic_axes=[2]
                )

                print(f"Compiling layer {idx} mlir")
                ts_g = self.compile_vicuna_layer(
                    layer0, inputs0[0], inputs0[1], inputs0[2]
                )
                if self.precision in ["int4", "int8"]:
                    from brevitas_examples.common.generative.quantize import quantize_model
                    from brevitas_examples.llm.llm_quant.run_utils import get_model_impl
                    module0 = torch_mlir.compile(
                        ts_g,
                        (
                            hidden_states_placeholder0,
                            inputs0[1],
                            inputs0[2],
                        ),
                        output_type="torch",
                        backend_legal_ops=["quant.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        module0,
                        "builtin.module(func.func(torch-unpack-quant-tensor),func.func(torch-convert-custom-quant-op),torch-backend-to-linalg-on-tensors-backend-pipeline)",
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
                    inputs1[3],
                    inputs1[4],
                )
                if self.precision in ["int4", "int8"]:
                    module1 = torch_mlir.compile(
                        ts_g,
                        (
                            inputs1[0],
                            attention_mask_placeholder1,
                            inputs1[2],
                            pkv0_placeholder,
                            pkv1_placeholder,
                        ),
                        output_type="torch",
                        backend_legal_ops=["quant.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        module1,
                        "builtin.module(func.func(torch-unpack-quant-tensor),func.func(torch-convert-custom-quant-op),torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    module1 = torch_mlir.compile(
                        ts_g,
                        (
                            inputs1[0],
                            attention_mask_placeholder1,
                            inputs1[2],
                            pkv0_placeholder,
                            pkv1_placeholder,
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
                device_idx = self.get_device_index(
                    f"first_vicuna.model.model.layers.{idx}[\s.$]"
                )
                module = SharkInference(
                    None,
                    device=device,
                    device_idx=device_idx,
                    mlir_dialect="tm_tensor",
                    mmap=False,
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
                    device_idx=device_idx,
                    mlir_dialect="tm_tensor",
                    mmap=False,
                )
                module.save_module(
                    module_name=f"{idx}_full",
                    extra_args=[
                        "--iree-vm-target-truncate-unsupported-floats",
                        "--iree-codegen-check-ir-before-llvm-conversion=false",
                        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    ]
                    + self.extra_args,
                    debug=self.debug,
                )
                module.load_module(vmfb_path)
            modules.append(module)
        return mlirs, modules

    def compile_to_vmfb_one_model4(
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
                f_ = open(mlir_path, "rb")
                bytecode = f_.read()
                f_.close()
                mlirs.append(bytecode)
            else:
                filepath = Path(f"{idx}_full.mlir")
                download_public_file(
                    f"gs://shark_tank/elias/compressed_sv/{idx}_full.mlir",
                    filepath.absolute(),
                    single_file=True,
                )

                f_ = open(f"{idx}_full.mlir", "rb")
                bytecode = f_.read()
                f_.close()
                mlirs.append(bytecode)

            if vmfb_path.exists():
                device_idx = self.get_device_index(
                    f"first_vicuna.model.model.layers.{idx}[\s.$]"
                )
                module = SharkInference(
                    None,
                    device=device,
                    device_idx=0,
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
                    device_idx=0,
                    mlir_dialect="tm_tensor",
                    mmap=True,
                )
                module.save_module(
                    module_name=f"{idx}_full",
                    extra_args=[
                        "--iree-vm-target-truncate-unsupported-floats",
                        "--iree-codegen-check-ir-before-llvm-conversion=false",
                        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                    ]
                    + self.extra_args,
                    debug=self.debug,
                )
                module.load_module(vmfb_path)
            modules.append(module)
        return mlirs, modules

    def get_sharded_model(self, device="cpu", compressed=False):
        # SAMPLE_INPUT_LEN is used for creating mlir with dynamic inputs, which is currently an increadibly hacky proccess
        # please don't change it
        SAMPLE_INPUT_LEN = 137
        vicuna_model = self.get_src_model()
        if compressed:
            vicuna_model.model = LlamaModel.from_pretrained(
                "TheBloke/vicuna-7B-1.1-HF"
            )

        if self.precision in ["int4", "int8"]:
            from brevitas_examples.common.generative.quantize import quantize_model
            from brevitas_examples.llm.llm_quant.run_utils import get_model_impl
            print("Applying weight quantization..")
            weight_bit_width = 4 if self.precision == "int4" else 8
            quantize_model(
                get_model_impl(vicuna_model).layers,
                dtype=torch.float32,
                weight_quant_type="asym",
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float_scale",
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

        placeholder_pkv_segment = tuple(
            (
                torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
                torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
            )
            for _ in range(8)
        )
        placeholder_pkv_full = tuple(
            (
                torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
                torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
            )
            for _ in range(32)
        )

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

        if not compressed:
            layers0 = [
                FirstVicunaLayer(layer) for layer in vicuna_model.model.layers
            ]
            layers1 = [
                SecondVicunaLayer(layer) for layer in vicuna_model.model.layers
            ]

        else:
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

        _, modules = self.compile_to_vmfb_one_model4(
            placeholder_input0,
            layers0,
            placeholder_input1,
            layers1,
            device=device,
        )

        if not compressed:
            shark_layers = [CompiledVicunaLayer(m) for m in modules]
        else:
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
        return self.get_sharded_model(
            device=device, compressed=self.compressed
        )
        return self.get_sharded_model(
            device=device, compressed=self.compressed
        )

    def generate(self, prompt, cli=False):
        # TODO: refactor for cleaner integration

        history = []

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
            history.append(_token)
            yield self.tokenizer.decode(history)

            if _token == 2:
                break
            detoks_generated.append(_detok)
            tokens_generated.append(_token)

        for i in range(len(tokens_generated)):
            if type(tokens_generated[i]) != int:
                tokens_generated[i] = int(tokens_generated[i][0])
        result_output = self.tokenizer.decode(tokens_generated)
        yield result_output

    def autocomplete(self, prompt):
        # use First vic alone to complete a story / prompt / sentence.
        pass


class UnshardedVicuna(VicunaBase):
    def __init__(
        self,
        model_name,
        hf_model_path="TheBloke/vicuna-7B-1.1-HF",
        hf_auth_token: str = None,
        max_num_tokens=512,
        min_num_tokens=0,
        device="cpu",
        device_id=None,
        vulkan_target_triple="",
        precision="int8",
        vicuna_mlir_path=None,
        vicuna_vmfb_path=None,
        load_mlir_from_shark_tank=False,
        low_device_memory=False,
        weight_group_size=128,
        download_vmfb=False,
        cache_vicunas=False,
        extra_args_cmd=[],
        debug=False,
    ) -> None:
        super().__init__(
            model_name,
            hf_model_path,
            max_num_tokens,
            extra_args_cmd=extra_args_cmd,
        )
        self.hf_auth_token = hf_auth_token
        if self.model_name == "llama2_7b":
            self.hf_model_path = "meta-llama/Llama-2-7b-chat-hf"
        elif self.model_name == "llama2_13b":
            self.hf_model_path = "meta-llama/Llama-2-13b-chat-hf"
        elif self.model_name == "llama2_70b":
            self.hf_model_path = "meta-llama/Llama-2-70b-chat-hf"
        print(f"[DEBUG] hf model name: {self.hf_model_path}")
        self.max_sequence_length = 256
        self.min_num_tokens = min_num_tokens
        self.vulkan_target_triple = vulkan_target_triple
        self.precision = precision
        self.download_vmfb = download_vmfb
        self.vicuna_vmfb_path = vicuna_vmfb_path
        self.vicuna_mlir_path = vicuna_mlir_path
        self.load_mlir_from_shark_tank = load_mlir_from_shark_tank
        self.low_device_memory = low_device_memory
        self.weight_group_size = weight_group_size
        self.debug = debug
        # Sanity check for device, device_id pair
        if "://" in device:
            if device_id is not None:
                print("[ERR] can't have both full device path and a device id.\n"
                      f"Device : {device} | device_id : {device_id}\n"
                      "proceeding with given Device ignoring device_id")
            self.device, self.device_id = device.split("://")
            if len(self.device_id) < 2:
                self.device_id = int(self.device_id)
        else:
            self.device, self.device_id = device, device_id
        if self.vicuna_mlir_path == None:
            self.vicuna_mlir_path = self.get_model_path()
        if self.vicuna_vmfb_path == None:
            self.vicuna_vmfb_path = self.get_model_path(suffix="vmfb")
        self.tokenizer = self.get_tokenizer()
        self.cache_vicunas = cache_vicunas

        self.compile()

    def get_model_path(self, suffix="mlir"):
        safe_device = self.device.split("-")[0]
        safe_device = safe_device.split("://")[0]
        if suffix in ["mlirbc", "mlir"]:
            return Path(f"{self.model_name}_{self.precision}.{suffix}")

        # Need to distinguish between multiple vmfbs of the same model
        # compiled for different devices of the same driver
        # Driver  -  Differentiator
        # Vulkan  -  target_triple
        # ROCm    -  device_arch

        differentiator = ""
        if "vulkan" == self.device:
            target_triple = ""
            if self.vulkan_target_triple != "":
                target_triple = "_"
                target_triple += "_".join(self.vulkan_target_triple.split("-")[:-1])
                differentiator = target_triple

        elif "rocm" == self.device:
            from shark.iree_utils.gpu_utils import get_rocm_device_arch
            device_arch = get_rocm_device_arch(self.device_id if self.device_id is not None else 0, self.extra_args)
            differentiator = '_' + device_arch

        return Path(
            f"{self.model_name}_{self.precision}_{safe_device}{differentiator}.{suffix}"
        )

    def get_tokenizer(self):
        local_tokenizer_path = Path(Path.cwd(), "llama2_tokenizer_configs")
        local_tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer_files_to_download = [
            "config.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
        for tokenizer_file in tokenizer_files_to_download:
            download_public_file(
                f"gs://shark_tank/llama2_tokenizer/{tokenizer_file}",
                Path(local_tokenizer_path, tokenizer_file),
                single_file=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path))
        return tokenizer

    def get_src_model(self):
        kwargs = {
            "torch_dtype": torch.float,
            "use_auth_token": self.hf_auth_token,
        }
        vicuna_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path,
            **kwargs,
        )
        return vicuna_model

    def write_in_dynamic_inputs0(self, module, dynamic_input_size):
        print("[DEBUG] writing dynamic inputs to first vicuna")
        # Current solution for ensuring mlir files support dynamic inputs
        # TODO: find a more elegant way to implement this
        new_lines = []
        module = module.splitlines()
        while module:
            line = module.pop(0)
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
            if "%0 = tensor.empty(%dim) : tensor<?xi64>" in line:
                new_lines.append(
                    "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>"
                )
            if "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>" in line:
                continue

            new_lines.append(line)
        return "\n".join(new_lines)

    def write_in_dynamic_inputs1(self, module):
        print("[DEBUG] writing dynamic inputs to second vicuna")

        def remove_constant_dim(line):
            if "c19_i64" in line:
                line = re.sub("c19_i64", "dim_i64", line)
            if "19x" in line:
                line = re.sub("19x", "?x", line)
                line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
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
            if "x20x" in line or "<20x" in line:
                line = re.sub("20x", "?x", line)
                line = re.sub("tensor.empty\(\)", "tensor.empty(%dimp1)", line)
            if " 20," in line:
                line = re.sub(" 20,", " %dimp1,", line)
            return line

        module = module.splitlines()
        new_lines = []

        # Using a while loop and the pop method to avoid creating a copy of module
        if "llama2_13b" in self.model_name:
            pkv_tensor_shape = "tensor<1x40x?x128x"
        elif "llama2_70b" in self.model_name:
            pkv_tensor_shape = "tensor<1x8x?x128x"
        else:
            pkv_tensor_shape = "tensor<1x32x?x128x"
        if self.precision in ["fp16", "int4", "int8"]:
            pkv_tensor_shape += "f16>"
        else:
            pkv_tensor_shape += "f32>"

        while module:
            line = module.pop(0)
            if "%c19_i64 = arith.constant 19 : i64" in line:
                new_lines.append("%c2 = arith.constant 2 : index")
                new_lines.append(
                    f"%dim_4_int = tensor.dim %arg1, %c2 : {pkv_tensor_shape}"
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

        return "\n".join(new_lines)

    def compile(self):
        # Testing : DO NOT Download Vmfbs if not found. Modify later
        # download vmfbs for A100
        if not self.vicuna_vmfb_path.exists() and self.download_vmfb:
            print(
                f"Looking into gs://shark_tank/{self.model_name}/unsharded/vmfb/{self.vicuna_vmfb_path.name}"
            )
            download_public_file(
                f"gs://shark_tank/{self.model_name}/unsharded/vmfb/{self.vicuna_vmfb_path.name}",
                self.vicuna_vmfb_path.absolute(),
                single_file=True,
            )
        self.shark_model = get_vmfb_from_path(
            self.vicuna_vmfb_path, self.device, "tm_tensor", self.device_id
        )
        if self.shark_model is not None:
            print(f"[DEBUG] vmfb found at {self.vicuna_vmfb_path.absolute()}")
            return

        print(f"[DEBUG] vmfb not found (search path: {self.vicuna_vmfb_path})")
        mlir_generated = False
        for suffix in ["mlirbc", "mlir"]:
            self.vicuna_mlir_path = self.get_model_path(suffix)
            if "cpu" in self.device and "llama2_7b" in self.vicuna_mlir_path.name:
                self.vicuna_mlir_path = Path("llama2_7b_int4_f32.mlir")
            if not self.vicuna_mlir_path.exists() and self.load_mlir_from_shark_tank:
                print(
                    f"Looking into gs://shark_tank/{self.model_name}/unsharded/mlir/{self.vicuna_mlir_path.name}"
                )
                download_public_file(
                    f"gs://shark_tank/{self.model_name}/unsharded/mlir/{self.vicuna_mlir_path.name}",
                    self.vicuna_mlir_path.absolute(),
                    single_file=True,
                )
            if self.vicuna_mlir_path.exists():
                print(f"[DEBUG] mlir found at {self.vicuna_mlir_path.absolute()}")
                combined_module = self.vicuna_mlir_path.absolute()
                mlir_generated = True
                break

        if not mlir_generated:
            print(f"[DEBUG] mlir not found")

            print("[DEBUG] generating mlir on device")
            # Select a compilation prompt such that the resulting input_ids
            # from the model's tokenizer has shape [1, 19]
            compilation_prompt = "".join(["0" for _ in range(17)])

            first_model_path = f"first_{self.model_name}_{self.precision}.mlir"
            if Path(first_model_path).exists():
                print(f"loading {first_model_path}")
                with open(Path(first_model_path), "r") as f:
                    first_module = f.read()
            else:
                # generate first vicuna
                compilation_input_ids = self.tokenizer(
                    compilation_prompt,
                    return_tensors="pt",
                ).input_ids
                compilation_input_ids = torch.tensor(
                    compilation_input_ids
                ).reshape([1, 19])
                firstVicunaCompileInput = (compilation_input_ids,)
                if "cpu" in self.device:
                    model = FirstVicuna(
                        self.hf_model_path,
                        self.precision,
                        "fp32" if self.device=="cpu" else "fp16",
                        self.weight_group_size,
                        self.model_name,
                        self.hf_auth_token,
                    )
                else:
                    model = FirstVicunaGPU(
                        self.hf_model_path,
                        self.precision,
                        "fp32" if self.device=="cpu" else "fp16",
                        self.weight_group_size,
                        self.model_name,
                        self.hf_auth_token,
                    )
                print(f"[DEBUG] generating torchscript graph")
                is_f16 = self.precision in ["fp16", "int4"]
                ts_graph = import_with_fx(
                    model,
                    firstVicunaCompileInput,
                    is_f16=is_f16,
                    precision=self.precision,
                    f16_input_mask=[False, False],
                    mlir_type="torchscript",
                )
                del model
                firstVicunaCompileInput = list(firstVicunaCompileInput)
                firstVicunaCompileInput[
                    0
                ] = torch_mlir.TensorPlaceholder.like(
                    firstVicunaCompileInput[0], dynamic_axes=[1]
                )

                firstVicunaCompileInput = tuple(firstVicunaCompileInput)
                first_module = None
                print(f"[DEBUG] generating torch mlir")
                if self.precision in ["int4", "int8"]:
                    first_module = torch_mlir.compile(
                        ts_graph,
                        [*firstVicunaCompileInput],
                        output_type=torch_mlir.OutputType.TORCH,
                        backend_legal_ops=["quant.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    if self.cache_vicunas:
                        with open(first_model_path[:-5]+"_torch.mlir", "w+") as f:
                            f.write(str(first_module))
                    print(f"[DEBUG] converting torch to linalg")
                    run_pipeline_with_repro_report(
                        first_module,
                        "builtin.module(func.func(torch-unpack-quant-tensor),func.func(torch-convert-custom-quant-op),torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    first_module = torch_mlir.compile(
                        ts_graph,
                        [*firstVicunaCompileInput],
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )
                del ts_graph
                del firstVicunaCompileInput
                gc.collect()

                print(
                    "[DEBUG] successfully generated first vicuna linalg mlir"
                )
                first_module = self.write_in_dynamic_inputs0(
                    str(first_module), dynamic_input_size=19
                )
                if self.cache_vicunas:
                    with open(first_model_path, "w+") as f:
                        f.write(first_module)
                    print("Finished writing IR after dynamic")

            print(f"[DEBUG] Starting generation of second llama")
            second_model_path = f"second_{self.model_name}_{self.precision}.mlir"
            if Path(second_model_path).exists():
                print(f"loading {second_model_path}")
                with open(Path(second_model_path), "r") as f:
                    second_module = f.read()
            else:
                # generate second vicuna
                compilation_input_ids = torch.zeros(
                    [1, 1], dtype=torch.int64
                )
                if self.model_name == "llama2_13b":
                    dim1 = 40
                    total_tuple = 80
                elif self.model_name == "llama2_70b":
                    dim1 = 8
                    total_tuple = 160
                else:
                    dim1 = 32
                    total_tuple = 64
                pkv = tuple(
                    (torch.zeros([1, dim1, 19, 128], dtype=torch.float32))
                    for _ in range(total_tuple)
                )
                secondVicunaCompileInput = (compilation_input_ids,) + pkv
                if "cpu" in self.device:
                    if self.model_name == "llama2_13b":
                        model = SecondVicuna13B(
                            self.hf_model_path,
                            self.precision,
                            "fp32",
                            self.weight_group_size,
                            self.model_name,
                            self.hf_auth_token,
                        )
                    elif self.model_name == "llama2_70b":
                        model = SecondVicuna70B(
                            self.hf_model_path,
                            self.precision,
                            "fp32",
                            self.weight_group_size,
                            self.model_name,
                            self.hf_auth_token,
                        )
                    else:
                        model = SecondVicuna7B(
                            self.hf_model_path,
                            self.precision,
                            "fp32",
                            self.weight_group_size,
                            self.model_name,
                            self.hf_auth_token,
                        )
                else:
                    if self.model_name == "llama2_13b":
                        model = SecondVicuna13BGPU(
                            self.hf_model_path,
                            self.precision,
                            "fp16",
                            self.weight_group_size,
                            self.model_name,
                            self.hf_auth_token,
                        )
                    elif self.model_name == "llama2_70b":
                        model = SecondVicuna70BGPU(
                            self.hf_model_path,
                            self.precision,
                            "fp16",
                            self.weight_group_size,
                            self.model_name,
                            self.hf_auth_token,
                        )
                    else:
                        model = SecondVicuna7BGPU(
                            self.hf_model_path,
                            self.precision,
                            "fp16",
                            self.weight_group_size,
                            self.model_name,
                            self.hf_auth_token,
                        )
                print(f"[DEBUG] generating torchscript graph")
                is_f16 = self.precision in ["fp16", "int4"]
                ts_graph = import_with_fx(
                    model,
                    secondVicunaCompileInput,
                    is_f16=is_f16,
                    precision=self.precision,
                    f16_input_mask=[False] + [True] * total_tuple,
                    mlir_type="torchscript",
                )
                del model
                if self.precision in ["fp16", "int4"]:
                    secondVicunaCompileInput = get_f16_inputs(
                        secondVicunaCompileInput,
                        True,
                        f16_input_mask=[False] + [True] * total_tuple,
                    )
                secondVicunaCompileInput = list(secondVicunaCompileInput)
                for i in range(len(secondVicunaCompileInput)):
                    if i != 0:
                        secondVicunaCompileInput[i] = torch_mlir.TensorPlaceholder.like(
                            secondVicunaCompileInput[i], dynamic_axes=[2]
                        )
                secondVicunaCompileInput = tuple(secondVicunaCompileInput)
                print(f"[DEBUG] generating torch mlir")
                if self.precision in ["int4", "int8"]:
                    second_module = torch_mlir.compile(
                        ts_graph,
                        [*secondVicunaCompileInput],
                        output_type=torch_mlir.OutputType.TORCH,
                        backend_legal_ops=["quant.matmul_rhs_group_quant"],
                        extra_library=brevitas_matmul_rhs_group_quant_library,
                        use_tracing=False,
                        verbose=False,
                    )
                    print(f"[DEBUG] converting torch to linalg")
                    if self.cache_vicunas:
                        with open(second_model_path[:-5]+"_torch.mlir", "w+") as f:
                            f.write(str(second_module))
                    run_pipeline_with_repro_report(
                        second_module,
                        "builtin.module(func.func(torch-unpack-quant-tensor),func.func(torch-convert-custom-quant-op),torch-backend-to-linalg-on-tensors-backend-pipeline)",
                        description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
                    )
                else:
                    second_module = torch_mlir.compile(
                        ts_graph,
                        [*secondVicunaCompileInput],
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )
                del ts_graph
                del secondVicunaCompileInput
                gc.collect()

                print(
                    "[DEBUG] successfully generated second vicuna linalg mlir"
                )
                second_module = self.write_in_dynamic_inputs1(
                    str(second_module)
                )
                if self.cache_vicunas:
                    with open(second_model_path, "w+") as f:
                        f.write(second_module)
                    print("Finished writing IR after dynamic")

            combined_module = self.combine_mlir_scripts(
                first_module,
                second_module,
                self.vicuna_mlir_path,
            )
            combined_module = save_mlir(
                combined_module,
                model_name="combined_llama",
                mlir_dialect="tm_tensor",
                dir=self.vicuna_mlir_path,
            )
            del first_module, second_module

        print(f"Compiling for device : {self.device}"
              f"{'://' + str(self.device_id) if self.device_id is not None else ''}")
        shark_module = SharkInference(
            mlir_module=combined_module,
            device=self.device,
            mlir_dialect="tm_tensor",
            device_idx=self.device_id
        )
        path = shark_module.save_module(
            self.vicuna_vmfb_path.parent.absolute(),
            self.vicuna_vmfb_path.stem,
            extra_args=[
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ]
            + self.extra_args,
            debug=self.debug,
        )
        print("Saved vic vmfb at ", str(path))
        shark_module.load_module(path)
        self.shark_model = shark_module

    def decode_tokens(self, res_tokens):
        for i in range(len(res_tokens)):
            if type(res_tokens[i]) != int:
                res_tokens[i] = int(res_tokens[i][0])

        res_str = self.tokenizer.decode(
            res_tokens, skip_special_tokens=False
        )
        return res_str

    def generate(self, prompt, cli):
        # TODO: refactor for cleaner integration
        if self.shark_model is None:
            self.compile()
        res_tokens = []
        params = {"prompt": prompt, "is_first": True, "fv": self.shark_model}

        prefill_st_time = time.time()
        generated_token_op = self.generate_new_token(
            params=params, sharded=False, cli=cli
        )
        prefill_time = time.time() - prefill_st_time

        token = generated_token_op["token"]
        if "cpu" not in self.device:
            logits = generated_token_op["logits"]
        pkv = generated_token_op["past_key_values"]
        detok = generated_token_op["detok"]
        yield detok, None, prefill_time

        res_tokens.append(token)
        if cli:
            print(f"Assistant: {detok}", end=" ", flush=True)

        for idx in range(self.max_num_tokens):
            params = {
                "token": token,
                "is_first": False,
                "past_key_values": pkv,
                "sv": self.shark_model,
            }
            if "cpu" not in self.device:
                params["logits"] = logits

            decode_st_time = time.time()
            generated_token_op = self.generate_new_token(
                params=params, sharded=False, cli=cli
            )
            decode_time_ms = (time.time() - decode_st_time)*1000

            token = generated_token_op["token"]
            if "cpu" not in self.device:
                logits = generated_token_op["logits"]
            pkv = generated_token_op["past_key_values"]
            detok = generated_token_op["detok"]

            if token == 2 and idx >= self.min_num_tokens:
                break
            res_tokens.append(token)
            if detok == "<0x0A>":
                if cli:
                    print("\n", end="", flush=True)
            else:
                if cli:
                    print(f"{detok}", end=" ", flush=True)
            yield detok, None, decode_time_ms

        res_str = self.decode_tokens(res_tokens)
        yield res_str, "formatted", None

    def autocomplete(self, prompt):
        # use First vic alone to complete a story / prompt / sentence.
        pass


# NOTE: Each `model_name` should have its own start message
start_message = {
    "llama2_7b": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "llama2_13b": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "llama2_70b": (
        "System: You are a helpful, respectful and honest assistant. Always answer "
        "as helpfully as possible, while being safe.  Your answers should not "
        "include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
        "content. Please ensure that your responses are socially unbiased and positive "
        "in nature. If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. If you don't know the "
        "answer to a question, please don't share false information."
    ),
    "vicuna": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions.\n"
    ),
}


def create_prompt(model_name, history):
    global start_message
    system_message = start_message[model_name]
    if "llama2" in model_name:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        conversation = "".join(
            [
                f"{B_INST} {item[0].strip()} {E_INST} {item[1].strip()} "
                for item in history[1:]
            ]
        )
        msg = f"{B_INST} {B_SYS} {system_message} {E_SYS} {history[0][0]} {E_INST} {history[0][1]} {conversation}"

    else:
        conversation = "".join(
            [
                "".join(["<|USER|>" + item[0], "<|ASSISTANT|>" + item[1]])
                for item in history
            ]
        )
        msg = system_message + conversation
        msg = msg.strip()
    return msg


def miliseconds_to_seconds(ms: float) -> float:
    return ms / 1000.0


@dataclass
class BenchmarkRunInfo:
    num_prompt_tokens : int
    prefill_time_ms : float
    token_times_ms : list[float]

    def get_prefill_speed(self) -> float:
        return self.num_prompt_tokens / miliseconds_to_seconds(self.prefill_time_ms)

    def num_generated_tokens(self) -> int:
        return len(self.token_times_ms)

    def get_decode_time_ms(self) -> float:
        return sum(self.token_times_ms)

    def get_decode_speed(self) -> float:
        return self.num_generated_tokens() / miliseconds_to_seconds(self.get_decode_time_ms())

    def get_e2e_time_ms(self) -> float:
        return self.prefill_time_ms + self.get_decode_time_ms()

    def get_e2e_decode_speed(self) -> float:
        return self.num_generated_tokens() / miliseconds_to_seconds(self.get_e2e_time_ms())

    def get_e2e_token_processing_speed(self) -> float:
        return (self.num_prompt_tokens + self.num_generated_tokens()) / miliseconds_to_seconds(self.get_e2e_time_ms())

    def print(self) -> None:
        total_tokens = self.num_prompt_tokens + self.num_generated_tokens()
        print(f"Num tokens: {self.num_prompt_tokens:} (prompt), {self.num_generated_tokens()} (generated), {total_tokens} (total)")
        print(f"Prefill: {self.prefill_time_ms:.2f} ms, {self.get_prefill_speed():.2f} tokens/s")
        print(f"Decode: {self.get_decode_time_ms():.2f} ms, {self.get_decode_speed():.2f} tokens/s")
        print(f"Decode end-2-end: {self.get_e2e_decode_speed():.2f} tokens/s (w/o prompt), {self.get_e2e_token_processing_speed():.2f} tokens/s (w/ prompt)")


def print_aggregate_stats(run_infos: list[BenchmarkRunInfo]) -> None:
    num_iterations = len(run_infos)
    print(f'Number of iterations: {num_iterations}')
    if num_iterations == 0:
        return

    if len(run_infos) == 1:
        print(run_infos[0])
        return

    total_tokens = run_infos[0].num_prompt_tokens + run_infos[0].num_generated_tokens()
    print(f"Num tokens: {run_infos[0].num_prompt_tokens} (prompt), {run_infos[0].num_generated_tokens()} (generated), {total_tokens} (total)")

    def avg_and_stdev(data):
        x = list(data)
        return mean(x), stdev(x)

    avg_prefill_ms, stdev_prefill = avg_and_stdev(x.prefill_time_ms for x in run_infos)
    avg_prefill_speed = mean(x.get_prefill_speed() for x in run_infos)
    print(f"Prefill: avg. {avg_prefill_ms:.2f} ms (stdev {stdev_prefill:.2f}), avg. {avg_prefill_speed:.2f} tokens/s")

    avg_decode_ms, stdev_decode = avg_and_stdev(x.get_decode_time_ms() for x in run_infos)
    avg_decode_speed = mean(x.get_prefill_speed() for x in run_infos)
    print(f"Decode: avg. {avg_decode_ms:.2f} ms (stdev {stdev_decode:.2f}), avg. {avg_decode_speed:.2f} tokens/s")

    avg_e2e_decode_speed = mean(x.get_e2e_decode_speed() for x in run_infos)
    avg_e2e_processing_speed = mean(x.get_e2e_token_processing_speed() for x in run_infos)
    print(f"Decode end-2-end: avg. {avg_e2e_decode_speed:.2f} tokens/s (w/o prompt), avg. {avg_e2e_processing_speed:.2f} (w/ prompt)")


if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    _extra_args = list(args.Xiree_compile)

    device_id = None
    # Process vulkan target triple.
    # TODO: This feature should just be in a common utils for other LLMs and in general
    #       any model run via SHARK for Vulkan backend.
    vulkan_target_triple = args.iree_vulkan_target_triple
    if vulkan_target_triple != "":
        _extra_args.append(
            f"-iree-vulkan-target-triple={args.iree_vulkan_target_triple}"
        )
        # Step 1. Fetch the device ID.
        from shark.iree_utils.vulkan_utils import (
            get_all_vulkan_devices,
            get_vulkan_target_triple
        )
        vulkaninfo_list = get_all_vulkan_devices()
        id = 0
        for device in vulkaninfo_list:
            target_triple = get_vulkan_target_triple(vulkaninfo_list[id])
            if target_triple == vulkan_target_triple:
                device_id = id
                break
            id += 1

        assert device_id, f"no vulkan hardware for target-triple '{vulkan_target_triple}' exists"
        # Step 2. Add a few flags targetting specific hardwares.
        if "rdna" in vulkan_target_triple:
            flags_to_add = [
                "--iree-spirv-index-bits=64",
            ]
            _extra_args = _extra_args + flags_to_add


    vic = None
    if not args.sharded:
        vic_mlir_path = (
            None
            if args.vicuna_mlir_path is None
            else Path(args.vicuna_mlir_path)
        )
        vic_vmfb_path = (
            None
            if args.vicuna_vmfb_path is None
            else Path(args.vicuna_vmfb_path)
        )
        min_tokens = 0
        max_tokens = 512
        if args.enable_microbenchmark:
            min_tokens = max_tokens = args.microbenchmark_num_tokens

        vic = UnshardedVicuna(
            model_name=args.model_name,
            hf_auth_token=args.hf_auth_token,
            max_num_tokens=max_tokens,
            min_num_tokens=min_tokens,
            device=args.device,
            vulkan_target_triple=vulkan_target_triple,
            precision=args.precision,
            vicuna_mlir_path=vic_mlir_path,
            vicuna_vmfb_path=vic_vmfb_path,
            load_mlir_from_shark_tank=args.load_mlir_from_shark_tank,
            weight_group_size=args.weight_group_size,
            download_vmfb=args.download_vmfb,
            cache_vicunas=args.cache_vicunas,
            extra_args_cmd=_extra_args,
            device_id=device_id
        )
    else:
        if args.config is not None:
            config_file = open(args.config)
            config_json = json.load(config_file)
            config_file.close()
        else:
            config_json = None
        vic = ShardedVicuna(
            model_name=args.model_name,
            device=args.device,
            precision=args.precision,
            config_json=config_json,
            weight_group_size=args.weight_group_size,
            extra_args_cmd=_extra_args,
        )

    history = []

    model_list = {
        "vicuna": "vicuna=>TheBloke/vicuna-7B-1.1-HF",
        "llama2_7b": "llama2_7b=>meta-llama/Llama-2-7b-chat-hf",
        "llama2_13b": "llama2_13b=>meta-llama/Llama-2-13b-chat-hf",
        "llama2_70b": "llama2_70b=>meta-llama/Llama-2-70b-chat-hf",
    }

    iteration = 0

    benchmark_run_infos = []

    while True:
        # TODO: Add break condition from user input
        iteration += 1
        if not args.enable_microbenchmark:
            user_prompt = input("User prompt: ")
            history.append([user_prompt, ""])
            prompt = create_prompt(args.model_name, history)
        else:
            if iteration > args.microbenchmark_iterations:
                break
            user_prompt = args.user_prompt
            prompt = args.system_prompt + user_prompt
            history = [[user_prompt, ""]]

        prompt_token_count = 1 # TODO
        total_time_ms = 0.0  # In order to avoid divide by zero error
        prefill_time_ms = 0
        is_first = True
        token_times_ms = []

        for text, msg, exec_time_ms in vic.generate(prompt, cli=True):
            if msg is None:
                if is_first:
                    prefill_time_ms = exec_time_ms
                    is_first = False
                else:
                    token_times_ms.append(exec_time_ms)
            elif "formatted" in msg:
                history[-1][1] = text
                print(f"\nResponse:\n{text.strip()}\n")
                run_info = BenchmarkRunInfo(prompt_token_count, prefill_time_ms, token_times_ms)
                run_info.print()
                benchmark_run_infos.append(run_info)

            else:
                sys.exit(
                    "unexpected message from the vicuna generate call, exiting."
                )

    if args.enable_microbenchmark:
        print("\n### Final Statistics ###")
        print_aggregate_stats(benchmark_run_infos)
