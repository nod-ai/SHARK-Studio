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
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from shark.shark_importer import import_with_fx
from io import BytesIO
from pathlib import Path
from shark.shark_inference import SharkInference
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from torch_mlir import TensorPlaceholder


import re
import torch
import torch_mlir
import os
import json


class Vicuna(SharkLLMBase):
    # Class representing Sharded Vicuna Model
    def __init__(
        self,
        model_name,
        hf_model_path="TheBloke/vicuna-7B-1.1-HF",
        max_num_tokens=512,
        device="cuda",
        precision="fp32",
        config_json=None,
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.tokenizer = self.get_tokenizer()
        self.config = config_json
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
            vdtype = vbody.split(":")[1].strip()
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
            is_f16=self.precision == "fp16",
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
            mmap=False,
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
            mmap=False,
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
            mmap=False,
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
                # print(f"Found layer {idx} vmfb")
                device_idx = self.get_device_index(
                    f"first_vicuna.model.model.layers.{idx}[\s.$]"
                )
                module = SharkInference(
                    None,
                    device=device,
                    device_idx=idx % 4,
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
                    device_idx=idx % 4,
                    mlir_dialect="tm_tensor",
                    mmap=False,
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

        layers0 = [
            FirstVicunaLayer(layer) for layer in vicuna_model.model.layers
        ]

        layers1 = [
            SecondVicunaLayer(layer) for layer in vicuna_model.model.layers
        ]

        _, modules = self.compile_to_vmfb_one_model(
            placeholder_input0,
            layers0,
            placeholder_input1,
            layers1,
            device=device,
        )
        shark_layers = [CompiledVicunaLayer(m) for m in modules]

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
