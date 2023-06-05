from apps.language_models.src.model_wrappers.vicuna_sharded_model import (
    FirstVicunaLayer,
    SecondVicunaLayer,
    CompiledFirstVicunaLayer,
    CompiledSecondVicunaLayer,
    ShardedVicunaModel,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.utils import get_torch_mlir_module_bytecode
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


class Vicuna(SharkLLMBase):
    def __init__(
        self,
        model_name,
        hf_model_path="TheBloke/vicuna-7B-1.1-HF",
        max_num_tokens=512,
        device="cuda",
        precision="fp32",
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.tokenizer = self.get_tokenizer()
        self.shark_model = self.compile()

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

    def write_in_dynamic_inputs0(self, module, dynamic_input_size):
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
        mlir_bytecode = get_torch_mlir_module_bytecode(
            vicuna_layer, model_inputs
        )
        return mlir_bytecode

    def compile_to_vmfb(self, inputs, layers, is_first=True):
        mlirs, modules = [], []
        for idx, layer in tqdm(enumerate(layers), desc="Getting mlirs"):
            if is_first:
                mlir_path = Path(f"{idx}_0.mlir")
                vmfb_path = Path(f"{idx}_0.vmfb")
            else:
                mlir_path = Path(f"{idx}_1.mlir")
                vmfb_path = Path(f"{idx}_1.vmfb")
            if vmfb_path.exists():
                continue
            if mlir_path.exists():
                # print(f"Found layer {idx} mlir")
                f_ = open(mlir_path, "rb")
                bytecode = f_.read()
                f_.close()
            else:
                hidden_states_placeholder = TensorPlaceholder.like(
                    inputs[0], dynamic_axes=[1]
                )
                attention_mask_placeholder = TensorPlaceholder.like(
                    inputs[1], dynamic_axes=[3]
                )
                position_ids_placeholder = TensorPlaceholder.like(
                    inputs[2], dynamic_axes=[1]
                )
                if not is_first:
                    pkv0_placeholder = TensorPlaceholder.like(
                        inputs[3], dynamic_axes=[2]
                    )
                    pkv1_placeholder = TensorPlaceholder.like(
                        inputs[4], dynamic_axes=[2]
                    )
                print(f"Compiling layer {idx} mlir")
                if is_first:
                    ts_g = self.compile_vicuna_layer(
                        layer, inputs[0], inputs[1], inputs[2]
                    )
                    module = torch_mlir.compile(
                        ts_g,
                        (
                            hidden_states_placeholder,
                            inputs[1],
                            inputs[2],
                        ),
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )
                else:
                    ts_g = self.compile_vicuna_layer(
                        layer,
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                    )
                    module = torch_mlir.compile(
                        ts_g,
                        (
                            inputs[0],
                            attention_mask_placeholder,
                            inputs[2],
                            pkv0_placeholder,
                            pkv1_placeholder,
                        ),
                        torch_mlir.OutputType.LINALG_ON_TENSORS,
                        use_tracing=False,
                        verbose=False,
                    )

                # bytecode_stream = BytesIO()
                # module.operation.write_bytecode(bytecode_stream)
                # bytecode = bytecode_stream.getvalue()

                if is_first:
                    module = self.write_in_dynamic_inputs0(str(module), 137)
                    bytecode = module.encode("UTF-8")
                    bytecode_stream = BytesIO(bytecode)
                    bytecode = bytecode_stream.read()

                else:
                    module = self.write_in_dynamic_inputs1(str(module), 138)
                    if idx in [0, 5, 6, 7]:
                        module_str = module
                        module_str = module_str.splitlines()
                        new_lines = []
                        for line in module_str:
                            if len(line) < 1000:
                                new_lines.append(line)
                            else:
                                new_lines.append(line[:999])
                        module_str = "\n".join(new_lines)
                        f1_ = open(f"{idx}_1_test.mlir", "w+")
                        f1_.write(module_str)
                        f1_.close()

                    bytecode = module.encode("UTF-8")
                    bytecode_stream = BytesIO(bytecode)
                    bytecode = bytecode_stream.read()

                f_ = open(mlir_path, "wb")
                f_.write(bytecode)
                f_.close()
            mlirs.append(bytecode)

        for idx, layer in tqdm(enumerate(layers), desc="compiling modules"):
            if is_first:
                vmfb_path = Path(f"{idx}_0.vmfb")
                if idx < 25:
                    device = "cpu"
                else:
                    device = "cpu"
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
                    module.save_module(
                        module_name=f"{idx}_0",
                        extra_args=[
                            "--iree-hal-dump-executable-sources-to=ies",
                            "--iree-vm-target-truncate-unsupported-floats",
                            "--iree-codegen-check-ir-before-llvm-conversion=false",
                            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                        ],
                    )
                    module.load_module(vmfb_path)
                modules.append(module)
            else:
                vmfb_path = Path(f"{idx}_1.vmfb")
                if idx < 25:
                    device = "cpu"
                else:
                    device = "cpu"
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
                    module.save_module(
                        module_name=f"{idx}_1",
                        extra_args=[
                            "--iree-hal-dump-executable-sources-to=ies",
                            "--iree-vm-target-truncate-unsupported-floats",
                            "--iree-codegen-check-ir-before-llvm-conversion=false",
                            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
                        ],
                    )
                    module.load_module(vmfb_path)
                modules.append(module)

        return mlirs, modules

    def get_sharded_model(self):
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

        layers0 = [
            FirstVicunaLayer(layer) for layer in vicuna_model.model.layers
        ]
        _, modules0 = self.compile_to_vmfb(
            placeholder_input0, layers0, is_first=True
        )
        shark_layers0 = [CompiledFirstVicunaLayer(m) for m in modules0]

        layers1 = [
            SecondVicunaLayer(layer) for layer in vicuna_model.model.layers
        ]
        _, modules1 = self.compile_to_vmfb(
            placeholder_input1, layers1, is_first=False
        )
        shark_layers1 = [CompiledSecondVicunaLayer(m) for m in modules1]

        sharded_model = ShardedVicunaModel(
            vicuna_model, shark_layers0, shark_layers1
        )
        return sharded_model

    def compile(self):
        return self.get_sharded_model()

    def generate(self, prompt):
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


if __name__ == "__main__":
    vic = Vicuna("vicuna")
    prompt_history = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    prologue_prompt = "ASSISTANT:\n"
    user_prompt = input("User: ")
    prompt_history = prompt_history + "USER:\n" + user_prompt + prologue_prompt
    prompt = prompt_history.strip()

    res = vic.generate(prompt)
    print(prompt + res)
