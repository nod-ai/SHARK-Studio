from apps.language_models.src.model_wrappers.vicuna_model import (
    FirstVicuna,
    SecondVicuna,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.utils import get_torch_mlir_module_bytecode
from io import BytesIO
from pathlib import Path
from shark.shark_inference import SharkInference
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    def compile_first_vicuna(self):
        vmfb_path = Path(self.model_name + ".vmfb")
        if vmfb_path.exists():
            shark_module = SharkInference(
                None, device=self.device, mlir_dialect="tm_tensor"
            )
            shark_module.load_module(vmfb_path)
            # self.shark_module = shark_module
            return shark_module
        mlir_path = Path(self.model_name + ".mlir")
        print(
            f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
        )
        if mlir_path.exists():
            with open(mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            compilation_prompt = "".join(["0" for _ in range(17)])
            compilation_input_ids = self.tokenizer(
                compilation_prompt
            ).input_ids
            compilation_input_ids = torch.tensor(
                compilation_input_ids
            ).reshape([1, 19])
            firstVicunaCompileInput = (compilation_input_ids,)
            model = FirstVicuna(self.hf_model_path)

            ts_graph = get_torch_mlir_module_bytecode(
                model, firstVicunaCompileInput
            )

            firstVicunaCompileInput = list(firstVicunaCompileInput)
            firstVicunaCompileInput[0] = torch_mlir.TensorPlaceholder.like(
                firstVicunaCompileInput[0], dynamic_axes=[1]
            )
            firstVicunaCompileInput = tuple(firstVicunaCompileInput)
            module = torch_mlir.compile(
                ts_graph,
                [*firstVicunaCompileInput],
                torch_mlir.OutputType.LINALG_ON_TENSORS,
                use_tracing=False,
                verbose=False,
            )

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

            module_str = str(module)
            new_lines = []

            for line in module_str.splitlines():
                line = remove_constant_dim(line)
                if "%0 = tensor.empty(%dim) : tensor<?xi64>" in line:
                    new_lines.append(
                        "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>"
                    )
                if "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>" in line:
                    continue

                new_lines.append(line)

            module_str = "\n".join(new_lines)
            bytecode = module_str.encode("UTF-8")
            bytecode_stream = BytesIO(bytecode)
            bytecode = bytecode_stream.read()
            f_ = open(f"{self.model_name}.mlir", "wb")
            f_.write(bytecode)
            f_.close()

        shark_module = SharkInference(
            mlir_module=bytecode, device=self.device, mlir_dialect="tm_tensor"
        )

        path = shark_module.save_module(
            os.getcwd(),
            self.model_name,
            extra_args=[
                "--iree-hal-dump-executable-sources-to=ies",
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ],
        )
        print("Saved vmfb at ", str(path))
        shark_module.load_module(vmfb_path)

        return shark_module

    def compile_second_vicuna(self):
        vmfb_path = Path(self.model_name + ".vmfb")
        if vmfb_path.exists():
            shark_module = SharkInference(
                None, device=self.device, mlir_dialect="tm_tensor"
            )
            shark_module.load_module(vmfb_path)
            # self.shark_module = shark_module
            return shark_module
        mlir_path = Path(self.model_name + ".mlir")
        print(
            f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
        )
        if mlir_path.exists():
            with open(mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            compilation_input_ids = torch.zeros([1, 1], dtype=torch.int64)
            pkv = tuple(
                (torch.zeros([1, 32, 19, 128], dtype=torch.float32))
                for _ in range(64)
            )
            secondVicunaCompileInput = (compilation_input_ids,) + pkv
            model = SecondVicuna(self.hf_model_path)
            ts_graph = get_torch_mlir_module_bytecode(
                model, secondVicunaCompileInput
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

            for line in module_str.splitlines():
                if "%c19_i64 = arith.constant 19 : i64" in line:
                    new_lines.append("%c2 = arith.constant 2 : index")
                    new_lines.append(
                        "%dim_4_int = tensor.dim %arg1, %c2 : tensor<1x32x?x128xf32>"
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
            bytecode = module_str.encode("UTF-8")
            bytecode_stream = BytesIO(bytecode)
            bytecode = bytecode_stream.read()
            f_ = open(f"{self.model_name}.mlir", "wb")
            f_.write(bytecode)
            f_.close()

        shark_module = SharkInference(
            mlir_module=bytecode, device=self.device, mlir_dialect="tm_tensor"
        )

        path = shark_module.save_module(
            os.getcwd(),
            self.model_name,
            extra_args=[
                "--iree-hal-dump-executable-sources-to=ies",
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ],
        )
        print("Saved vmfb at ", str(path))
        shark_module.load_module(vmfb_path)

        # self.shark_module = shark_module

        return shark_module

    def compile(self):
        # get first vic
        # fvic_shark_model = self.compile_first_vicuna()
        # get second vic
        # svic_shark_model = self.compile_second_vicuna()
        # return tuple of shark_modules
        # return fvic_shark_model, svic_shark_model
        return None

    def generate(self, prompt):
        # TODO: refactor for cleaner integration

        res = []
        params = {
            "prompt": prompt,
            "is_first": True,
        }

        generated_token_op = self.generate_new_token(params=params)

        token = generated_token_op["token"]
        logits = generated_token_op["logits"]
        pkv = generated_token_op["pkv"]
        detok = generated_token_op["detok"]

        res.append(detok)

        for _ in range(self.max_num_tokens - 2):
            # t1 = time.time()
            params = {
                "prompt": None,
                "is_first": False,
                "logits": logits,
                "pkv": pkv,
            }

            generated_token_op = self.generate_new_token(params=params)
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            token = generated_token_op["token"]
            logits = generated_token_op["logits"]
            pkv = generated_token_op["pkv"]
            detok = generated_token_op["detok"]

            if token == 2:
                break
            if detok == "<0x0A>":
                res.append("\n")
            else:
                res.append(detok)

        return res

    def generate_new_token(self, params):
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
            fv = self.compile_first_vicuna()
            token, logits, pkv = forward_first(
                fv,  # self.shark_model[0],
                prompt=prompt,
                cache_outputs=False,
            )
            del fv
        else:
            _logits = params["logits"]
            _pkv = params["pkv"]
            inputs = (_logits,) + tuple(_pkv)
            sv = self.compile_second_vicuna()
            token, logits, pkv = forward_second(
                sv,  # self.shark_model[1],
                inputs=inputs,
                load_inputs=False,
            )
            del sv

        detok = self.tokenizer.decode(token)
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
