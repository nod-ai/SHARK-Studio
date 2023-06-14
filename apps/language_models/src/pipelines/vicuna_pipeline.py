from apps.language_models.src.model_wrappers.vicuna_model import (
    FirstVicuna,
    SecondVicuna,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.utils import (
    get_vmfb_from_path,
)

from io import BytesIO
from pathlib import Path
from shark.shark_downloader import download_public_file
from shark.shark_importer import import_with_fx
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
        first_vicuna_mlir_path=Path("first_vicuna.mlir"),
        second_vicuna_mlir_path=Path("second_vicuna.mlir"),
        first_vicuna_vmfb_path=Path("first_vicuna.vmfb"),
        second_vicuna_vmfb_path=Path("second_vicuna.vmfb"),
        load_mlir_from_shark_tank=True,
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_length = 256
        self.device = device
        self.precision = precision
        self.first_vicuna_vmfb_path = first_vicuna_vmfb_path
        self.second_vicuna_vmfb_path = second_vicuna_vmfb_path
        self.first_vicuna_mlir_path = first_vicuna_mlir_path
        self.second_vicuna_mlir_path = second_vicuna_mlir_path
        self.tokenizer = self.get_tokenizer()
        self.shark_model = self.compile()
        self.load_mlir_from_shark_tank = load_mlir_from_shark_tank

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
            f"[DEBUG] vmfb not found at {self.first_vicuna_vmfb_path.absolute()}. Trying to work with"
            f"[DEBUG] mlir path { self.first_vicuna_mlir_path} {'exists' if self.first_vicuna_mlir_path.exists() else 'does not exist'}"
        )
        if self.first_vicuna_mlir_path.exists():
            with open(self.first_vicuna_mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            mlir_generated = False
            if self.load_mlir_from_shark_tank:
                if self.precision == "fp32":
                    # download MLIR from shark_tank for fp32
                    download_public_file(
                        "gs://shark_tank/vicuna/unsharded/mlir/first_vicuna.mlir",
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
                        "Only fp32 mlir added to tank, generating mlir on device."
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
                model = FirstVicuna(self.hf_model_path)

                print(f"[DEBUG] generating torchscript graph")
                ts_graph = import_with_fx(
                    model,
                    firstVicunaCompileInput,
                    is_f16=self.precision == "fp16",
                    f16_input_mask=[False, False],
                    mlir_type="torchscript",
                )
                del model
                print(f"[DEBUG] generating torch mlir")

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
        print("Saved first vic vmfb at vmfb at ", str(path))
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
                if self.precision == "fp32":
                    # download MLIR from shark_tank for fp32
                    download_public_file(
                        "gs://shark_tank/vicuna/unsharded/mlir/second_vicuna.mlir",
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
                        "Only fp32 mlir added to tank, generating mlir on device."
                    )

            if not mlir_generated:
                compilation_input_ids = torch.zeros([1, 1], dtype=torch.int64)
                pkv = tuple(
                    (torch.zeros([1, 32, 19, 128], dtype=torch.float32))
                    for _ in range(64)
                )
                secondVicunaCompileInput = (compilation_input_ids,) + pkv
                model = SecondVicuna(self.hf_model_path)
                ts_graph = import_with_fx(
                    model,
                    secondVicunaCompileInput,
                    is_f16=self.precision == "fp16",
                    f16_input_mask=[False, False],
                    mlir_type="torchscript",
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
        print("Saved vmfb at ", str(path))
        shark_module.load_module(self.second_vicuna_vmfb_path)

        # self.shark_module = shark_module

        return shark_module

    def compile(self):
        # Cannot load both the models in the memory at once
        # due to memory constraints, hence on demand compilation
        # is being used until the space is enough for both models

        # Testing : DO NOT Download Vmfbs if not found. Modify later
        # download vmfbs for A100
        if (
            not self.first_vicuna_vmfb_path.exists()
            and self.device == "cuda"
            and self.precision == "fp32"
        ):
            download_public_file(
                "gs://shark_tank/vicuna/unsharded/first_vicuna.vmfb",
                self.first_vicuna_vmfb_path.absolute(),
                single_file=True,
            )
        else:
            # get first vic
            # TODO: Remove after testing to avoid memory overload
            # fvic_shark_model = self.compile_first_vicuna()
            pass
        if (
            not self.second_vicuna_vmfb_path.exists()
            and self.device == "cuda"
            and self.precision == "fp32"
        ):
            download_public_file(
                "gs://shark_tank/vicuna/unsharded/second_vicuna.vmfb",
                self.second_vicuna_vmfb_path.absolute(),
                single_file=True,
            )
        else:
            # get second vic
            # TODO: Remove after testing to avoid memory overload
            # svic_shark_model = self.compile_second_vicuna()
            pass

        # get first vic
        # fvic_shark_model = self.compile_first_vicuna()
        # get second vic
        # svic_shark_model = self.compile_second_vicuna()
        # return tuple of shark_modules
        # return fvic_shark_model, svic_shark_model
        return None
        # return tuple of shark_modules once mem is supported
        # return fvic_shark_model, svic_shark_model

    def generate(self, prompt, cli=False):
        # TODO: refactor for cleaner integration
        import gc

        res = []
        res_tokens = []
        params = {
            "prompt": prompt,
            "is_first": True,
            "fv": self.compile_first_vicuna(),
        }

        generated_token_op = self.generate_new_token(params=params)

        token = generated_token_op["token"]
        logits = generated_token_op["logits"]
        pkv = generated_token_op["pkv"]
        detok = generated_token_op["detok"]

        res.append(detok)
        res_tokens.append(token)
        if cli:
            print(f"Assistant: {detok}", end=" ", flush=True)

        # Clear First Vic from Memory (main and cuda)
        del params
        torch.cuda.empty_cache()
        gc.collect()

        sec_vic = self.compile_second_vicuna()
        for _ in range(self.max_num_tokens - 2):
            params = {
                "prompt": None,
                "is_first": False,
                "logits": logits,
                "pkv": pkv,
                "sv": sec_vic,
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
                res.append("\n")
                if cli:
                    print("\n", end="", flush=True)
            else:
                res.append(detok)
                if cli:
                    print(f"{detok}", end=" ", flush=True)
        del sec_vic, pkv, logits
        torch.cuda.empty_cache()
        gc.collect()

        for i in range(len(res_tokens)):
            if type(res_tokens[i]) != int:
                res_tokens[i] = int(res_tokens[i][0])

        res_str = self.tokenizer.decode(res_tokens)
        # print(f"[DEBUG] final output : \n{res_str}")
        return res_str

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
