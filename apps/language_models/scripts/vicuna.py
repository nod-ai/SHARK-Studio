import torch
import torch_mlir
from transformers import AutoTokenizer, AutoModelForCausalLM
from io import BytesIO
from pathlib import Path
import re
from shark.shark_inference import SharkInference
from tqdm import tqdm
from torch_mlir import TensorPlaceholder
from apps.language_models.utils import get_torch_mlir_module_bytecode
import os

MODEL_PATH = "TheBloke/vicuna-7B-1.1-HF"


class FirstVicunaLayer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, attention_mask, position_ids):
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        next_hidden_states = outputs[0]
        past_key_value_out0, past_key_value_out1 = (
            outputs[-1][0],
            outputs[-1][1],
        )

        return (
            next_hidden_states,
            past_key_value_out0,
            past_key_value_out1,
        )


class SecondVicunaLayer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value0,
        past_key_value1,
    ):
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(
                past_key_value0,
                past_key_value1,
            ),
            use_cache=True,
        )
        next_hidden_states = outputs[0]
        past_key_value_out0, past_key_value_out1 = (
            outputs[-1][0],
            outputs[-1][1],
        )

        return (
            next_hidden_states,
            past_key_value_out0,
            past_key_value_out1,
        )


class CompiledFirstVicunaLayer(torch.nn.Module):
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
        use_cache=True,
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

        output0 = torch.tensor(output[0])
        output1 = torch.tensor(output[1])
        output2 = torch.tensor(output[2])

        return (
            output0,
            (
                output1,
                output2,
            ),
        )


class CompiledSecondVicunaLayer(torch.nn.Module):
    def __init__(self, shark_module):
        super().__init__()
        self.model = shark_module

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
        pkv0 = past_key_value[0].detach()
        pkv1 = past_key_value[1].detach()
        output = self.model(
            "forward",
            (
                hidden_states,
                attention_mask,
                position_ids,
                pkv0,
                pkv1,
            ),
        )

        output0 = torch.tensor(output[0])
        output1 = torch.tensor(output[1])
        output2 = torch.tensor(output[2])

        return (
            output0,
            (
                output1,
                output2,
            ),
        )


class FirstVicunaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, use_fast=False
        )
        self.shark_module = None

    def compile(self, device="cpu"):
        compilation_prompt = "".join(["0" for _ in range(17)])
        compilation_input_ids = self.tokenizer(compilation_prompt).input_ids
        compilation_input_ids = torch.tensor(compilation_input_ids).reshape(
            [1, 19]
        )
        firstVicunaCompileInput = (compilation_input_ids,)
        model = FirstVicuna(MODEL_PATH)

        vmfb_path = Path(self.model_name + ".vmfb")
        if vmfb_path.exists():
            shark_module = SharkInference(
                None, device=device, mlir_dialect="tm_tensor"
            )
            shark_module.load_module(vmfb_path)
            self.shark_module = shark_module
            return shark_module
        mlir_path = Path(self.model_name + ".mlir")
        print(
            f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
        )
        if mlir_path.exists():
            with open(mlir_path, "rb") as f:
                bytecode = f.read()
        else:
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
            mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
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

        self.shark_module = shark_module

        return shark_module

    def forward(self, prompt, cache_outputs=False):
        input_ids = self.tokenizer(prompt).input_ids
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.reshape([1, input_id_len])
        firstVicunaInput = (input_ids,)
        assert self.shark_module is not None
        output_first_vicuna = self.shark_module("forward", firstVicunaInput)
        output_first_vicuna_tensor = torch.tensor(output_first_vicuna[1:])
        logits_first_vicuna = torch.tensor(output_first_vicuna[0])
        if cache_outputs:
            torch.save(logits_first_vicuna, "logits_first_vicuna_tensor.pt")
            torch.save(
                output_first_vicuna_tensor, "output_first_vicuna_tensor.pt"
            )
        token = torch.argmax(
            torch.tensor(logits_first_vicuna)[:, -1, :], dim=1
        )
        return token, logits_first_vicuna, output_first_vicuna_tensor


class SecondVicunaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, use_fast=False
        )
        self.shark_module = None

    def compile(self, device="cpu"):
        compilation_input_ids = torch.zeros([1, 1], dtype=torch.int64)
        pkv = tuple(
            (torch.zeros([1, 32, 19, 128], dtype=torch.float32))
            for _ in range(64)
        )
        secondVicunaCompileInput = (compilation_input_ids,) + pkv
        model = SecondVicuna(MODEL_PATH)

        vmfb_path = Path(self.model_name + ".vmfb")
        if vmfb_path.exists():
            shark_module = SharkInference(
                None, device=device, mlir_dialect="tm_tensor"
            )
            shark_module.load_module(vmfb_path)
            self.shark_module = shark_module
            return shark_module
        mlir_path = Path(self.model_name + ".mlir")
        print(
            f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
        )
        if mlir_path.exists():
            with open(mlir_path, "rb") as f:
                bytecode = f.read()
        else:
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
            mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
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

        self.shark_module = shark_module

        return shark_module

    def forward(self, inputs=None, load_inputs=False):
        if inputs is not None:
            logits = inputs[0]
            token = torch.argmax(torch.tensor(logits)[:, -1, :], dim=1)
            token = token.to(torch.int64).reshape([1, 1])
            pkv = inputs[1:]
            secondVicunaInput = (token,) + tuple(pkv)
        elif load_inputs:
            pkv = torch.load("output_first_vicuna_tensor.pt")
            pkv = tuple(torch.tensor(x) for x in pkv)
            logits = torch.load("logits_first_vicuna_tensor.pt")
            token = torch.argmax(torch.tensor(logits)[:, -1, :], dim=1)
            token = token.to(torch.int64).reshape([1, 1])
            secondVicunaInput = (token,) + pkv
        else:
            print("Either inputs must be given, or load_inputs must be true")
            return None
        secondVicunaOutput = self.shark_module("forward", secondVicunaInput)
        new_pkv = secondVicunaOutput[1:]
        new_logits = secondVicunaOutput[0]
        new_token = torch.argmax(torch.tensor(new_logits)[:, -1, :], dim=1)
        return new_token, new_logits, new_pkv


class FirstVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    def forward(self, input_ids):
        op = self.model(input_ids=input_ids, use_cache=True)
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SecondVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )

    def forward(
        self,
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        i12,
        i13,
        i14,
        i15,
        i16,
        i17,
        i18,
        i19,
        i20,
        i21,
        i22,
        i23,
        i24,
        i25,
        i26,
        i27,
        i28,
        i29,
        i30,
        i31,
        i32,
        i33,
        i34,
        i35,
        i36,
        i37,
        i38,
        i39,
        i40,
        i41,
        i42,
        i43,
        i44,
        i45,
        i46,
        i47,
        i48,
        i49,
        i50,
        i51,
        i52,
        i53,
        i54,
        i55,
        i56,
        i57,
        i58,
        i59,
        i60,
        i61,
        i62,
        i63,
        i64,
    ):
        # input_ids = input_tuple[0]
        # input_tuple = torch.unbind(pkv, dim=0)
        token = i0
        past_key_values = (
            (i1, i2),
            (
                i3,
                i4,
            ),
            (
                i5,
                i6,
            ),
            (
                i7,
                i8,
            ),
            (
                i9,
                i10,
            ),
            (
                i11,
                i12,
            ),
            (
                i13,
                i14,
            ),
            (
                i15,
                i16,
            ),
            (
                i17,
                i18,
            ),
            (
                i19,
                i20,
            ),
            (
                i21,
                i22,
            ),
            (
                i23,
                i24,
            ),
            (
                i25,
                i26,
            ),
            (
                i27,
                i28,
            ),
            (
                i29,
                i30,
            ),
            (
                i31,
                i32,
            ),
            (
                i33,
                i34,
            ),
            (
                i35,
                i36,
            ),
            (
                i37,
                i38,
            ),
            (
                i39,
                i40,
            ),
            (
                i41,
                i42,
            ),
            (
                i43,
                i44,
            ),
            (
                i45,
                i46,
            ),
            (
                i47,
                i48,
            ),
            (
                i49,
                i50,
            ),
            (
                i51,
                i52,
            ),
            (
                i53,
                i54,
            ),
            (
                i55,
                i56,
            ),
            (
                i57,
                i58,
            ),
            (
                i59,
                i60,
            ),
            (
                i61,
                i62,
            ),
            (
                i63,
                i64,
            ),
        )
        op = self.model(
            input_ids=token, use_cache=True, past_key_values=past_key_values
        )
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class ShardedVicunaModel(torch.nn.Module):
    def __init__(self, model, layers0, layers1):
        super().__init__()
        self.model = model
        assert len(layers0) == len(model.model.layers)
        # self.model.model.layers = torch.nn.modules.container.ModuleList(layers0)
        self.model.model.config.use_cache = True
        self.model.model.config.output_attentions = False
        self.layers0 = layers0
        self.layers1 = layers1

    def forward(
        self,
        input_ids,
        is_first=True,
        past_key_values=None,
        attention_mask=None,
    ):
        if is_first:
            self.model.model.layers = torch.nn.modules.container.ModuleList(
                self.layers0
            )
            return self.model.forward(input_ids, attention_mask=attention_mask)
        else:
            self.model.model.layers = torch.nn.modules.container.ModuleList(
                self.layers1
            )
            return self.model.forward(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )


def write_in_dynamic_inputs0(module, dynamic_input_size):
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


def write_in_dynamic_inputs1(module, dynamic_input_size):
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
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim_42)", line)
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
    mlir_bytecode = get_torch_mlir_module_bytecode(vicuna_layer, model_inputs)
    return mlir_bytecode


def get_model_and_tokenizer(path="TheBloke/vicuna-7B-1.1-HF"):
    kwargs = {"torch_dtype": torch.float}
    vicuna_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    return vicuna_model, tokenizer


def get_tokenizer(path="TheBloke/vicuna-7B-1.1-HF"):
    kwargs = {"torch_dtype": torch.float}
    vicuna_model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    return vicuna_model, tokenizer


def compile_to_vmfb(inputs, layers, is_first=True):
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
                ts_g = compile_vicuna_layer(
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
                ts_g = compile_vicuna_layer(
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
                module = write_in_dynamic_inputs0(str(module), 137)
                bytecode = module.encode("UTF-8")
                bytecode_stream = BytesIO(bytecode)
                bytecode = bytecode_stream.read()

            else:
                module = write_in_dynamic_inputs1(str(module), 138)
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


def get_sharded_model():
    # SAMPLE_INPUT_LEN is used for creating mlir with dynamic inputs, which is currently an increadibly hacky proccess
    # please don't change it
    SAMPLE_INPUT_LEN = 137
    vicuna_model = get_model_and_tokenizer()[0]

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

    layers0 = [FirstVicunaLayer(layer) for layer in vicuna_model.model.layers]
    _, modules0 = compile_to_vmfb(placeholder_input0, layers0, is_first=True)
    shark_layers0 = [CompiledFirstVicunaLayer(m) for m in modules0]

    layers1 = [SecondVicunaLayer(layer) for layer in vicuna_model.model.layers]
    _, modules1 = compile_to_vmfb(placeholder_input1, layers1, is_first=False)
    shark_layers1 = [CompiledSecondVicunaLayer(m) for m in modules1]

    sharded_model = ShardedVicunaModel(
        vicuna_model, shark_layers0, shark_layers1
    )
    return sharded_model


def generate(
    sharded_model, tokenizer, past_key_values, prompt, max_response_len
):
    new_sentence = []
    input_ids = tokenizer(prompt).input_ids
    for iteration in range(max_response_len):
        original_input_ids = input_ids
        input_id_len = len(input_ids)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.reshape([1, input_id_len])

        if iteration == 0:
            output = sharded_model.forward(input_ids, is_first=True)
        else:
            output = sharded_model.forward(
                input_ids, past_key_values=past_key_values, is_first=False
            )
        logits = output["logits"]
        past_key_values = output["past_key_values"]
        new_token = int(torch.argmax(logits[:, -1, :], dim=1)[0])
        if new_token == 2:
            break
        new_sentence += [new_token]
        tokens.append(new_token)
        original_input_ids.append(new_token)
        input_ids = [new_token]

    for i in range(len(tokens)):
        if type(tokens[i]) != int:
            tokens[i] = int(tokens[i][0])
    return new_sentence


def generate_new_token(shark_model, tokenizer, params):
    # TODO : Add warnings and exits for improper params
    input_ids = params["input_ids"]
    iteration = params["iteration"]
    past_key_values = params["past_key_values"]

    input_id_len = len(input_ids)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.reshape([1, input_id_len])

    if iteration == 0:
        output = shark_model.forward(input_ids, is_first=True)
    else:
        output = shark_model.forward(
            input_ids, past_key_values=past_key_values, is_first=False
        )
    logits = output["logits"]
    past_key_values = output["past_key_values"]
    new_token = int(torch.argmax(logits[:, -1, :], dim=1)[0])
    detok = tokenizer.decode(new_token)

    ret_dict = {
        "new_token": new_token,
        "detok": detok,
        "past_key_values": past_key_values,
    }
    return ret_dict


if __name__ == "__main__":
    prompt_history = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    prologue_prompt = "ASSISTANT:\n"
    sharded_model = get_sharded_model()
    tokenizer = get_model_and_tokenizer()[1]
    past_key_values = None
    while True:
        print("\n\n")
        user_prompt = input("User: ")
        prompt_history = (
            prompt_history + "USER:\n" + user_prompt + prologue_prompt
        )
        prompt = prompt_history.strip()
        max_response_len = 1000
        input_ids = tokenizer(prompt).input_ids
        tokens = input_ids
        print("Robot:", end=" ")
        new_sentence = []
        past_key_values = None  # for iteration 0
        for iteration in range(max_response_len):
            original_input_ids = input_ids

            params = {
                "past_key_values": past_key_values,
                "input_ids": input_ids,
                "iteration": iteration,
            }
            generated_token_op = generate_new_token(
                sharded_model, tokenizer, params
            )
            # extract result from generate new token
            new_token = generated_token_op["new_token"]
            detok = generated_token_op["detok"]
            past_key_values = generated_token_op["past_key_values"]

            if new_token == 2:
                break
            new_sentence += [new_token]
            tokens.append(new_token)
            if detok == "<0x0A>":
                print("\n", end="", flush=True)
            else:
                print(f"{detok}", end=" ", flush=True)
            original_input_ids.append(new_token)
            input_ids = [new_token]

        for i in range(len(tokens)):
            if type(tokens[i]) != int:
                tokens[i] = int(tokens[i][0])
        new_sentence_str = tokenizer.decode(new_sentence)
        print(
            "\n-----\nRobot: Here's the complete formatted reply:\n",
            new_sentence_str,
        )
        prompt_history += f"\n{new_sentence_str}\n"
