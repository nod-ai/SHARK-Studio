import warnings

warnings.filterwarnings("ignore")
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
        input_ids = tokenizer(prompt).input_ids
        tokens = input_ids
        prompt = print("Robot:", end=" ")
        new_sentence = []
        max_response_len = 1000
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
            detok = tokenizer.decode(new_token)
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
