import torch
import torch_mlir
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
)
from io import BytesIO
from pathlib import Path
from apps.language_models.utils import (
    get_torch_mlir_module_bytecode,
    get_vmfb_from_path,
)


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


def compile_stableLM(
    model,
    model_inputs,
    model_name,
    model_vmfb_name,
    device="cuda",
    precision="fp32",
):
    from shark.shark_inference import SharkInference

    # device = "cuda"  # "cpu"
    # TODO: vmfb and mlir name should include precision and device
    vmfb_path = (
        Path(model_name + f"_{device}.vmfb")
        if model_vmfb_name is None
        else Path(model_vmfb_name)
    )
    shark_module = get_vmfb_from_path(
        vmfb_path, device, mlir_dialect="tm_tensor"
    )
    if shark_module is not None:
        return shark_module

    mlir_path = Path(model_name + ".mlir")
    print(
        f"[DEBUG] mlir path {mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
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

    path = shark_module.save_module(
        vmfb_path.parent.absolute(), vmfb_path.stem
    )
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
    print("Sucessfully loaded the tokenizer to the memory")
    return tok


# sharkStableLM = compile_stableLM
# (
#   None,
#   tuple([input_ids, attention_mask]),
#   "stableLM_linalg_f32_seqLen256",
#   "/home/shark/vivek/stableLM_shark_f32_seqLen256"
# )
def generate(
    new_text,
    max_new_tokens,
    sharkStableLM,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = get_tokenizer()
    # Construct the input message string for the model by
    # concatenating the current system message and conversation history
    # Tokenize the messages string
    # sharkStableLM = compile_stableLM
    # (
    #   None,
    #   tuple([input_ids, attention_mask]),
    #   "stableLM_linalg_f32_seqLen256",
    #   "/home/shark/vivek/stableLM_shark_f32_seqLen256"
    # )
    words_list = []
    for i in range(max_new_tokens):
        # numWords = len(new_text.split())
        # if(numWords>220):
        #  break
        params = {
            "new_text": new_text,
        }
        generated_token_op = generate_new_token(
            sharkStableLM, tokenizer, params
        )
        detok = generated_token_op["detok"]
        stop_generation = generated_token_op["stop_generation"]
        if stop_generation:
            break
        print(detok, end="", flush=True)
        words_list.append(detok)
        if detok == "":
            break
        new_text = new_text + detok
    return words_list


def generate_new_token(shark_model, tokenizer, params):
    new_text = params["new_text"]
    model_inputs = tokenizer(
        [new_text],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    sum_attentionmask = torch.sum(model_inputs.attention_mask)
    # sharkStableLM = compile_stableLM(None, tuple([input_ids, attention_mask]), "stableLM_linalg_f32_seqLen256", "/home/shark/vivek/stableLM_shark_f32_seqLen256")
    output = shark_model(
        "forward", [model_inputs.input_ids, model_inputs.attention_mask]
    )
    output = torch.from_numpy(output)
    next_toks = torch.topk(output, 1)
    stop_generation = False
    if shouldStop(next_toks.indices):
        stop_generation = True
    new_token = next_toks.indices[0][int(sum_attentionmask) - 1]
    detok = tokenizer.decode(
        new_token,
        skip_special_tokens=True,
    )
    ret_dict = {
        "new_token": new_token,
        "detok": detok,
        "stop_generation": stop_generation,
    }
    return ret_dict
