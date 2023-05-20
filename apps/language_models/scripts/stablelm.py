import torch
import torch_mlir
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
)
from torch.nn import functional as F
from io import BytesIO
from pathlib import Path
from apps.language_models.utils import get_torch_mlir_module_bytecode


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
