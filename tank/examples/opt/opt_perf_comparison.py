import collections
import json
import time
import os

from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM
from shark_opt_wrapper import OPTForCausalLMModel

MODEL_NAME = "facebook/opt-1.3b"
OPT_MODELNAME = "opt-1.3b"
OPT_FS_NAME = "opt_1-3b"
MAX_SEQUENCE_LENGTH = 8
DEVICE = "cpu"

PROMPTS = [
    "What is the meaning of life?",
    "Tell me something you don't know.",
    "What does Xilinx do?",
    "What is the mass of earth?",
    "What is a poem?",
    "What is recursion?",
    "Tell me a one line joke.",
    "Who is Gilgamesh?",
    "Tell me something about cryptocurrency.",
    "How did it all begin?",
]

ModelWrapper = collections.namedtuple("ModelWrapper", ["model", "tokenizer"])


def create_vmfb_module(model_name, tokenizer, device):
    opt_base_model = OPTForCausalLM.from_pretrained("facebook/" + model_name)
    opt_base_model.eval()
    opt_model = OPTForCausalLMModel(opt_base_model)
    encoded_inputs = tokenizer(
        "What is the meaning of life?",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    # np.save("model_inputs_0.npy", inputs[0])
    # np.save("model_inputs_1.npy", inputs[1])

    mlir_path = f"./{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch.mlir"
    if os.path.isfile(mlir_path):
        with open(mlir_path, "r") as f:
            model_mlir = f.read()
        print(f"Loaded .mlir from {mlir_path}")
    else:
        (model_mlir, func_name) = import_with_fx(
            model=opt_model,
            inputs=inputs,
            is_f16=False,
            model_name=OPT_FS_NAME,
            return_str=True,
        )
        with open(mlir_path, "w") as f:
            f.write(model_mlir)
        print(f"Saved mlir at {mlir_path}")

    shark_module = SharkInference(
        model_mlir,
        device=device,
        mlir_dialect="tm_tensor",
        is_benchmark=False,
    )

    vmfb_name = f"{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{DEVICE}"
    shark_module.save_module(module_name=vmfb_name)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


def load_shark_model() -> ModelWrapper:
    vmfb_name = (
        f"{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{DEVICE}.vmfb"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if not os.path.isfile(vmfb_name):
        print(f"vmfb not found. compiling and saving to {vmfb_name}")
        create_vmfb_module(OPT_MODELNAME, tokenizer, DEVICE)
    shark_module = SharkInference(mlir_module=None, device="cpu-task")
    shark_module.load_module(vmfb_name)
    return ModelWrapper(model=shark_module, tokenizer=tokenizer)


def run_shark_model(model_wrapper: ModelWrapper, prompt: str):
    model_inputs = model_wrapper.tokenizer(
        prompt,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
    )
    # Generate logits output of OPT model.
    return model_wrapper.model("forward", inputs)


def run_shark():
    model_wrapper = load_shark_model()

    prompt = "What is the meaning of life?"
    logits = run_shark_model(model_wrapper, prompt)

    # Print output logits to validate vs. pytorch + base transformers
    print(logits[0])


def load_huggingface_model() -> ModelWrapper:
    return ModelWrapper(
        model=OPTForCausalLM.from_pretrained(MODEL_NAME),
        tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    )


def run_huggingface_model(model_wrapper: ModelWrapper, prompt: str):
    inputs = model_wrapper.tokenizer(prompt, return_tensors="pt")
    return model_wrapper.model.forward(
        inputs.input_ids, inputs.attention_mask, return_dict=False
    )


def run_huggingface():
    model_wrapper = load_huggingface_model()

    prompt = "What is the meaning of life?"
    logits = run_huggingface_model(model_wrapper, prompt)

    print(logits[0])


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)


def collect_huggingface_logits():
    t0 = time.time()
    model_wrapper = load_huggingface_model()
    print("--- Took {} seconds to load Huggingface.".format(time.time() - t0))
    results = []
    t0 = time.time()
    for prompt in PROMPTS:
        print("prompt: {}".format(prompt))
        logits = run_huggingface_model(model_wrapper, prompt)
        results.append([prompt, logits[0].tolist()])
    print("--- Took {} seconds to run Huggingface.".format(time.time() - t0))
    save_json(results, "/tmp/huggingface.json")


def collect_shark_logits():
    t0 = time.time()
    model_wrapper = load_shark_model()
    print("--- Took {} seconds to load Shark.".format(time.time() - t0))
    results = []
    t0 = time.time()
    for prompt in PROMPTS:
        print("prompt: {}".format(prompt))
        logits = run_shark_model(model_wrapper, prompt)
        lst = [e.tolist() for e in logits]
        results.append([prompt, lst])
    print("--- Took {} seconds to run Shark.".format(time.time() - t0))
    save_json(results, "/tmp/shark.json")


if __name__ == "__main__":
    collect_shark_logits()
    collect_huggingface_logits()
