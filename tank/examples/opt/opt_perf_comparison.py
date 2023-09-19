"""
Script for comparing OPT model performance between SHARK and Huggingface
PyTorch.

Usage Example:

python opt_perf_comparison.py --max-seq-len=32 --model-name=facebook/opt-125m \
        --platform=shark

python opt_perf_comparison.py --max-seq-len=512 --model-name=facebook/opt-1.3b \
        --platform=shark

See parse_args() below for command line argument usage.
"""

import argparse
import collections
import json
import os
import psutil
import resource
import time
from typing import Tuple

from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM
from shark_opt_wrapper import OPTForCausalLMModel

DEVICE = "cpu"
PLATFORM_SHARK = "shark"
PLATFORM_HUGGINGFACE = "huggingface"

# Dict keys for reports.
REPORT_PLATFORM = "platform"
REPORT_MODEL_NAME = "model"
REPORT_MAX_SEQ_LEN = "max_seq_len"
REPORT_LOAD_TIME = "load_time_sec"
REPORT_RUN_TIME = "run_time_sec"
REPORT_LOAD_PHYSICAL_MEMORY_MB = "load_physical_MB"
REPORT_LOAD_VIRTUAL_MEMORY_MB = "load_virtual_MB"
REPORT_RUN_PHYSICAL_MEMORY_MB = "run_physical_MB"
REPORT_RUN_VIRTUAL_MEMORY_MB = "run_virtual_MB"

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


def get_memory_info():
    pid = os.getpid()
    process = psutil.Process(pid)
    return process.memory_info()


def create_vmfb_module(
    model_name: str,
    tokenizer,
    device: str,
    max_seq_len: int,
    recompile_shark: bool,
):
    opt_base_model = OPTForCausalLM.from_pretrained(model_name)
    opt_base_model.eval()
    opt_model = OPTForCausalLMModel(opt_base_model)
    encoded_inputs = tokenizer(
        PROMPTS[0],
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    # np.save("model_inputs_0.npy", inputs[0])
    # np.save("model_inputs_1.npy", inputs[1])

    opt_fs_name = get_opt_fs_name(model_name)
    mlir_path = f"./{opt_fs_name}_causallm_{max_seq_len}_torch.mlir"
    # If MLIR has already been loaded and recompilation is not requested, use
    # the loaded MLIR file.
    has_mlir = os.path.isfile(mlir_path)
    # The purpose of recompile_shark is to measure compilation time; the
    # compilation time can be correctly measured only when MLIR has already been
    # loaded.
    assert not recompile_shark or has_mlir
    if has_mlir:
        with open(mlir_path, "r") as f:
            model_mlir = f.read()
        print(f"Loaded .mlir from {mlir_path}")
    else:
        (model_mlir, func_name) = import_with_fx(
            model=opt_model,
            inputs=inputs,
            is_f16=False,
            model_name=opt_fs_name,
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

    vmfb_name = (
        f"{opt_fs_name}_causallm_{max_seq_len}_torch_{DEVICE}_tiled_ukernels"
    )
    shark_module.save_module(module_name=vmfb_name)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


def load_shark_model(
    model_name: str, max_seq_len: int, recompile_shark: bool
) -> ModelWrapper:
    opt_fs_name = get_opt_fs_name(model_name)
    vmfb_name = f"{opt_fs_name}_causallm_{max_seq_len}_torch_{DEVICE}_tiled_ukernels.vmfb"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if recompile_shark or not os.path.isfile(vmfb_name):
        print(f"vmfb not found. compiling and saving to {vmfb_name}")
        create_vmfb_module(
            model_name, tokenizer, DEVICE, max_seq_len, recompile_shark
        )
    shark_module = SharkInference(mlir_module=None, device="cpu-task")
    shark_module.load_module(vmfb_name)
    return ModelWrapper(model=shark_module, tokenizer=tokenizer)


def run_shark_model(model_wrapper: ModelWrapper, tokens):
    # Generate logits output of OPT model.
    return model_wrapper.model("forward", tokens)


def load_huggingface_model(model_name: str) -> ModelWrapper:
    return ModelWrapper(
        model=OPTForCausalLM.from_pretrained(model_name),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
    )


def run_huggingface_model(model_wrapper: ModelWrapper, tokens):
    return model_wrapper.model.forward(
        tokens.input_ids, tokens.attention_mask, return_dict=False
    )


def save_json(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)


def collect_huggingface_logits(
    model_name: str, max_seq_len: int, to_save_json: bool
) -> Tuple[float, float]:
    # Load
    t0 = time.time()
    model_wrapper = load_huggingface_model(model_name)
    load_time = time.time() - t0
    print("--- Took {} seconds to load Huggingface.".format(load_time))
    load_memory_info = get_memory_info()

    results = []
    tokenized_prompts = []
    for prompt in PROMPTS:
        tokens = model_wrapper.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_prompts.append(tokens)

    # Run
    t0 = time.time()
    for idx, tokens in enumerate(tokenized_prompts):
        print("prompt: {}".format(PROMPTS[idx]))
        logits = run_huggingface_model(model_wrapper, tokens)
        if to_save_json:
            results.append([PROMPTS[idx], logits[0].tolist()])
    run_time = time.time() - t0
    print("--- Took {} seconds to run Huggingface.".format(run_time))
    if to_save_json:
        save_json(results, "/tmp/huggingface.json")
    run_memory_info = get_memory_info()
    return {
        REPORT_PLATFORM: PLATFORM_HUGGINGFACE,
        REPORT_MODEL_NAME: model_name,
        REPORT_MAX_SEQ_LEN: max_seq_len,
        REPORT_LOAD_TIME: load_time,
        REPORT_RUN_TIME: run_time / len(PROMPTS),
        REPORT_LOAD_PHYSICAL_MEMORY_MB: load_memory_info.rss >> 20,
        REPORT_LOAD_VIRTUAL_MEMORY_MB: load_memory_info.vms >> 20,
        REPORT_RUN_PHYSICAL_MEMORY_MB: run_memory_info.rss >> 20,
        REPORT_RUN_VIRTUAL_MEMORY_MB: run_memory_info.vms >> 20,
    }


def collect_shark_logits(
    model_name: str,
    max_seq_len: int,
    recompile_shark: bool,
    to_save_json: bool,
) -> Tuple[float, float]:
    # Load
    t0 = time.time()
    model_wrapper = load_shark_model(model_name, max_seq_len, recompile_shark)
    load_time = time.time() - t0
    print("--- Took {} seconds to load Shark.".format(load_time))
    load_memory_info = get_memory_info()

    results = []
    tokenized_prompts = []
    for prompt in PROMPTS:
        tokens = model_wrapper.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        inputs = (
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        tokenized_prompts.append(inputs)

    # Run
    t0 = time.time()
    for idx, tokens in enumerate(tokenized_prompts):
        print("prompt: {}".format(PROMPTS[idx]))
        logits = run_shark_model(model_wrapper, tokens)
        lst = [e.tolist() for e in logits]
        if to_save_json:
            results.append([PROMPTS[idx], lst])
    run_time = time.time() - t0
    print("--- Took {} seconds to run Shark.".format(run_time))
    if to_save_json:
        save_json(results, "/tmp/shark.json")
    platform_postfix = "-compile" if recompile_shark else "-precompiled"
    run_memory_info = get_memory_info()
    return {
        REPORT_PLATFORM: PLATFORM_SHARK + platform_postfix,
        REPORT_MODEL_NAME: model_name,
        REPORT_MAX_SEQ_LEN: max_seq_len,
        REPORT_LOAD_TIME: load_time,
        REPORT_RUN_TIME: run_time / len(PROMPTS),
        REPORT_LOAD_PHYSICAL_MEMORY_MB: load_memory_info.rss >> 20,
        REPORT_LOAD_VIRTUAL_MEMORY_MB: load_memory_info.vms >> 20,
        REPORT_RUN_PHYSICAL_MEMORY_MB: run_memory_info.rss >> 20,
        REPORT_RUN_VIRTUAL_MEMORY_MB: run_memory_info.vms >> 20,
    }


def get_opt_fs_name(model_name: str) -> str:
    """Cleanses the model name ino a file system-friendly name.

    Example: get_opt_fs_name('facebook/opt-1.3b') == 'opt_1-3b'
    """
    slash_split = model_name.split("/")
    assert 1 <= len(slash_split) <= 2, "There should be at most one slash."
    model_name = slash_split[-1]
    for src_pattern, dest_pattern in (("-", "_"), (".", "-")):
        model_name = model_name.replace(src_pattern, dest_pattern)
    return model_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-json",
        help="If set, saves output JSON.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--max-seq-len", help="Max sequence length", type=int, default=32
    )
    parser.add_argument(
        "--model-name",
        help="Model name",
        type=str,
        choices=[
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-6.7b",
        ],
        default="facebook/opt-1.3b",
    )
    parser.add_argument(
        "--recompile-shark",
        help="If set, recompiles MLIR",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--platform",
        help="Either shark or huggingface",
        type=str,
        choices=[PLATFORM_SHARK, PLATFORM_HUGGINGFACE],
        default=PLATFORM_SHARK,
    )
    args = parser.parse_args()
    print("args={}".format(args))
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.platform == PLATFORM_SHARK:
        shark_report = collect_shark_logits(
            args.model_name,
            args.max_seq_len,
            args.recompile_shark,
            args.save_json,
        )
        print("# Summary: {}".format(json.dumps(shark_report)))
    else:
        huggingface_report = collect_huggingface_logits(
            args.model_name, args.max_seq_len, args.save_json
        )
        print("# Summary: {}".format(json.dumps(huggingface_report)))
