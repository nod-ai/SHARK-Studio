import unittest

import os
import pytest
import torch_mlir
import torch
import numpy as np
from shark_hf_opt import OPTForCausalLM
from shark.iree_utils._common import check_device_drivers, device_driver_info
from shark.shark_inference import SharkInference
from tank.model_utils import compare_tensors
from transformers import AutoTokenizer

OPT_MODEL = "opt-350m"
OPT_MODEL_66B = "facebook/opt-66b"
MAX_SEQUENCE_LENGTH = 256
MAX_NEW_TOKENS = 200


def create_module(model_name, tokenizer, device):
    opt_model = OPTForCausalLM.from_pretrained(
        "facebook/" + model_name, return_dict=False
    )
    opt_model.eval()

    encoded_inputs = tokenizer(
        "This is a sample input for generating the model.",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    mlir_path = f"./{OPT_MODEL}_causallm_{MAX_SEQUENCE_LENGTH}_torch.mlir"
    if os.path.isfile(mlir_path):
        with open(mlir_path, "r") as f:
            model_mlir = f.read()
        print(f"Loaded .mlir from {mlir_path}")
    else:
        module = torch_mlir.compile(
            opt_model,
            inputs,
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=True,
        )

        model_mlir = module.operation.get_asm(
            large_elements_limit=None, enable_debug_info=True
        )

        with open(mlir_path, "w") as f:
            f.write(model_mlir)
        print(f"Saved mlir at {mlir_path}")

    func_name = "forward"
    act_out = opt_model(inputs[0], attention_mask=inputs[1], return_dict=False)

    shark_module = SharkInference(
        model_mlir,
        device=device,
        mlir_dialect="tm_tensor",
        is_benchmark=False,
    )
    vmfb_name = f"{OPT_MODEL}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{device}"
    shark_module.save_module(module_name=vmfb_name)
    shark_module.load_module(vmfb_name + ".vmfb")

    results = shark_module("forward", inputs)
    print(
        "SHARK logits have shape: ",
        str(results[0].shape) + " : " + str(results[0]),
    )
    print(
        "PyTorch logits have shape: "
        + str(act_out[0].shape)
        + " : "
        + str(act_out[0])
    )
    # exp_out = tokenizer.decode(act_out[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # shark_out = tokenizer.decode(results[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return shark_module


def shouldStop(tokens):
    stop_ids = [50278, 50279, 50277, 0]
    for stop_id in stop_ids:
        if tokens[0][-1] == stop_id:
            return True
    return False


def generate_new_token(shark_model, tokenizer, new_text):
    model_inputs = tokenizer(
        new_text,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
    )
    sum_attentionmask = torch.sum(model_inputs.attention_mask)
    output = shark_model("forward", inputs)
    output = torch.FloatTensor(output[0])
    next_toks = torch.topk(output, 1)
    stop_generation = False
    if shouldStop(next_toks.indices):
        stop_generation = True
    new_token = next_toks.indices[int(sum_attentionmask) - 1]
    detok = tokenizer.decode(
        new_token,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    ret_dict = {
        "new_token": new_token,
        "detok": detok,
        "stop_generation": stop_generation,
    }
    return ret_dict


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/" + OPT_MODEL, use_fast=False
    )
    vmfb_path = f"./{OPT_MODEL}_causallm_{MAX_SEQUENCE_LENGTH}_torch_cpu.vmfb"
    if os.path.isfile(vmfb_path):
        opt_shark_module = SharkInference(mlir_module=None, device="cpu")
        opt_shark_module.load_module(vmfb_path)
    else:
        opt_shark_module = create_module(OPT_MODEL, tokenizer, "cpu")
    while True:
        try:
            new_text = input("Give me a sentence to complete:")
            words_list = []

            for i in range(MAX_NEW_TOKENS):
                generated_token_op = generate_new_token(
                    opt_shark_module, tokenizer, new_text
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

        except KeyboardInterrupt:
            print("Exiting program.")
            break
