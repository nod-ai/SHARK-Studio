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
from transformers import AutoTokenizer, top_k_top_p_filtering

OPT_MODEL = "opt-350m"
OPT_MODEL_66B = "facebook/opt-66b"
MAX_SEQUENCE_LENGTH = 256


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
    shark_module.compile()
    shark_module.save_module(module_name=f"{OPT_MODEL}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{device}")
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

def append_output_logits(logits, inputs):
    top_logits = top_k_top_p_filtering(torch.tensor(logits), top_k=256, top_p=1.0)
    probs = torch.nn.functional.softmax(top_logits, dim=-1)
    generated_next_token = torch.multinomial(probs, num_samples=1)
    for idx, x in enumerate(inputs[0]):
        if idx > 1 and x == 1:
            inputs[0][idx] = generated_next_token[0]
            break
        else:
            continue
    return response[0]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("facebook/" + OPT_MODEL, use_fast=False)
    vmfb_path = f"./{OPT_MODEL}_causallm_{MAX_SEQUENCE_LENGTH}_torch_cpu.vmfb"
    if os.path.isfile(vmfb_path):
        opt_shark_module = SharkInference(mlir_module=None, device="cpu")
        opt_shark_module.load_module(vmfb_path)
    else:
    	opt_shark_module = create_module(OPT_MODEL, tokenizer, "cpu")
    while True:
        try:
            new_text = input("Give me a sentence to complete:")
            encoded_inputs = tokenizer(
                new_text,
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
            )
            inputs = (
                encoded_inputs["input_ids"],
                encoded_inputs["attention_mask"],
            )
            response = inputs
            for i in range(10):
                token_logits = opt_shark_module("forward", response)[0]
                response[0][0] = append_output_logits(token_logits, response[0])
                
            print(torch.tensor(response[0]))
            print(tokenizer.decode(response[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        except KeyboardInterrupt:
            print("Exiting program.")
            break
