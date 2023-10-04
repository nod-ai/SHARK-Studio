import os
import torch
import numpy as np
from shark_opt_wrapper import OPTForCausalLMModel
from shark.iree_utils._common import (
    check_device_drivers,
    device_driver_info,
)
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM

OPT_MODEL = "opt-1.3b"
OPT_FS_NAME = "opt-1_3b"
MAX_SEQUENCE_LENGTH = 128
MAX_NEW_TOKENS = 60


def create_module(model_name, tokenizer, device):
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
        print(f"Found .mlir from {mlir_path}")
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
        del model_mlir

    shark_module = SharkInference(
        mlir_path,
        device=device,
        mlir_dialect="tm_tensor",
        is_benchmark=False,
    )

    vmfb_name = f"{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch_{device}"
    shark_module.save_module(module_name=vmfb_name, debug=False)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


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
    vmfb_path = (
        f"./{OPT_FS_NAME}_causallm_{MAX_SEQUENCE_LENGTH}_torch_cpu-task.vmfb"
    )
    opt_shark_module = SharkInference(mlir_module=None, device="cpu-task")
    if os.path.isfile(vmfb_path):
        opt_shark_module.load_module(vmfb_path)
    else:
        vmfb_path = create_module(OPT_MODEL, tokenizer, "cpu-task")
        opt_shark_module.load_module(vmfb_path)
    while True:
        try:
            new_text = input("Give me a sentence to complete:")
            new_text_init = new_text
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
