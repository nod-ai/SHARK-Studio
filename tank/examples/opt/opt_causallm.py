import argparse
import os
import torch
import numpy as np
from shark_opt_wrapper import OPTForCausalLMModel
from shark.shark_inference import SharkInference
from shark.shark_importer import import_with_fx
from transformers import AutoTokenizer, OPTForCausalLM
from typing import Iterable


def create_module(model_name, tokenizer, device, args):
    opt_base_model = OPTForCausalLM.from_pretrained(model_name)
    opt_base_model.eval()
    opt_model = OPTForCausalLMModel(opt_base_model)
    encoded_inputs = tokenizer(
        "What is the meaning of life?",
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
    )
    inputs = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    # np.save("model_inputs_0.npy", inputs[0])
    # np.save("model_inputs_1.npy", inputs[1])
    opt_fs_name = "-".join(
        "_".join(args.model_name.split("/")[1].split("-")).split(".")
    )

    mlir_path = f"./{opt_fs_name}_causallm_{args.max_seq_len}_torch.mlir"
    if os.path.isfile(mlir_path):
        print(f"Found .mlir from {mlir_path}")
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
        del model_mlir

    shark_module = SharkInference(
        mlir_path,
        device=device,
        mlir_dialect="tm_tensor",
        is_benchmark=False,
    )

    vmfb_name = f"{opt_fs_name}_causallm_{args.max_seq_len}_torch_cpu"
    shark_module.save_module(module_name=vmfb_name, debug=False)
    vmfb_path = vmfb_name + ".vmfb"
    return vmfb_path


def shouldStop(tokens):
    stop_ids = [50278, 50279, 50277, 0]
    for stop_id in stop_ids:
        if tokens[0][-1] == stop_id:
            return True
    return False


def generate_new_token(shark_module, tokenizer, new_text, max_seq_len: int):
    model_inputs = tokenizer(
        new_text,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    inputs = (
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
    )
    sum_attentionmask = torch.sum(model_inputs.attention_mask)
    output = shark_module("forward", inputs)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=32)
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
        "--recompile",
        help="If set, recompiles MLIR -> .vmfb",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--plugin-path",
        help="path to executable plugin",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print("args={}".format(args))
    return args


def generate_tokens(
    opt_shark_module: "SharkInference",
    tokenizer,
    input_text: str,
    max_output_len: int,
    print_intermediate_results: True,
) -> Iterable[str]:
    words_list = []
    new_text = input_text
    try:
        for _ in range(max_output_len):
            generated_token_op = generate_new_token(
                opt_shark_module, tokenizer, new_text, max_output_len
            )
            detok = generated_token_op["detok"]
            if generated_token_op["stop_generation"]:
                break
            if print_intermediate_results:
                print(detok, end="", flush=True)
            words_list.append(detok)
            if detok == "":
                break
            new_text += detok
    except KeyboardInterrupt as e:
        print("Exiting token generation.")
    return words_list


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    opt_fs_name = "-".join(
        "_".join(args.model_name.split("/")[1].split("-")).split(".")
    )
    vmfb_path = f"./{opt_fs_name}_causallm_{args.max_seq_len}_torch_cpu.vmfb"
    if args.plugin_path is not None:
        rt_flags = [f"--executable_plugin={args.plugin_path}"]
    else:
        rt_flags = []
    opt_shark_module = SharkInference(
        mlir_module=None, device="cpu-task", rt_flags=rt_flags
    )
    if os.path.isfile(vmfb_path):
        opt_shark_module.load_module(vmfb_path)
    else:
        vmfb_path = create_module(args.model_name, tokenizer, "cpu-task", args)
        opt_shark_module.load_module(vmfb_path)
    while True:
        input_text = input("Give me a sentence to complete:")
        generate_tokens(
            opt_shark_module, tokenizer, input_text, args.max_seq_len
        )
