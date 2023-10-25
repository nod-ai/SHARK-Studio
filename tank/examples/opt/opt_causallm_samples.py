import argparse
import os

import opt_causallm
import opt_util

from shark.shark_inference import SharkInference
from transformers import AutoTokenizer, OPTForCausalLM


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
        vmfb_path = opt_causallm.create_module(
            args.model_name, tokenizer, "cpu-task", args
        )
        opt_shark_module.load_module(vmfb_path)

    for prompt in opt_util.PROMPTS:
        print("\n\nprompt: {}".format(prompt))
        response = opt_causallm.generate_tokens(
            opt_shark_module,
            tokenizer,
            prompt,
            args.max_seq_len,
            print_intermediate_results=False,
        )
        print("reponse: {}".format("".join(response)))
