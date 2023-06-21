import argparse
from pathlib import Path
from apps.language_models.src.pipelines import vicuna_pipeline as vp
from apps.language_models.src.pipelines import vicuna_sharded_pipeline as vsp
import torch
import json

if __name__ == "__main__":
    import gc


parser = argparse.ArgumentParser(
    prog="vicuna runner",
    description="runs a vicuna model",
)

parser.add_argument(
    "--precision", "-p", default="fp32", help="fp32, fp16, int8, int4"
)
parser.add_argument("--device", "-d", default="cuda", help="vulkan, cpu, cuda")
parser.add_argument(
    "--first_vicuna_vmfb_path", default=None, help="path to first vicuna vmfb"
)
parser.add_argument(
    "-s",
    "--sharded",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Run model as sharded",
)
# TODO: sharded config

parser.add_argument(
    "--second_vicuna_vmfb_path",
    default=None,
    help="path to second vicuna vmfb",
)
parser.add_argument(
    "--first_vicuna_mlir_path",
    default=None,
    help="path to first vicuna mlir file",
)
parser.add_argument(
    "--second_vicuna_mlir_path",
    default=None,
    help="path to second vicuna mlir",
)

parser.add_argument(
    "--load_mlir_from_shark_tank",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="download precompile mlir from shark tank",
)
parser.add_argument(
    "--cli",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Run model in cli mode",
)

parser.add_argument(
    "--config",
    default=None,
    help="configuration file",
)

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()

    vic = None
    if not args.sharded:
        first_vic_mlir_path = (
            Path(f"first_vicuna_{args.precision}.mlir")
            if args.first_vicuna_mlir_path is None
            else Path(args.first_vicuna_mlir_path)
        )
        second_vic_mlir_path = (
            Path(f"second_vicuna_{args.precision}.mlir")
            if args.second_vicuna_mlir_path is None
            else Path(args.second_vicuna_mlir_path)
        )
        first_vic_vmfb_path = (
            Path(
                f"first_vicuna_{args.precision}_{args.device.replace('://', '_')}.vmfb"
            )
            if args.first_vicuna_vmfb_path is None
            else Path(args.first_vicuna_vmfb_path)
        )
        second_vic_vmfb_path = (
            Path(
                f"second_vicuna_{args.precision}_{args.device.replace('://', '_')}.vmfb"
            )
            if args.second_vicuna_vmfb_path is None
            else Path(args.second_vicuna_vmfb_path)
        )

        vic = vp.Vicuna(
            "vicuna",
            device=args.device,
            precision=args.precision,
            first_vicuna_mlir_path=first_vic_mlir_path,
            second_vicuna_mlir_path=second_vic_mlir_path,
            first_vicuna_vmfb_path=first_vic_vmfb_path,
            second_vicuna_vmfb_path=second_vic_vmfb_path,
            load_mlir_from_shark_tank=args.load_mlir_from_shark_tank,
        )
    else:
        if args.config is not None:
            config_file = open(args.config)
            config_json = json.load(config_file)
            config_file.close()
        else:
            config_json = None
        vic = vsp.Vicuna(
            "vicuna",
            device=args.device,
            precision=args.precision,
            config_json=config_json,
        )
    prompt_history = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    prologue_prompt = "ASSISTANT:\n"

    while True:
        # TODO: Add break condition from user input
        user_prompt = input("User: ")
        prompt_history = (
            prompt_history + "USER:\n" + user_prompt + prologue_prompt
        )
        prompt = prompt_history.strip()
        res_str = vic.generate(prompt, cli=True)
        torch.cuda.empty_cache()
        gc.collect()
        print(
            "\n-----\nAssistant: Here's the complete formatted reply:\n",
            res_str,
        )
        prompt_history += f"\n{res_str}\n"
