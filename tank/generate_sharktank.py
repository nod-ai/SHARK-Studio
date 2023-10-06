# Lint as: python3
"""SHARK Tank"""
# python generate_sharktank.py, you have to give a csv tile with [model_name, model_download_url]
# will generate local shark tank folder like this:
#   /SHARK
#     /gen_shark_tank
#       /albert_lite_base
#       /...model_name...
#

import os
import csv
import argparse
from shark.shark_importer import SharkImporter
import subprocess as sp
import hashlib
import numpy as np
from pathlib import Path


def create_hash(file_name):
    with open(file_name, "rb") as f:
        file_hash = hashlib.blake2b(digest_size=64)
        while chunk := f.read(2**10):
            file_hash.update(chunk)

    return file_hash.hexdigest()


def save_torch_model(torch_model_list, local_tank_cache, import_args):
    from tank.model_utils import (
        get_hf_model,
        get_hf_seq2seq_model,
        get_hf_causallm_model,
        get_vision_model,
        get_hf_img_cls_model,
        get_fp16_model,
    )
    from shark.shark_importer import import_with_fx

    with open(torch_model_list) as csvfile:
        torch_reader = csv.reader(csvfile, delimiter=",")
        fields = next(torch_reader)
        for row in torch_reader:
            torch_model_name = row[0]
            tracing_required = row[1]
            model_type = row[2]
            is_dynamic = row[3]
            mlir_type = row[4]
            is_decompose = row[5]

            tracing_required = False if tracing_required == "False" else True
            is_dynamic = False if is_dynamic == "False" else True
            print("generating artifacts for: " + torch_model_name)
            model = None
            input = None
            if model_type == "vision":
                model, input, _ = get_vision_model(
                    torch_model_name, import_args
                )
            elif model_type == "hf":
                model, input, _ = get_hf_model(torch_model_name, import_args)
            elif model_type == "hf_seq2seq":
                model, input, _ = get_hf_seq2seq_model(
                    torch_model_name, import_args
                )
            elif model_type == "hf_causallm":
                model, input, _ = get_hf_causallm_model(
                    torch_model_name, import_args
                )
            elif model_type == "hf_img_cls":
                model, input, _ = get_hf_img_cls_model(
                    torch_model_name, import_args
                )
            torch_model_name = torch_model_name.replace("/", "_")
            if import_args["batch_size"] > 1:
                print(
                    f"Batch size for this model set to {import_args['batch_size']}"
                )
                torch_model_dir = os.path.join(
                    local_tank_cache,
                    str(torch_model_name)
                    + "_torch"
                    + f"_BS{str(import_args['batch_size'])}",
                )
            else:
                torch_model_dir = os.path.join(
                    local_tank_cache, str(torch_model_name) + "_torch"
                )
            os.makedirs(torch_model_dir, exist_ok=True)

            if is_decompose:
                # Add decomposition to some torch ops
                # TODO add op whitelist/blacklist
                import_with_fx(
                    model,
                    (input,),
                    is_f16=False,
                    f16_input_mask=None,
                    debug=True,
                    training=False,
                    return_str=False,
                    save_dir=torch_model_dir,
                    model_name=torch_model_name,
                    mlir_type=mlir_type,
                    is_dynamic=False,
                    tracing_required=tracing_required,
                )
            else:
                mlir_importer = SharkImporter(
                    model,
                    (input,),
                    frontend="torch",
                )
                mlir_importer.import_debug(
                    is_dynamic=False,
                    tracing_required=tracing_required,
                    dir=torch_model_dir,
                    model_name=torch_model_name,
                    mlir_type=mlir_type,
                )
                # Generate torch dynamic models.
                if is_dynamic:
                    mlir_importer.import_debug(
                        is_dynamic=True,
                        tracing_required=tracing_required,
                        dir=torch_model_dir,
                        model_name=torch_model_name + "_dynamic",
                        mlir_type=mlir_type,
                    )

def check_requirements(frontend):
    import importlib

    has_pkgs = False
    if frontend == "torch":
        tv_spec = importlib.util.find_spec("torchvision")
        has_pkgs = tv_spec is not None

    return has_pkgs


class NoImportException(Exception):
    "Raised when requirements are not met for OTF model artifact generation."
    pass


def gen_shark_files(modelname, frontend, tank_dir, importer_args):
    # If a model's artifacts are requested by shark_downloader but they don't exist in the cloud, we call this function to generate the artifacts on-the-fly.
    # TODO: Add TFlite support.
    import tempfile

    import_args = importer_args
    if check_requirements(frontend):
        torch_model_csv = os.path.join(
            os.path.dirname(__file__), "torch_model_list.csv"
        )
        custom_model_csv = tempfile.NamedTemporaryFile(
            dir=os.path.dirname(__file__),
            delete=True,
        )
        elif frontend == "torch":
            with open(torch_model_csv, mode="r") as src:
                reader = csv.reader(src)
                for row in reader:
                    if row[0] == modelname:
                        target = row
            with open(custom_model_csv.name, mode="w") as trg:
                writer = csv.writer(trg)
                writer.writerow(["modelname", "src"])
                writer.writerow(target)
            save_torch_model(custom_model_csv.name, tank_dir, import_args)
    else:
        raise NoImportException


# Validates whether the file is present or not.
def is_valid_file(arg):
    if not os.path.exists(arg):
        return None
    else:
        return arg


if __name__ == "__main__":
    # Note, all of these flags are overridden by the import of import_args from stable_args.py, flags are duplicated temporarily to preserve functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #    "--torch_model_csv",
    #    type=lambda x: is_valid_file(x),
    #    default="./tank/torch_model_list.csv",
    #    help="""Contains the file with torch_model name and args.
    #         Please see: https://github.com/nod-ai/SHARK/blob/main/tank/torch_model_list.csv""",
    # )
    # parser.add_argument(
    #    "--ci_tank_dir",
    #    type=bool,
    #    default=False,
    # )
    # parser.add_argument("--upload", type=bool, default=False)

    # old_import_args = parser.parse_import_args()
    import_args = {
        "batch_size": 1,
    }
    print(import_args)
    home = str(Path.home())
    WORKDIR = os.path.join(os.path.dirname(__file__), "..", "gen_shark_tank")
    torch_model_csv = os.path.join(
        os.path.dirname(__file__), "torch_model_list.csv"
    )

    save_torch_model(torch_model_csv, WORKDIR, import_args)
