# Lint as: python3
"""SHARK Downloader"""
# Requirements : Put shark_tank in SHARK directory
#   /SHARK
#     /gen_shark_tank
#       /tflite
#         /albert_lite_base
#         /...model_name...
#       /tf
#       /pytorch
#
#
#

import numpy as np
import os
import urllib.request
import json

input_type_to_np_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
}

WORKDIR = os.path.join(os.path.dirname(__file__), "./../gen_shark_tank")

# Checks whether the directory and files exists.
def check_dir_exists(model_name, frontend="torch", dynamic=""):
    model_dir = os.path.join(WORKDIR, model_name)

    # Remove the _tf keyword from end.
    if frontend in ["tf", "tensorflow"]:
        model_name = model_name[:-3]

    if os.path.isdir(model_dir):
        if (
            os.path.isfile(
                os.path.join(model_dir, model_name + dynamic + ".mlir")
            )
            and os.path.isfile(os.path.join(model_dir, "function_name.npy"))
            and os.path.isfile(os.path.join(model_dir, "inputs.npz"))
            and os.path.isfile(os.path.join(model_dir, "golden_out.npz"))
        ):
            print(
                f"""The models are present in the {WORKDIR}. If you want a fresh 
                download, consider deleting the directory."""
            )
            return True
    return False


# Downloads the torch model from gs://shark_tank dir.
def download_torch_model(model_name, dynamic=False):
    model_name = model_name.replace("/", "_")
    dyn_str = "_dynamic" if dynamic else ""
    os.makedirs(WORKDIR, exist_ok=True)
    if not check_dir_exists(model_name, dyn_str):
        gs_command = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp -r gs://shark_tank'
            + "/"
            + model_name
            + " "
            + WORKDIR
        )
        if os.system(gs_command) != 0:
            raise Exception("model not present in the tank. Contact Nod Admin")

    model_dir = os.path.join(WORKDIR, model_name)
    with open(os.path.join(model_dir, model_name + dyn_str + ".mlir")) as f:
        mlir_file = f.read()

    function_name = str(np.load(os.path.join(model_dir, "function_name.npy")))
    inputs = np.load(os.path.join(model_dir, "inputs.npz"))
    golden_out = np.load(os.path.join(model_dir, "golden_out.npz"))

    inputs_tuple = tuple([inputs[key] for key in inputs])
    golden_out_tuple = tuple([golden_out[key] for key in golden_out])
    return mlir_file, function_name, inputs_tuple, golden_out_tuple


# Downloads the tflite model from gs://shark_tank dir.
def download_tflite_model(model_name, dynamic=False):
    dyn_str = "_dynamic" if dynamic else ""
    os.makedirs(WORKDIR, exist_ok=True)
    if not check_dir_exists(model_name, dyn_str):
        gs_command = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp -r gs://shark_tank'
            + "/"
            + model_name
            + " "
            + WORKDIR
        )
        if os.system(gs_command) != 0:
            raise Exception("model not present in the tank. Contact Nod Admin")

    model_dir = os.path.join(WORKDIR, model_name)
    with open(
        os.path.join(model_dir, model_name + dyn_str + "_tflite.mlir")
    ) as f:
        mlir_file = f.read()

    function_name = str(np.load(os.path.join(model_dir, "function_name.npy")))
    inputs = np.load(os.path.join(model_dir, "inputs.npz"))
    golden_out = np.load(os.path.join(model_dir, "golden_out.npz"))

    inputs_tuple = tuple([inputs[key] for key in inputs])
    golden_out_tuple = tuple([golden_out[key] for key in golden_out])
    return mlir_file, function_name, inputs_tuple, golden_out_tuple


def download_tf_model(model_name):
    model_name = model_name.replace("/", "_")
    os.makedirs(WORKDIR, exist_ok=True)
    model_dir_name = model_name + "_tf"
    if not check_dir_exists(model_dir_name, frontend="tf"):
        gs_command = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp -r gs://shark_tank'
            + "/"
            + model_dir_name
            + " "
            + WORKDIR
        )
        if os.system(gs_command) != 0:
            raise Exception("model not present in the tank. Contact Nod Admin")

    model_dir = os.path.join(WORKDIR, model_dir_name)
    with open(os.path.join(model_dir, model_name + "_tf.mlir")) as f:
        mlir_file = f.read()

    function_name = str(np.load(os.path.join(model_dir, "function_name.npy")))
    inputs = np.load(os.path.join(model_dir, "inputs.npz"))
    golden_out = np.load(os.path.join(model_dir, "golden_out.npz"))

    inputs_tuple = tuple([inputs[key] for key in inputs])
    golden_out_tuple = tuple([golden_out[key] for key in golden_out])
    return mlir_file, function_name, inputs_tuple, golden_out_tuple
