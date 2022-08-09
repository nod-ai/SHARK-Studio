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
import hashlib
from pathlib import Path

input_type_to_np_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
}

#default hash is updated when nightly populate_sharktank_ci is successful
shark_default_sha = "274650f"

# Save the model in the home local so it needn't be fetched everytime in the CI.
home = str(Path.home())
WORKDIR = os.path.join(home, ".local/shark_tank/")
print(WORKDIR)


# Checks whether the directory and files exists.
def check_dir_exists(model_name, frontend="torch", dynamic=""):
    model_dir = os.path.join(WORKDIR, model_name)

    # Remove the _tf keyword from end.
    if frontend in ["tf", "tensorflow"]:
        model_name = model_name[:-3]
    elif frontend in ["tflite"]:
        model_name = model_name[:-7]
    elif frontend in ["torch", "pytorch"]:
        model_name = model_name[:-6]

    if os.path.isdir(model_dir):
        if (
            os.path.isfile(
                os.path.join(
                    model_dir,
                    model_name + dynamic + "_" + str(frontend) + ".mlir",
                )
            )
            and os.path.isfile(os.path.join(model_dir, "function_name.npy"))
            and os.path.isfile(os.path.join(model_dir, "inputs.npz"))
            and os.path.isfile(os.path.join(model_dir, "golden_out.npz"))
            and os.path.isfile(os.path.join(model_dir, "hash.npy"))
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
    model_dir_name = model_name + "_torch"

    def gs_download_model():
        gs_command = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp -r gs://shark_tank/'
            + shark_default_sha
            + "/"
            + model_dir_name
            + " "
            + WORKDIR
        )
        if os.system(gs_command) != 0:
            raise Exception("model not present in the tank. Contact Nod Admin")

    if not check_dir_exists(model_dir_name, frontend="torch", dynamic=dyn_str):
        gs_download_model()
    else:
        model_dir = os.path.join(WORKDIR, model_dir_name)
        local_hash = str(np.load(os.path.join(model_dir, "hash.npy")))
        gs_hash = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp gs://shark_tank/'
            + shark_default_sha
            + "/"
            + model_dir_name
            + "/hash.npy"
            + " "
            + os.path.join(model_dir, "upstream_hash.npy")
        )
        if os.system(gs_hash) != 0:
            raise Exception("hash of the model not present in the tank.")
        upstream_hash = str(
            np.load(os.path.join(model_dir, "upstream_hash.npy"))
        )
        if local_hash != upstream_hash:
            gs_download_model()

    model_dir = os.path.join(WORKDIR, model_dir_name)
    with open(
        os.path.join(model_dir, model_name + dyn_str + "_torch.mlir")
    ) as f:
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
    model_dir_name = model_name + "_tflite"

    def gs_download_model():
        gs_command = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp -r gs://shark_tank/'
            + shark_default_sha
            + "/"
            + model_dir_name
            + " "
            + WORKDIR
        )
        if os.system(gs_command) != 0:
            raise Exception("model not present in the tank. Contact Nod Admin")

    if not check_dir_exists(
        model_dir_name, frontend="tflite", dynamic=dyn_str
    ):
        gs_download_model()
    else:
        model_dir = os.path.join(WORKDIR, model_dir_name)
        local_hash = str(np.load(os.path.join(model_dir, "hash.npy")))
        gs_hash = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp gs://shark_tank'
            + "/"
            + model_dir_name
            + "/hash.npy"
            + " "
            + os.path.join(model_dir, "upstream_hash.npy")
        )
        if os.system(gs_hash) != 0:
            raise Exception("hash of the model not present in the tank.")
        upstream_hash = str(
            np.load(os.path.join(model_dir, "upstream_hash.npy"))
        )
        if local_hash != upstream_hash:
            gs_download_model()

    model_dir = os.path.join(WORKDIR, model_dir_name)
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

    def gs_download_model():
        gs_command = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp -r gs://shark_tank/'
            + shark_default_sha
            + "/"
            + model_dir_name
            + " "
            + WORKDIR
        )
        if os.system(gs_command) != 0:
            raise Exception("model not present in the tank. Contact Nod Admin")

    if not check_dir_exists(model_dir_name, frontend="tf"):
        gs_download_model()
    else:
        model_dir = os.path.join(WORKDIR, model_dir_name)
        local_hash = str(np.load(os.path.join(model_dir, "hash.npy")))
        gs_hash = (
            'gsutil -o "GSUtil:parallel_process_count=1" cp gs://shark_tank/'
            + shark_default_sha
            + "/"
            + model_dir_name
            + "/hash.npy"
            + " "
            + os.path.join(model_dir, "upstream_hash.npy")
        )
        if os.system(gs_hash) != 0:
            raise Exception("hash of the model not present in the tank.")
        upstream_hash = str(
            np.load(os.path.join(model_dir, "upstream_hash.npy"))
        )
        if local_hash != upstream_hash:
            gs_download_model()

    model_dir = os.path.join(WORKDIR, model_dir_name)
    with open(os.path.join(model_dir, model_name + "_tf.mlir")) as f:
        mlir_file = f.read()

    function_name = str(np.load(os.path.join(model_dir, "function_name.npy")))
    inputs = np.load(os.path.join(model_dir, "inputs.npz"))
    golden_out = np.load(os.path.join(model_dir, "golden_out.npz"))

    inputs_tuple = tuple([inputs[key] for key in inputs])
    golden_out_tuple = tuple([golden_out[key] for key in golden_out])
    return mlir_file, function_name, inputs_tuple, golden_out_tuple
