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
from tqdm.std import tqdm
import sys
from pathlib import Path
from shark.parser import shark_args
from google.cloud import storage


def download_public_file(
    full_gs_url, destination_folder_name, single_file=False
):
    """Downloads a public blob from the bucket."""
    # bucket_name = "gs://your-bucket-name/path/to/file"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client.create_anonymous_client()
    bucket_name = full_gs_url.split("/")[2]
    source_blob_name = None
    dest_filename = None
    desired_file = None
    if single_file:
        desired_file = full_gs_url.split("/")[-1]
        source_blob_name = "/".join(full_gs_url.split("/")[3:-1])
        destination_folder_name, dest_filename = os.path.split(
            destination_folder_name
        )
    else:
        source_blob_name = "/".join(full_gs_url.split("/")[3:])
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)
    if not os.path.exists(destination_folder_name):
        os.mkdir(destination_folder_name)
    for blob in blobs:
        blob_name = blob.name.split("/")[-1]
        if single_file:
            if blob_name == desired_file:
                destination_filename = os.path.join(
                    destination_folder_name, dest_filename
                )
                with open(destination_filename, "wb") as f:
                    with tqdm.wrapattr(
                        f, "write", total=blob.size
                    ) as file_obj:
                        storage_client.download_blob_to_file(blob, file_obj)
            else:
                continue

        destination_filename = os.path.join(destination_folder_name, blob_name)
        with open(destination_filename, "wb") as f:
            with tqdm.wrapattr(f, "write", total=blob.size) as file_obj:
                storage_client.download_blob_to_file(blob, file_obj)


input_type_to_np_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "bool": np.bool_,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
}

# Save the model in the home local so it needn't be fetched everytime in the CI.
home = str(Path.home())
alt_path = os.path.join(os.path.dirname(__file__), "../gen_shark_tank/")
custom_path = shark_args.local_tank_cache
if os.path.exists(alt_path):
    WORKDIR = alt_path
    print(
        f"Using {WORKDIR} as shark_tank directory. Delete this directory if you aren't working from locally generated shark_tank."
    )
if custom_path:
    if not os.path.exists(custom_path):
        os.mkdir(custom_path)

    WORKDIR = custom_path

    print(f"Using {WORKDIR} as local shark_tank cache directory.")
else:
    WORKDIR = os.path.join(home, ".local/shark_tank/")
    print(
        f"shark_tank local cache is located at {WORKDIR} . You may change this by setting the --local_tank_cache= flag"
    )


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
            print(f"""Using cached models from {WORKDIR}...""")
            return True
    return False


# Downloads the torch model from gs://shark_tank dir.
def download_model(
    model_name,
    dynamic=False,
    tank_url="gs://shark_tank/latest",
    frontend=None,
    tuned=None,
):
    model_name = model_name.replace("/", "_")
    dyn_str = "_dynamic" if dynamic else ""
    os.makedirs(WORKDIR, exist_ok=True)
    model_dir_name = model_name + "_" + frontend
    model_dir = os.path.join(WORKDIR, model_dir_name)
    full_gs_url = tank_url.rstrip("/") + "/" + model_dir_name

    if shark_args.update_tank == True:
        print(f"Updating artifacts for model {model_name}...")
        download_public_file(full_gs_url, model_dir)

    elif not check_dir_exists(
        model_dir_name, frontend=frontend, dynamic=dyn_str
    ):
        print(f"Downloading artifacts for model {model_name}...")
        download_public_file(full_gs_url, model_dir)
    else:
        if not _internet_connected():
            print(
                "No internet connection. Using the model already present in the tank."
            )
        else:
            local_hash = str(np.load(os.path.join(model_dir, "hash.npy")))
            gs_hash_url = (
                tank_url.rstrip("/") + "/" + model_dir_name + "/hash.npy"
            )
            download_public_file(
                gs_hash_url,
                os.path.join(model_dir, "upstream_hash.npy"),
                single_file=True,
            )
            try:
                upstream_hash = str(
                    np.load(os.path.join(model_dir, "upstream_hash.npy"))
                )
            except FileNotFoundError:
                upstream_hash = None
            if local_hash != upstream_hash:
                print(
                    "Hash does not match upstream in gs://shark_tank/latest. If you want to use locally generated artifacts, this is working as intended. Otherwise, run with --update_tank."
                )

    model_dir = os.path.join(WORKDIR, model_dir_name)
    tuned_str = "" if tuned is None else "_" + tuned
    suffix = f"{dyn_str}_{frontend}{tuned_str}.mlir"
    filename = os.path.join(model_dir, model_name + suffix)

    with open(filename, mode="rb") as f:
        mlir_file = f.read()

    function_name = str(np.load(os.path.join(model_dir, "function_name.npy")))
    inputs = np.load(os.path.join(model_dir, "inputs.npz"))
    golden_out = np.load(os.path.join(model_dir, "golden_out.npz"))

    inputs_tuple = tuple([inputs[key] for key in inputs])
    golden_out_tuple = tuple([golden_out[key] for key in golden_out])
    return mlir_file, function_name, inputs_tuple, golden_out_tuple


def _internet_connected():
    import requests as req

    try:
        req.get("http://1.1.1.1")
        return True
    except:
        return False
