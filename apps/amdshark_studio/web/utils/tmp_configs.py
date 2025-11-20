import os
import shutil
from time import time

from apps.amdshark_studio.modules.shared_cmd_opts import cmd_opts

amdshark_tmp = cmd_opts.tmp_dir  # os.path.join(os.getcwd(), "amdshark_tmp/")


def clear_tmp_mlir():
    cleanup_start = time()
    print("Clearing .mlir temporary files from a prior run. This may take some time...")
    mlir_files = [
        filename
        for filename in os.listdir(amdshark_tmp)
        if os.path.isfile(os.path.join(amdshark_tmp, filename))
        and filename.endswith(".mlir")
    ]
    for filename in mlir_files:
        os.remove(os.path.join(amdshark_tmp, filename))
    print(f"Clearing .mlir temporary files took {time() - cleanup_start:.4f} seconds.")


def clear_tmp_imgs():
    # tell gradio to use a directory under amdshark_tmp for its temporary
    # image files unless somewhere else has been set
    if "GRADIO_TEMP_DIR" not in os.environ:
        os.environ["GRADIO_TEMP_DIR"] = os.path.join(amdshark_tmp, "gradio")

    print(
        f"gradio temporary image cache located at {os.environ['GRADIO_TEMP_DIR']}. "
        + "You may change this by setting the GRADIO_TEMP_DIR environment variable."
    )

    # Clear all gradio tmp images from the last session
    if os.path.exists(os.environ["GRADIO_TEMP_DIR"]):
        cleanup_start = time()
        print(
            "Clearing gradio UI temporary image files from a prior run. This may take some time..."
        )
        shutil.rmtree(os.environ["GRADIO_TEMP_DIR"], ignore_errors=True)
        print(
            f"Clearing gradio UI temporary image files took {time() - cleanup_start:.4f} seconds."
        )

    # older AMDSHARK versions had to workaround gradio bugs and stored things differently
    else:
        image_files = [
            filename
            for filename in os.listdir(amdshark_tmp)
            if os.path.isfile(os.path.join(amdshark_tmp, filename))
            and filename.startswith("tmp")
            and filename.endswith(".png")
        ]
        if len(image_files) > 0:
            print(
                "Clearing temporary image files of a prior run of a previous AMDSHARK version. This may take some time..."
            )
            cleanup_start = time()
            for filename in image_files:
                os.remove(amdshark_tmp + filename)
            print(
                f"Clearing temporary image files took {time() - cleanup_start:.4f} seconds."
            )
        else:
            print("No temporary images files to clear.")


def config_tmp():
    # create amdshark_tmp if it does not exist
    if not os.path.exists(amdshark_tmp):
        os.mkdir(amdshark_tmp)

    clear_tmp_mlir()
    clear_tmp_imgs()
