import os
import shutil
from time import time

from apps.shark_studio.modules.shared_cmd_opts import cmd_opts

shark_tmp = cmd_opts.tmp_dir  # os.path.join(os.getcwd(), "shark_tmp/")


def clear_tmp_mlir():
    cleanup_start = time()
    print("Clearing .mlir temporary files from a prior run. This may take some time...")
    mlir_files = [
        filename
        for filename in os.listdir(shark_tmp)
        if os.path.isfile(os.path.join(shark_tmp, filename))
        and filename.endswith(".mlir")
    ]
    for filename in mlir_files:
        os.remove(os.path.join(shark_tmp, filename))
    print(f"Clearing .mlir temporary files took {time() - cleanup_start:.4f} seconds.")


def clear_tmp_imgs():
    # tell gradio to use a directory under shark_tmp for its temporary
    # image files unless somewhere else has been set
    if "GRADIO_TEMP_DIR" not in os.environ:
        os.environ["GRADIO_TEMP_DIR"] = os.path.join(shark_tmp, "gradio")

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

    # older SHARK versions had to workaround gradio bugs and stored things differently
    else:
        image_files = [
            filename
            for filename in os.listdir(shark_tmp)
            if os.path.isfile(os.path.join(shark_tmp, filename))
            and filename.startswith("tmp")
            and filename.endswith(".png")
        ]
        if len(image_files) > 0:
            print(
                "Clearing temporary image files of a prior run of a previous SHARK version. This may take some time..."
            )
            cleanup_start = time()
            for filename in image_files:
                os.remove(shark_tmp + filename)
            print(
                f"Clearing temporary image files took {time() - cleanup_start:.4f} seconds."
            )
        else:
            print("No temporary images files to clear.")


def config_tmp():
    # create shark_tmp if it does not exist
    if not os.path.exists(shark_tmp):
        os.mkdir(shark_tmp)

    clear_tmp_mlir()
    clear_tmp_imgs()
