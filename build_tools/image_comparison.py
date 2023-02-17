import argparse
from PIL import Image
import numpy as np

import requests
import shutil
import os
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--newfile")
parser.add_argument(
    "-g",
    "--golden_url",
    default="https://storage.googleapis.com/shark_tank/testdata/cyberpunk_fores_42_0_230119_021148.png",
)


def get_image(url, local_filename):
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(local_filename, "wb") as f:
            shutil.copyfileobj(res.raw, f)


def compare_images(new_filename, golden_filename):
    new = np.array(Image.open(new_filename)) / 255.0
    golden = np.array(Image.open(golden_filename)) / 255.0
    diff = np.abs(new - golden)
    mean = np.mean(diff)
    if mean > 0.1:
        if os.name != "nt":
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    new_filename,
                    "gs://shark_tank/testdata/builder/",
                ]
            )
        raise SystemExit("new and golden not close")
    else:
        print("SUCCESS")


if __name__ == "__main__":
    args = parser.parse_args()
    tempfile_name = os.path.join(os.getcwd(), "golden.png")
    get_image(args.golden_url, tempfile_name)
    compare_images(args.newfile, tempfile_name)
