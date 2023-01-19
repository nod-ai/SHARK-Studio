import argparse
import torchvision
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
    return torchvision.io.read_image(local_filename).numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    new = torchvision.io.read_image(args.newfile).numpy() / 255.0
    tempfile_name = os.path.join(os.getcwd(), "golden.png")
    golden = get_image(args.golden_url, tempfile_name) / 255.0
    diff = np.abs(new - golden)
    mean = np.mean(diff)
    if not mean < 0.2:
        subprocess.run(
            ["gsutil", "cp", args.newfile, "gs://shark_tank/testdata/builder/"]
        )
        raise SystemExit("new and golden not close")
    else:
        print("SUCCESS")
