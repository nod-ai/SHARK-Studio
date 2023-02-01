import os
import subprocess
from apps.stable_diffusion.src.utils.resources import (
    get_json_file,
)
from shark.shark_downloader import download_public_file
from image_comparison import compare_images
import argparse
from glob import glob
import shutil

model_config_dicts = get_json_file(
    os.path.join(
        os.getcwd(),
        "apps/stable_diffusion/src/utils/resources/model_config.json",
    )
)


def test_loop(device="vulkan", beta=False, extra_flags=[]):
    # Get golden values from tank
    shutil.rmtree("./test_images", ignore_errors=True)
    os.mkdir("./test_images")
    os.mkdir("./test_images/golden")
    hf_model_names = model_config_dicts[0].values()
    tuned_options = ["--no-use_tuned"]  #'use_tuned']
    devices = ["vulkan"]
    if beta:
        extra_flags.append("--beta_models=True")
    for model_name in hf_model_names:
        for use_tune in tuned_options:
            command = [
                "python",
                "apps/stable_diffusion/src/txt2img.py",
                "--device=" + device,
                "--output_dir=./test_images/" + model_name,
                "--hf_model_id=" + model_name,
                use_tune,
            ]
            command += extra_flags
            generated_image = not subprocess.call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if generated_image:
                os.makedirs(
                    "./test_images/golden/" + model_name, exist_ok=True
                )
                download_public_file(
                    "gs://shark_tank/testdata/golden/" + model_name,
                    "./test_images/golden/" + model_name,
                )
                comparison = [
                    "python",
                    "build_tools/image_comparison.py",
                    "--golden_url=gs://shark_tank/testdata/golden/"
                    + model_name
                    + "/*.png",
                    "--newfile=./test_images/" + model_name + "/*.png",
                ]
                test_file = glob("./test_images/" + model_name + "/*.png")[0]
                golden_path = "./test_images/golden/" + model_name + "/*.png"
                golden_file = glob(golden_path)[0]
                compare_images(test_file, golden_file)


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--device", default="vulkan")
parser.add_argument(
    "-b", "--beta", action=argparse.BooleanOptionalAction, default=False
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    test_loop(args.device, args.beta, [])
