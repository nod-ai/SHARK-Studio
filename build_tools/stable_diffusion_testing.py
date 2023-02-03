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
    tuned_options = ["--no-use_tuned", "use_tuned"]
    if beta:
        extra_flags.append("--beta_models=True")
    for model_name in hf_model_names:
        for use_tune in tuned_options:
            command = [
                "python",
                "apps/stable_diffusion/scripts/txt2img.py",
                "--device=" + device,
                "--prompt=cyberpunk forest by Salvador Dali",
                "--output_dir="
                + os.path.join(os.getcwd(), "test_images", model_name),
                "--hf_model_id=" + model_name,
                use_tune,
            ]
            command += extra_flags
            generated_image = not subprocess.call(
                command, stdout=subprocess.DEVNULL
            )
            if generated_image:
                print(" ".join(command))
                print("Successfully generated image")
                os.makedirs(
                    "./test_images/golden/" + model_name, exist_ok=True
                )
                download_public_file(
                    "gs://shark_tank/testdata/golden/" + model_name,
                    "./test_images/golden/" + model_name,
                )
                test_file_path = os.path.join(
                    os.getcwd(), "test_images", model_name, "generated_imgs"
                )
                test_file = glob(test_file_path + "/*.png")[0]
                golden_path = "./test_images/golden/" + model_name + "/*.png"
                golden_file = glob(golden_path)[0]
                compare_images(test_file, golden_file)
            else:
                print(" ".join(command))
                print("failed to generate image for this configuration")


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--device", default="vulkan")
parser.add_argument(
    "-b", "--beta", action=argparse.BooleanOptionalAction, default=False
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    test_loop(args.device, args.beta, [])
