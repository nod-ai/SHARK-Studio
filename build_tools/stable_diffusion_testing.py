import os
from sys import executable
import subprocess
from apps.stable_diffusion.src.utils.resources import (
    get_json_file,
)
from datetime import datetime as dt
from shark.shark_downloader import download_public_file
from image_comparison import compare_images
import argparse
from glob import glob
import shutil
import requests

model_config_dicts = get_json_file(
    os.path.join(
        os.getcwd(),
        "apps/stable_diffusion/src/utils/resources/model_config.json",
    )
)


def parse_sd_out(filename, command, device, use_tune, model_name, import_mlir):
    with open(filename, "r+") as f:
        lines = f.readlines()
    metrics = {}
    vals_to_read = [
        "Clip Inference time",
        "Average step",
        "VAE Inference time",
        "Total image generation",
    ]
    for line in lines:
        for val in vals_to_read:
            if val in line:
                metrics[val] = line.split(" ")[-1].strip("\n")

    metrics["Average step"] = metrics["Average step"].strip("ms/it")
    metrics["Total image generation"] = metrics[
        "Total image generation"
    ].strip("sec")
    metrics["device"] = device
    metrics["use_tune"] = use_tune
    metrics["model_name"] = model_name
    metrics["import_mlir"] = import_mlir
    metrics["command"] = command
    return metrics


def get_inpaint_inputs():
    os.mkdir("./test_images/inputs")
    img_url = (
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve"
        "/main/stable_diffusion_inpaint/input_bench_image.png"
    )
    mask_url = (
        "https://huggingface.co/datasets/diffusers/test-arrays/resolve"
        "/main/stable_diffusion_inpaint/input_bench_mask.png"
    )
    img = requests.get(img_url)
    mask = requests.get(mask_url)
    open("./test_images/inputs/image.png", "wb").write(img.content)
    open("./test_images/inputs/mask.png", "wb").write(mask.content)


def test_loop(device="vulkan", beta=False, extra_flags=[]):
    # Get golden values from tank
    shutil.rmtree("./test_images", ignore_errors=True)
    model_metrics = []
    os.mkdir("./test_images")
    os.mkdir("./test_images/golden")
    get_inpaint_inputs()
    hf_model_names = model_config_dicts[0].values()
    tuned_options = ["--no-use_tuned", "--use_tuned"]
    import_options = ["--import_mlir", "--no-import_mlir"]
    prompt_text = "--prompt=cyberpunk forest by Salvador Dali"
    inpaint_prompt_text = "--prompt=Face of a yellow cat, high resolution, sitting on a park bench"
    if os.name == "nt":
        prompt_text = '--prompt="cyberpunk forest by Salvador Dali"'
        inpaint_prompt_text = '--prompt="Face of a yellow cat, high resolution, sitting on a park bench"'
    if beta:
        extra_flags.append("--beta_models=True")
    extra_flags.append("--no-progress_bar")
    for import_opt in import_options:
        for model_name in hf_model_names:
            if model_name == "Linaqruf/anything-v3.0":
                continue
            for use_tune in tuned_options:
                command = (
                    [
                        executable,  # executable is the python from the venv used to run this
                        "apps/stable_diffusion/scripts/txt2img.py",
                        "--device=" + device,
                        prompt_text,
                        "--negative_prompts=" + '""',
                        "--seed=42",
                        import_opt,
                        "--output_dir="
                        + os.path.join(os.getcwd(), "test_images", model_name),
                        "--hf_model_id=" + model_name,
                        use_tune,
                    ]
                    if "inpainting" not in model_name
                    else [
                        executable,
                        "apps/stable_diffusion/scripts/inpaint.py",
                        "--device=" + device,
                        inpaint_prompt_text,
                        "--negative_prompts=" + '""',
                        "--img_path=./test_images/inputs/image.png",
                        "--mask_path=./test_images/inputs/mask.png",
                        "--seed=42",
                        "--import_mlir",
                        "--output_dir="
                        + os.path.join(os.getcwd(), "test_images", model_name),
                        "--hf_model_id=" + model_name,
                        use_tune,
                    ]
                )
                command += extra_flags
                if os.name == "nt":
                    command = " ".join(command)
                dumpfile_name = "_".join(model_name.split("/")) + ".txt"
                dumpfile_name = os.path.join(os.getcwd(), dumpfile_name)
                with open(dumpfile_name, "w+") as f:
                    generated_image = not subprocess.call(
                        command,
                        stdout=f,
                        stderr=f,
                    )
                if os.name != "nt":
                    command = " ".join(command)
                if generated_image:
                    model_metrics.append(
                        parse_sd_out(
                            dumpfile_name,
                            command,
                            device,
                            use_tune,
                            model_name,
                            import_opt,
                        )
                    )
                    print(command)
                    print("Successfully generated image")
                    os.makedirs(
                        "./test_images/golden/" + model_name, exist_ok=True
                    )
                    download_public_file(
                        "gs://shark_tank/testdata/golden/" + model_name,
                        "./test_images/golden/" + model_name,
                    )
                    test_file_path = os.path.join(
                        os.getcwd(),
                        "test_images",
                        model_name,
                        "generated_imgs",
                        dt.now().strftime("%Y%m%d"),
                        "*.png",
                    )
                    test_file = glob(test_file_path)[0]

                    golden_path = (
                        "./test_images/golden/" + model_name + "/*.png"
                    )
                    golden_file = glob(golden_path)[0]
                    compare_images(test_file, golden_file)
                else:
                    print(command)
                    print("failed to generate image for this configuration")
                    if "2_1_base" in model_name:
                        print("failed a known successful model.")
                        exit(1)
    with open(os.path.join(os.getcwd(), "sd_testing_metrics.csv"), "w+") as f:
        header = "model_name;device;use_tune;import_opt;Clip Inference time(ms);Average Step (ms/it);VAE Inference time(ms);total image generation(s);command\n"
        f.write(header)
        for metric in model_metrics:
            output = [
                metric["model_name"],
                metric["device"],
                metric["use_tune"],
                metric["import_mlir"],
                metric["Clip Inference time"],
                metric["Average step"],
                metric["VAE Inference time"],
                metric["Total image generation"],
                metric["command"],
            ]
            f.write(";".join(output) + "\n")


parser = argparse.ArgumentParser()

parser.add_argument("-d", "--device", default="vulkan")
parser.add_argument(
    "-b", "--beta", action=argparse.BooleanOptionalAction, default=False
)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    test_loop(args.device, args.beta, [])
