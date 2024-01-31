import requests
from PIL import Image
import base64
from io import BytesIO


def upscaler_test(verbose=False):
    # Define values here
    prompt = ""
    negative_prompt = ""
    seed = 2121991605
    height = 512
    width = 512
    steps = 50
    noise_level = 10
    cfg_scale = 7
    image_path = r"./rest_api_tests/dog.png"

    # Converting Image to base64
    img_file = open(image_path, "rb")
    init_images = [
        "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
    ]

    url = "http://127.0.0.1:8080/sdapi/v1/upscaler"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "steps": steps,
        "noise_level": noise_level,
        "cfg_scale": cfg_scale,
        "init_images": init_images,
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    print(f"[upscaler] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json()['info'] if res.status_code == 200 else res.content}\n")


def img2img_test(verbose=False):
    # Define values here
    prompt = "Paint a rabbit riding on the dog"
    negative_prompt = "ugly, bad art, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, tiling, signature, cut off, draft"
    seed = 2121991605
    height = 512
    width = 512
    steps = 50
    denoising_strength = 0.75
    cfg_scale = 7
    image_path = r"./rest_api_tests/dog.png"

    # Converting Image to Base64
    img_file = open(image_path, "rb")
    init_images = [
        "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
    ]

    url = "http://127.0.0.1:8080/sdapi/v1/img2img"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_images": init_images,
        "height": height,
        "width": width,
        "steps": steps,
        "denoising_strength": denoising_strength,
        "cfg_scale": cfg_scale,
        "seed": seed,
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    print(f"[img2img] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json()['info'] if res.status_code == 200 else res.content}\n")

    # NOTE Uncomment below to save the picture

    # print("Extracting response object")
    # response_obj = res.json()
    # img_b64 = response_obj.get("images", [False])[0] or response_obj.get(
    #     "image"
    # )
    # img_b2 = base64.b64decode(img_b64.replace("data:image/png;base64,", ""))
    # im_file = BytesIO(img_b2)
    # response_img = Image.open(im_file)
    # print("Saving Response Image to: response_img")
    # response_img.save(r"rest_api_tests/response_img.png")


def inpainting_test(verbose=False):
    prompt = "Paint a rabbit riding on the dog"
    negative_prompt = "ugly, bad art, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, tiling, signature, cut off, draft"
    seed = 2121991605
    height = 512
    width = 512
    steps = 50
    noise_level = 10
    cfg_scale = 7
    is_full_res = False
    full_res_padding = 32
    image_path = r"./rest_api_tests/dog.png"

    img_file = open(image_path, "rb")
    image = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
    img_file = open(image_path, "rb")
    mask = "data:image/png;base64," + base64.b64encode(img_file.read()).decode()

    url = "http://127.0.0.1:8080/sdapi/v1/inpaint"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "mask": mask,
        "height": height,
        "width": width,
        "steps": steps,
        "noise_level": noise_level,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "is_full_res": is_full_res,
        "full_res_padding": full_res_padding,
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    print(f"[inpaint] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json()['info'] if res.status_code == 200 else res.content}\n")


def outpainting_test(verbose=False):
    prompt = "Paint a rabbit riding on the dog"
    negative_prompt = "ugly, bad art, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, tiling, signature, cut off, draft"
    seed = 2121991605
    height = 512
    width = 512
    steps = 50
    cfg_scale = 7
    color_variation = 0.2
    noise_q = 0.2
    directions = ["up", "down", "right", "left"]
    pixels = 32
    mask_blur = 64
    image_path = r"./rest_api_tests/dog.png"

    # Converting Image to Base64
    img_file = open(image_path, "rb")
    init_images = [
        "data:image/png;base64," + base64.b64encode(img_file.read()).decode()
    ]

    url = "http://127.0.0.1:8080/sdapi/v1/outpaint"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "color_variation": color_variation,
        "noise_q": noise_q,
        "directions": directions,
        "pixels": pixels,
        "mask_blur": mask_blur,
        "init_images": init_images,
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    print(f"[outpaint] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json()['info'] if res.status_code == 200 else res.content}\n")


def txt2img_test(verbose=False):
    prompt = "Paint a rabbit in a top hate"
    negative_prompt = "ugly, bad art, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, tiling, signature, cut off, draft"
    seed = 2121991605
    height = 512
    width = 512
    steps = 50
    cfg_scale = 7

    url = "http://127.0.0.1:8080/sdapi/v1/txt2img"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "steps": steps,
        "cfg_scale": cfg_scale,
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    print(f"[txt2img] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json()['info'] if res.status_code == 200 else res.content}\n")


def sd_models_test(verbose=False):
    url = "http://127.0.0.1:8080/sdapi/v1/sd-models"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    res = requests.get(url=url, headers=headers, timeout=1000)

    print(f"[sd_models] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json() if res.status_code == 200 else res.content}\n")


def sd_samplers_test(verbose=False):
    url = "http://127.0.0.1:8080/sdapi/v1/samplers"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    res = requests.get(url=url, headers=headers, timeout=1000)

    print(f"[sd_samplers] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json() if res.status_code == 200 else res.content}\n")


def options_test(verbose=False):
    url = "http://127.0.0.1:8080/sdapi/v1/options"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    res = requests.get(url=url, headers=headers, timeout=1000)

    print(f"[options] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json() if res.status_code == 200 else res.content}\n")


def cmd_flags_test(verbose=False):
    url = "http://127.0.0.1:8080/sdapi/v1/cmd-flags"

    headers = {
        "User-Agent": "PythonTest",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
    }

    res = requests.get(url=url, headers=headers, timeout=1000)

    print(f"[cmd-flags] response from server was : {res.status_code} {res.reason}")

    if verbose or res.status_code != 200:
        print(f"\n{res.json() if res.status_code == 200 else res.content}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Exercises the Stable Diffusion REST API of Shark. Make sure "
            "Shark is running in API mode on 127.0.0.1:8080 before running"
            "this script."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "also display selected info from the JSON response for "
            "successful requests"
        ),
    )
    args = parser.parse_args()

    sd_models_test(args.verbose)
    sd_samplers_test(args.verbose)
    options_test(args.verbose)
    cmd_flags_test(args.verbose)
    txt2img_test(args.verbose)
    img2img_test(args.verbose)
    upscaler_test(args.verbose)
    inpainting_test(args.verbose)
    outpainting_test(args.verbose)
