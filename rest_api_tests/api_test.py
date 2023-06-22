import requests
from PIL import Image
import base64
from io import BytesIO


def upscaler_test():
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

    print(f"response from server was : {res.status_code}")


def img2img_test():
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

    print(f"response from server was : {res.status_code}")

    print("Extracting response object")

    # NOTE Uncomment below to save the picture

    # response_obj = res.json()
    # img_b64 = response_obj.get("images", [False])[0] or response_obj.get(
    #     "image"
    # )
    # img_b2 = base64.b64decode(img_b64.replace("data:image/png;base64,", ""))
    # im_file = BytesIO(img_b2)
    # response_img = Image.open(im_file)
    # print("Saving Response Image to: response_img")
    # response_img.save(r"rest_api_tests/response_img.png")

def inpainting_test():
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
        "mask" : mask,
        "height": height,
        "width": width,
        "steps": steps,
        "noise_level": noise_level,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "is_full_res" : is_full_res,
        "full_res_padding" : full_res_padding
    }

    res = requests.post(url=url, json=data, headers=headers, timeout=1000)

    print(f"[Inpainting] response from server was : {res.status_code}")



if __name__ == "__main__":
    img2img_test()
    upscaler_test()
    inpainting_test()
