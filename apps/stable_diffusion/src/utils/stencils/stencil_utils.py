import numpy as np
from PIL import Image
import torch
import os
from pathlib import Path
import torchvision
import time
from apps.stable_diffusion.src.utils.stencils import (
    CannyDetector,
    OpenposeDetector,
    ZoeDetector,
)

stencil = {}


def save_img(img):
    from apps.stable_diffusion.src.utils import (
        get_generated_imgs_path,
        get_generated_imgs_todays_subdir,
    )

    subdir = Path(
        get_generated_imgs_path(), get_generated_imgs_todays_subdir()
    )
    os.makedirs(subdir, exist_ok=True)
    if isinstance(img, Image.Image):
        img.save(
            os.path.join(
                subdir, "controlnet_" + str(int(time.time())) + ".png"
            )
        )
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        img.save(os.path.join(subdir, str(int(time.time())) + ".png"))
    else:
        converter = torchvision.transforms.ToPILImage()
        for i in img:
            converter(i).save(
                os.path.join(subdir, str(int(time.time())) + ".png")
            )


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def controlnet_hint_shaping(
    controlnet_hint, height, width, dtype, num_images_per_prompt=1
):
    channels = 3
    if isinstance(controlnet_hint, torch.Tensor):
        # torch.Tensor: acceptble shape are any of chw, bchw(b==1) or bchw(b==num_images_per_prompt)
        shape_chw = (channels, height, width)
        shape_bchw = (1, channels, height, width)
        shape_nchw = (num_images_per_prompt, channels, height, width)
        if controlnet_hint.shape in [shape_chw, shape_bchw, shape_nchw]:
            controlnet_hint = controlnet_hint.to(
                dtype=dtype, device=torch.device("cpu")
            )
            if controlnet_hint.shape != shape_nchw:
                controlnet_hint = controlnet_hint.repeat(
                    num_images_per_prompt, 1, 1, 1
                )
            return controlnet_hint
        else:
            return controlnet_hint_shaping(
                Image.fromarray(controlnet_hint.detach().numpy()),
                height,
                width,
                dtype,
                num_images_per_prompt,
            )
    elif isinstance(controlnet_hint, np.ndarray):
        # np.ndarray: acceptable shape is any of hw, hwc, bhwc(b==1) or bhwc(b==num_images_per_promot)
        # hwc is opencv compatible image format. Color channel must be BGR Format.
        if controlnet_hint.shape == (height, width):
            controlnet_hint = np.repeat(
                controlnet_hint[:, :, np.newaxis], channels, axis=2
            )  # hw -> hwc(c==3)
        shape_hwc = (height, width, channels)
        shape_bhwc = (1, height, width, channels)
        shape_nhwc = (num_images_per_prompt, height, width, channels)
        if controlnet_hint.shape in [shape_hwc, shape_bhwc, shape_nhwc]:
            controlnet_hint = torch.from_numpy(controlnet_hint.copy())
            controlnet_hint = controlnet_hint.to(
                dtype=dtype, device=torch.device("cpu")
            )
            controlnet_hint /= 255.0
            if controlnet_hint.shape != shape_nhwc:
                controlnet_hint = controlnet_hint.repeat(
                    num_images_per_prompt, 1, 1, 1
                )
            controlnet_hint = controlnet_hint.permute(
                0, 3, 1, 2
            )  # b h w c -> b c h w
            return controlnet_hint
        else:
            return controlnet_hint_shaping(
                Image.fromarray(controlnet_hint),
                height,
                width,
                dtype,
                num_images_per_prompt,
            )

    elif isinstance(controlnet_hint, Image.Image):
        controlnet_hint = controlnet_hint.convert(
            "RGB"
        )  # make sure 3 channel RGB format
        if controlnet_hint.size == (width, height):
            controlnet_hint = np.array(controlnet_hint).astype(
                np.float16
            )  # to numpy
            controlnet_hint = controlnet_hint[:, :, ::-1]  # RGB -> BGR
            return
        else:
            (hint_w, hint_h) = controlnet_hint.size
            left = int((hint_w - width) / 2)
            right = left + height
            controlnet_hint = controlnet_hint.crop((left, 0, right, hint_h))
            controlnet_hint = controlnet_hint.resize((width, height))
        return controlnet_hint_shaping(
            controlnet_hint, height, width, dtype, num_images_per_prompt
        )
    else:
        raise ValueError(
            f"Acceptible controlnet input types are any of torch.Tensor, np.ndarray, PIL.Image.Image but is {type(controlnet_hint)}"
        )


def controlnet_hint_conversion(
    image, use_stencil, height, width, dtype, num_images_per_prompt=1
):
    controlnet_hint = None
    match use_stencil:
        case "canny":
            print(
                "Converting controlnet hint to edge detection mask with canny preprocessor."
            )
            controlnet_hint = hint_canny(image)
        case "openpose":
            print(
                "Detecting human pose in controlnet hint with openpose preprocessor."
            )
            controlnet_hint = hint_openpose(image)
        case "scribble":
            print("Using your scribble as a controlnet hint.")
            controlnet_hint = hint_scribble(image)
        case "zoedepth":
            print(
                "Converting controlnet hint to a depth mapping with ZoeDepth."
            )
            controlnet_hint = hint_zoedepth(image)
        case _:
            return None
    controlnet_hint = controlnet_hint_shaping(
        controlnet_hint, height, width, dtype, num_images_per_prompt
    )
    return controlnet_hint


stencil_to_model_id_map = {
    "canny": "lllyasviel/control_v11p_sd15_canny",
    "zoedepth": "lllyasviel/control_v11f1p_sd15_depth",
    "hed": "lllyasviel/sd-controlnet-hed",
    "mlsd": "lllyasviel/control_v11p_sd15_mlsd",
    "normal": "lllyasviel/control_v11p_sd15_normalbae",
    "openpose": "lllyasviel/control_v11p_sd15_openpose",
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "seg": "lllyasviel/control_v11p_sd15_seg",
}


def get_stencil_model_id(use_stencil):
    if use_stencil in stencil_to_model_id_map:
        return stencil_to_model_id_map[use_stencil]
    return None


# Stencil 1. Canny
def hint_canny(
    image: Image.Image,
    low_threshold=100,
    high_threshold=200,
):
    with torch.no_grad():
        input_image = np.array(image)

        if not "canny" in stencil:
            stencil["canny"] = CannyDetector()
        detected_map = stencil["canny"](
            input_image, low_threshold, high_threshold
        )
        save_img(detected_map)
        detected_map = HWC3(detected_map)
        return detected_map


# Stencil 2. OpenPose.
def hint_openpose(
    image: Image.Image,
):
    with torch.no_grad():
        input_image = np.array(image)

        if not "openpose" in stencil:
            stencil["openpose"] = OpenposeDetector()

        detected_map, _ = stencil["openpose"](input_image)
        save_img(detected_map)
        detected_map = HWC3(detected_map)
        return detected_map


# Stencil 3. Scribble.
def hint_scribble(image: Image.Image):
    with torch.no_grad():
        input_image = np.array(image)

        detected_map = np.zeros_like(input_image, dtype=np.uint8)
        detected_map[np.min(input_image, axis=2) < 127] = 255
        save_img(detected_map)
        return detected_map


# Stencil 4. Depth (Only Zoe Preprocessing)
def hint_zoedepth(image: Image.Image):
    with torch.no_grad():
        input_image = np.array(image)

        if not "depth" in stencil:
            stencil["depth"] = ZoeDetector()

        detected_map = stencil["depth"](input_image)
        save_img(detected_map)
        detected_map = HWC3(detected_map)
        return detected_map
