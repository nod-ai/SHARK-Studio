# from turbine_models.custom_models.controlnet import control_adapter, preprocessors
import os
import PIL
import numpy as np
from apps.shark_studio.web.utils.file_utils import (
    get_generated_imgs_path,
)
from datetime import datetime
from PIL import Image
from gradio.components.image_editor import (
    EditorValue,
)


class control_adapter:
    def __init__(
        self,
        model: str,
    ):
        self.model = None

    def export_control_adapter_model(model_keyword):
        return None

    def export_xl_control_adapter_model(model_keyword):
        return None


class preprocessors:
    def __init__(
        self,
        model: str,
    ):
        self.model = None

    def export_controlnet_model(model_keyword):
        return None


control_adapter_map = {
    "sd15": {
        "canny": {"initializer": control_adapter.export_control_adapter_model},
        "openpose": {
            "initializer": control_adapter.export_control_adapter_model
        },
        "scribble": {
            "initializer": control_adapter.export_control_adapter_model
        },
        "zoedepth": {
            "initializer": control_adapter.export_control_adapter_model
        },
    },
    "sdxl": {
        "canny": {
            "initializer": control_adapter.export_xl_control_adapter_model
        },
    },
}
preprocessor_model_map = {
    "canny": {"initializer": preprocessors.export_controlnet_model},
    "openpose": {"initializer": preprocessors.export_controlnet_model},
    "scribble": {"initializer": preprocessors.export_controlnet_model},
    "zoedepth": {"initializer": preprocessors.export_controlnet_model},
}


class PreprocessorModel:
    def __init__(
        self,
        hf_model_id,
        device="cpu",
    ):
        self.model = hf_model_id
        self.device = device

    def compile(self):
        print("compile not implemented for preprocessor.")
        return

    def run(self, inputs):
        print("run not implemented for preprocessor.")
        return inputs


def cnet_preview(model, input_image):
    curr_datetime = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
    control_imgs_path = os.path.join(
        get_generated_imgs_path(), "control_hints"
    )
    if not os.path.exists(control_imgs_path):
        os.mkdir(control_imgs_path)
    img_dest = os.path.join(control_imgs_path, model + curr_datetime + ".png")
    match model:
        case "canny":
            canny = PreprocessorModel("canny")
            result = canny(
                np.array(input_image),
                100,
                200,
            )
            Image.fromarray(result).save(fp=img_dest)
            return result, img_dest
        case "openpose":
            openpose = PreprocessorModel("openpose")
            result = openpose(np.array(input_image))
            Image.fromarray(result[0]).save(fp=img_dest)
            return result, img_dest
        case "zoedepth":
            zoedepth = PreprocessorModel("ZoeDepth")
            result = zoedepth(np.array(input_image))
            Image.fromarray(result).save(fp=img_dest)
            return result, img_dest
        case "scribble":
            input_image.save(fp=img_dest)
            return input_image, img_dest
        case _:
            return None, None
