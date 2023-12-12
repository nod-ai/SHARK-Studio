# from turbine_models.custom_models.controlnet import control_adapter, preprocessors


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
        device,
    ):
        self.model = None

    def compile(self, device):
        print("compile not implemented for preprocessor.")
        return

    def run(self, inputs):
        print("run not implemented for preprocessor.")
        return


def cnet_preview(model, input_img, stencils, images, preprocessed_hints):
    if isinstance(input_image, PIL.Image.Image):
        img_dict = {
            "background": None,
            "layers": [None],
            "composite": input_image,
        }
        input_image = EditorValue(img_dict)
    images[index] = input_image
    if model:
        stencils[index] = model
    match model:
        case "canny":
            canny = CannyDetector()
            result = canny(
                np.array(input_image["composite"]),
                100,
                200,
            )
            preprocessed_hints[index] = Image.fromarray(result)
            return (
                Image.fromarray(result),
                stencils,
                images,
                preprocessed_hints,
            )
        case "openpose":
            openpose = OpenposeDetector()
            result = openpose(np.array(input_image["composite"]))
            preprocessed_hints[index] = Image.fromarray(result[0])
            return (
                Image.fromarray(result[0]),
                stencils,
                images,
                preprocessed_hints,
            )
        case "zoedepth":
            zoedepth = ZoeDetector()
            result = zoedepth(np.array(input_image["composite"]))
            preprocessed_hints[index] = Image.fromarray(result)
            return (
                Image.fromarray(result),
                stencils,
                images,
                preprocessed_hints,
            )
        case "scribble":
            preprocessed_hints[index] = input_image["composite"]
            return (
                input_image["composite"],
                stencils,
                images,
                preprocessed_hints,
            )
        case _:
            preprocessed_hints[index] = None
            return (
                None,
                stencils,
                images,
                preprocessed_hints,
            )
