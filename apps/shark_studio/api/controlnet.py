# from turbine_models.custom_models.controlnet import control_adapter, preprocessors
import os
import PIL
import numpy as np
from apps.shark_studio.modules.pipeline import SharkPipelineBase
from apps.shark_studio.web.utils.file_utils import (
    get_generated_imgs_path,
)
from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    load_vmfb_using_mmap,
    clean_device_info,
    get_iree_target_triple,
)
from apps.shark_studio.web.utils.file_utils import (
    safe_name,
    get_resource_path,
    get_checkpoints_path,
)
import cv2
from datetime import datetime
from PIL import Image
from gradio.components.image_editor import (
    EditorValue,
)
# from turbine_models.custom_models.sd_inference import export_controlnet_model, ControlNetModel
import gc

class control_adapter:
    def __init__(
        self,
        model: str,
    ):
        self.model = None

    def export_control_adapter_model(model_keyword):
        if model_keyword == "canny":
            return export_controlnet_model(
                ControlNetModel("lllyasviel/control_v11p_sd15_canny"),
                "lllyasviel/control_v11p_sd15_canny",
                1,
                512,
                512,
            )

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

ireec_flags = [
    "--iree-flow-collapse-reduction-dims",
    "--iree-opt-const-expr-hoisting=False",
    "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
    "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))",
    "--iree-flow-inline-constants-max-byte-length=1" # Stopgap, take out when not needed
]

control_adapter_map = {
    "runwayml/stable-diffusion-v1-5": {
        "canny": {"initializer": control_adapter.export_control_adapter_model},
        "openpose": {"initializer": control_adapter.export_control_adapter_model},
        "scribble": {"initializer": control_adapter.export_control_adapter_model},
        "zoedepth": {"initializer": control_adapter.export_control_adapter_model},
    },
    "sdxl": {
        "canny": {"initializer": control_adapter.export_xl_control_adapter_model},
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
        self.compiled_model = None

    def compile(self):
        if self.compiled_model is not None:
            return
        if "canny" in self.model:
            return
        if "openpose" in self.model:
            pass
        print("compile not implemented for preprocessor.")

    def run(self, inputs):
        if self.compiled_model is None:
            self.compile()
        if "canny" in self.model:
            out = cv2.Canny(*inputs)
            return out
        if "openpose" in self.model:
            self.compiled_model(*inputs)

    def __call__(self, *inputs):
        return self.run(inputs)


class SharkControlnetPipeline(SharkPipelineBase):
    def __init__(
        self,
        # model_map: dict,
        # static_kwargs: dict,
        device: str,
        # import_mlir: bool = True,
    ):
        self.model_map = control_adapter_map
        self.pipe_map = {}
        # self.static_kwargs = static_kwargs
        self.static_kwargs = {}
        self.triple = get_iree_target_triple(device)
        self.device, self.device_id = clean_device_info(device)
        self.import_mlir = False
        self.iree_module_dict = {}
        self.tmp_dir = get_resource_path(os.path.join("..", "shark_tmp"))
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        self.tempfiles = {}
        self.pipe_vmfb_path = ""
        self.ireec_flags = ireec_flags

    def get_compiled_map(self, model, init_kwargs={}):
        self.pipe_map[model] = {}
        if model in self.iree_module_dict:
            return
        elif model not in self.tempfiles:
            # if model in self.static_kwargs[model]:
            #     init_kwargs = self.static_kwargs[model]
            init_kwargs = {}
            # for key in self.static_kwargs["pipe"]:
            #         if key not in init_kwargs:
            #             init_kwargs[key] = self.static_kwargs["pipe"][key]
            self.import_torch_ir(model, init_kwargs)
            self.get_compiled_map(model)
        else:
            # weights_path = self.get_io_params(model)

            self.iree_module_dict[model] = get_iree_compiled_module(
                self.tempfiles[model],
                device=self.device,
                frontend="torch",
                mmap=True,
                # external_weight_file=weights_path,
                external_weight_file=None,
                extra_args=self.ireec_flags,
                write_to=os.path.join(self.pipe_vmfb_path, model + ".vmfb")
            )

    def import_torch_ir(self, model, kwargs):
        # torch_ir = self.model_map[model]["initializer"](
        #     **self.safe_dict(kwargs), compile_to="torch"
        # )
        tmp_kwargs = {
            "model_keyword": "canny"
        }
        torch_ir = self.model_map["sd15"][model]["initializer"](
            **self.safe_dict(tmp_kwargs) #, compile_to="torch"
        )

        self.tempfiles[model] = os.path.join(
            self.tmp_dir, f"{model}.torch.tempfile"
        )

        with open(self.tempfiles[model], "w+") as f:
            f.write(torch_ir)
        del torch_ir
        gc.collect()
        return

    def get_precompiled(self, model):
        vmfbs = []
        for dirpath, dirnames, filenames in os.walk(self.pipe_vmfb_path):
            vmfbs.extend(filenames)
            break
        for file in vmfbs:
            if model in file:
                self.pipe_map[model]["vmfb_path"] = os.path.join(
                    self.pipe_vmfb_path, file
                )
        return


def cnet_preview(model, input_image):
    curr_datetime = datetime.now().strftime("%Y-%m-%d.%H-%M-%S")
    control_imgs_path = os.path.join(get_generated_imgs_path(), "control_hints")
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
