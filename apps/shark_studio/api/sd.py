import gc
import torch
import time
import os
import json
import numpy as np
import transformers
import logging
import importlib
from packaging import version
from tqdm.auto import tqdm

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from turbine_models.custom_models.sd_inference import clip, unet, vae
from apps.shark_studio.api.controlnet import control_adapter_map
from apps.shark_studio.web.utils.state import status_label
from apps.shark_studio.web.utils.file_utils import (
    safe_name,
    get_resource_path,
    get_checkpoints_path,
)
from apps.shark_studio.modules.pipeline import SharkPipelineBase
from apps.shark_studio.modules.schedulers import get_schedulers
from apps.shark_studio.modules.prompt_encoding import (
    get_weighted_text_embeddings,
)
from apps.shark_studio.modules.img_processing import (
    resize_stencil,
    save_output_img,
    resamplers,
    resampler_list,
)

from apps.shark_studio.modules.ckpt_processing import (
    process_custom_pipe_weights,
)
from transformers import CLIPTokenizer, PreTrainedModel
from huggingface_hub import (
    ModelCard,
    create_repo,
    hf_hub_download,
    model_info,
    snapshot_download,
)
from huggingface_hub.utils import validate_hf_hub_args
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import (
    DiffusionPipeline,
    ImagePipelineOutput,
    AudioPipelineOutput,
    is_safetensors_compatible,
    variant_compatible_siblings,
    get_class_obj_and_candidates,
    maybe_raise_or_warn,
    _get_pipeline_class,
)

logger = logging.get_logger(__name__)

sd_model_map = {
    "clip": {
        "initializer": clip.export_clip_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
        ],
    },
    "vae_encode": {
        "initializer": vae.export_vae_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-detach-elementwise-from-named-ops,iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
        ],
    },
    "unet": {
        "initializer": unet.export_unet_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32}))",
        ],
    },
    "vae_decode": {
        "initializer": vae.export_vae_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-detach-elementwise-from-named-ops,iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
        ],
    },
}

INDEX_FILE = "diffusion_pytorch_model.bin"
CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "diffusers.utils"
TRANSFORMERS_DUMMY_MODULES_FOLDER = "transformers.utils"
CONNECTED_PIPES_KEYS = ["prior"]

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    #"onnxruntime.training": {
    #    "ORTModule": ["save_pretrained", "from_pretrained"],
    #},
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def load_sub_model(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    torch_dtype: torch.dtype,
    provider: Any,
    sess_options: Any,
    device_map: Optional[Union[Dict[str, torch.device], str]],
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]],
    offload_folder: Optional[Union[str, os.PathLike]],
    offload_state_dict: bool,
    model_variants: Dict[str, str],
    name: str,
    from_flax: bool,
    variant: str,
    low_cpu_mem_usage: bool,
    cached_folder: Union[str, os.PathLike],
    revision: str = None,
):
    """Helper method to load the module `name` from `library_name` and `class_name`"""
    # retrieve class candidates
    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # retrive load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
            TRANSFORMERS_DUMMY_MODULES_FOLDER
        )
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    load_method = getattr(class_obj, load_method_name)

    # add kwargs to loading method
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    loading_kwargs = {}
    if issubclass(class_obj, torch.nn.Module):
        loading_kwargs["torch_dtype"] = torch_dtype
    if issubclass(class_obj, diffusers_module.OnnxRuntimeModel):
        raise Exception("Support for onnx imports not implemented.")

    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)

    transformers_version = version.parse(version.parse(transformers.__version__).base_version)

    is_transformers_model = (
        issubclass(class_obj, PreTrainedModel)
        and transformers_version >= version.parse("4.20.0")
    )

    # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
    # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
    # This makes sure that the weights won't be initialized which significantly speeds up loading.
    if is_diffusers_model or is_transformers_model:
        loading_kwargs["device_map"] = device_map
        loading_kwargs["max_memory"] = max_memory
        loading_kwargs["offload_folder"] = offload_folder
        loading_kwargs["offload_state_dict"] = offload_state_dict
        loading_kwargs["variant"] = model_variants.pop(name, None)
        if from_flax:
            loading_kwargs["from_flax"] = True

        # the following can be deleted once the minimum required `transformers` version
        # is higher than 4.27
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            loading_kwargs.pop("variant")

        # if `from_flax` and model is transformer model, can currently not load with `low_cpu_mem_usage`
        if not (from_flax and is_transformers_model):
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            loading_kwargs["low_cpu_mem_usage"] = False

    # check if the module is in a subdirectory
    if os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
    else:
        # else load from the root directory
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    return loaded_sub_model

class SharkDiffusionPipeline(DiffusionPipeline, SharkPipelineBase):
    # This class is responsible for executing image generation and creating
    # /managing a set of compiled modules to run Stable Diffusion. The init
    # aims to be as general as possible, and the class will infer and compile
    # a list of necessary modules or a combined "pipeline module" for a
    # specified job based on the inference task.

    config_name = "model_index.json"
    model_cpu_offload_seq = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_onnx = False

    def __init__(
        self,
        base_model_id,
        height: int,
        width: int,
        batch_size: int,
        precision: str,
        device: str,
        custom_vae: str = None,
        num_loras: int = 0,
        import_ir: bool = True,
        is_controlled: bool = False,
    ):

        self.model_max_length = 77
        self.batch_size = batch_size
        self.precision = precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.height = height
        self.width = width
        self.scheduler_obj = {}
        compile_static_args = {
            "pipe": {
                "external_weights": "safetensors",
            },
            "clip": {"hf_model_name": base_model_id},
            "unet": {
                "hf_model_name": base_model_id,
                "unet_model": unet.UnetModel(
                    hf_model_name=base_model_id, hf_auth_token=None
                ),
                "batch_size": batch_size,
                # "is_controlled": is_controlled,
                # "num_loras": num_loras,
                "height": height,
                "width": width,
                "precision": precision,
                "max_length": self.model_max_length,
            },
            "vae_encode": {
                "hf_model_name": base_model_id,
                "vae_model": self.vae_encode,
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "precision": precision,
            },
            "vae_decode": {
                "hf_model_name": base_model_id,
                "vae_model": self.vae_decode,
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "precision": precision,
            },
        }
        super().__init__(sd_model_map, base_model_id, compile_static_args, device, import_ir)
        pipe_id_list = [
            safe_name(base_model_id),
            str(batch_size),
            str(static_kwargs["unet"]["max_length"]),
            f"{str(height)}x{str(width)}",
            precision,
        ]
        if num_loras > 0:
            pipe_id_list.append(str(num_loras) + "lora")
        if is_controlled:
            pipe_id_list.append("controlled")
        if custom_vae:
            pipe_id_list.append(custom_vae)
        self.pipe_id = "_".join(pipe_id_list)
        print(f"\n[LOG] Pipeline initialized with pipe_id: {self.pipe_id}.")
        del static_kwargs
        gc.collect()

    @property
    def device(self):
        r"""
        Returns:
            `device`: The device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.device

        return torch.device('cpu')
    
    def prepare_pipe(self, custom_weights, adapters, embeddings, is_img2img):
        print(f"\n[LOG] Preparing pipeline...")
        self.is_img2img = is_img2img
        self.schedulers = get_schedulers(self.base_model_id)

        self.weights_path = os.path.join(
            get_checkpoints_path(), self.safe_name(self.base_model_id)
        )
        if not os.path.exists(self.weights_path):
            os.mkdir(self.weights_path)

        for model in adapters:
            self.model_map[model] = adapters[model]

        for submodel in self.static_kwargs:
            if custom_weights:
                custom_weights_params, _ = process_custom_pipe_weights(custom_weights)
                if submodel not in ["clip", "clip2"]:
                    self.static_kwargs[submodel][
                        "external_weight_file"
                    ] = custom_weights_params
                else:
                    self.static_kwargs[submodel]["external_weight_path"] = os.path.join(
                        self.weights_path, submodel + ".safetensors"
                    )
            else:
                self.static_kwargs[submodel]["external_weight_path"] = os.path.join(
                    self.weights_path, submodel + ".safetensors"
                )

        self.get_compiled_map(pipe_id=self.pipe_id)
        print("\n[LOG] Pipeline successfully prepared for runtime.")
        return
    
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a Sharkified PyTorch diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            if pretrained_model_name_or_path.count("/") > 1:
                raise ValueError(
                    f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )
            cached_folder = cls.download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                from_flax=from_flax,
                use_safetensors=use_safetensors,
                use_onnx=use_onnx,
                custom_pipeline=custom_pipeline,
                custom_revision=custom_revision,
                variant=variant,
                load_connected_pipeline=load_connected_pipeline,
                **kwargs,
            )
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)

        # pop out "_ignore_files" as it is only needed for download
        config_dict.pop("_ignore_files", None)

        # 2. Define which model components should load variants
        # We retrieve the information by matching whether variant
        # model checkpoints exist in the subfolders
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in config_dict
                variant_exists = is_folder and any(
                    p.split(".")[1].startswith(variant) for p in os.listdir(folder_path)
                )
                if variant_exists:
                    model_variants[folder] = variant

        # 3. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        custom_class_name = None
        if os.path.isfile(os.path.join(cached_folder, f"{custom_pipeline}.py")):
            custom_pipeline = os.path.join(cached_folder, f"{custom_pipeline}.py")
        elif isinstance(config_dict["_class_name"], (list, tuple)) and os.path.isfile(
            os.path.join(cached_folder, f"{config_dict['_class_name'][0]}.py")
        ):
            custom_pipeline = os.path.join(cached_folder, f"{config_dict['_class_name'][0]}.py")
            custom_class_name = config_dict["_class_name"][1]

        pipeline_class = _get_pipeline_class(
            cls,
            config_dict,
            load_connected_pipeline=load_connected_pipeline,
            custom_pipeline=custom_pipeline,
            class_name=custom_class_name,
            cache_dir=cache_dir,
            revision=custom_revision,
        )

        # DEPRECATED: To be removed in 1.0.0
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
            version.parse(config_dict["_diffusers_version"]).base_version
        ) <= version.parse("0.5.1"):
            from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

            pipeline_class = StableDiffusionInpaintPipelineLegacy

        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs and make sure that optional component modules are filtered out
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Special case: safety_checker must be loaded separately when using `from_flax`
        if from_flax and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
            raise NotImplementedError(
                "The safety checker cannot be automatically loaded when loading weights `from_flax`."
                " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
                " separately if you need it."
            )

        # 5. Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        # import it here to avoid circular import
        from diffusers import pipelines

        # 6. Load each module in the pipeline
        for name, (library_name, class_name) in logging.tqdm(init_dict.items(), desc="Loading pipeline components..."):
            # 6.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            class_name = class_name[4:] if class_name.startswith("Flax") else class_name

            # 6.2 Define all importable classes
            is_pipeline_module = hasattr(pipelines, library_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            loaded_sub_model = None

            # 6.3 Use passed sub model or load class_name from library_name
            if name in passed_class_obj:
                # if the model is in a pipeline module, then we load it from the pipeline
                # check that passed_class_obj has correct parent class
                maybe_raise_or_warn(
                    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
                )

                loaded_sub_model = passed_class_obj[name]
            else:
                # load sub model
                loaded_sub_model = load_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=importable_classes,
                    pipelines=pipelines,
                    is_pipeline_module=is_pipeline_module,
                    pipeline_class=pipeline_class,
                    torch_dtype=torch_dtype,
                    provider=provider,
                    sess_options=sess_options,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder=offload_folder,
                    offload_state_dict=offload_state_dict,
                    model_variants=model_variants,
                    name=name,
                    from_flax=from_flax,
                    variant=variant,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cached_folder=cached_folder,
                    revision=revision,
                )
                logger.info(
                    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
                )

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        if pipeline_class._load_connected_pipes and os.path.isfile(os.path.join(cached_folder, "README.md")):
            modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
            connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for prefix in CONNECTED_PIPES_KEYS}
            load_kwargs = {
                "cache_dir": cache_dir,
                "resume_download": resume_download,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "torch_dtype": torch_dtype,
                "custom_pipeline": custom_pipeline,
                "custom_revision": custom_revision,
                "provider": provider,
                "sess_options": sess_options,
                "device_map": device_map,
                "max_memory": max_memory,
                "offload_folder": offload_folder,
                "offload_state_dict": offload_state_dict,
                "low_cpu_mem_usage": low_cpu_mem_usage,
                "variant": variant,
                "use_safetensors": use_safetensors,
            }

            def get_connected_passed_kwargs(prefix):
                connected_passed_class_obj = {
                    k.replace(f"{prefix}_", ""): w for k, w in passed_class_obj.items() if k.split("_")[0] == prefix
                }
                connected_passed_pipe_kwargs = {
                    k.replace(f"{prefix}_", ""): w for k, w in passed_pipe_kwargs.items() if k.split("_")[0] == prefix
                }

                connected_passed_kwargs = {**connected_passed_class_obj, **connected_passed_pipe_kwargs}
                return connected_passed_kwargs

            connected_pipes = {
                prefix: DiffusionPipeline.from_pretrained(
                    repo_id, **load_kwargs.copy(), **get_connected_passed_kwargs(prefix)
                )
                for prefix, repo_id in connected_pipes.items()
                if repo_id is not None
            }

            for prefix, connected_pipe in connected_pipes.items():
                # add connected pipes to `init_kwargs` with <prefix>_<component_name>, e.g. "prior_text_encoder"
                init_kwargs.update(
                    {"_".join([prefix, name]): component for name, component in connected_pipe.components.items()}
                )

        # 7. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 8. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 9. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        breakpoint()
        return model


def shark_sd_fn_dict_input(
    sd_kwargs: dict,
):
    print("[LOG] Submitting Request...")

    for key in sd_kwargs:
        if sd_kwargs[key] in [None, []]:
            sd_kwargs[key] = None
        if sd_kwargs[key] in ["None"]:
            sd_kwargs[key] = ""
        if key == "seed":
            sd_kwargs[key] = int(sd_kwargs[key])

    for i in range(1):
        generated_imgs = yield from shark_sd_fn(**sd_kwargs)
        yield generated_imgs


def shark_sd_fn(
    prompt,
    negative_prompt,
    sd_init_image: list,
    height: int,
    width: int,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: list,
    batch_count: int,
    batch_size: int,
    scheduler: str,
    base_model_id: str,
    custom_weights: str,
    custom_vae: str,
    precision: str,
    device: str,
    ondemand: bool,
    repeatable_seeds: bool,
    resample_type: str,
    controlnets: dict,
    embeddings: dict,
):
    sd_kwargs = locals()
    if not isinstance(sd_init_image, list):
        sd_init_image = [sd_init_image]
    is_img2img = True if sd_init_image[0] is not None else False

    print("\n[LOG] Performing Stable Diffusion Pipeline setup...")

    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    adapters = {}
    is_controlled = False
    control_mode = None
    hints = []
    num_loras = 0
    for i in embeddings:
        num_loras += 1 if embeddings[i] else 0
    if "model" in controlnets:
        for i, model in enumerate(controlnets["model"]):
            if "xl" not in base_model_id.lower():
                adapters[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map["runwayml/stable-diffusion-v1-5"][
                        model
                    ],
                    "strength": controlnets["strength"][i],
                }
            else:
                adapters[f"control_adapter_{model}"] = {
                    "hf_id": control_adapter_map["stabilityai/stable-diffusion-xl-1.0"][
                        model
                    ],
                    "strength": controlnets["strength"][i],
                }
            if model is not None:
                is_controlled = True
        control_mode = controlnets["control_mode"]
        for i in controlnets["hint"]:
            hints.append[i]

    submit_pipe_kwargs = {
        "pretrained_model_name_or_path": base_model_id,
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "precision": precision,
        "device": device,
        "custom_vae": custom_vae,
        "num_loras": num_loras,
        "import_ir": cmd_opts.import_mlir,
        "is_controlled": is_controlled,
    }
    submit_prep_kwargs = {
        "custom_weights": custom_weights,
        "adapters": adapters,
        "embeddings": embeddings,
        "is_img2img": is_img2img,
    }
    submit_run_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": sd_init_image,
        "steps": steps,
        "scheduler": scheduler,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "ondemand": ondemand,
        "repeatable_seeds": repeatable_seeds,
        "resample_type": resample_type,
        "control_mode": control_mode,
        "hints": hints,
    }
    if (
        not global_obj.get_sd_obj()
        or global_obj.get_pipe_kwargs() != submit_pipe_kwargs
    ):
        print("\n[LOG] Initializing new pipeline...")
        global_obj.clear_cache()
        gc.collect()

        # Initializes the pipeline and retrieves IR based on all
        # parameters that are static in the turbine output format,
        # which is currently MLIR in the torch dialect.

        sd_pipe = SharkDiffusionPipeline.from_pretrained(
            **submit_pipe_kwargs,
        )
        global_obj.set_sd_obj(sd_pipe)
        global_obj.set_pipe_kwargs(submit_pipe_kwargs)
    if (
        not global_obj.get_prep_kwargs()
        or global_obj.get_prep_kwargs() != submit_prep_kwargs
    ):
        global_obj.set_prep_kwargs(submit_prep_kwargs)
        global_obj.get_sd_obj().prepare_pipe(**submit_prep_kwargs)

    generated_imgs = []
    for current_batch in range(batch_count):
        start_time = time.time()
        out_imgs = global_obj.get_sd_obj().generate_images(**submit_run_kwargs)
        total_time = time.time() - start_time
        text_output = f"Total image(s) generation time: {total_time:.4f}sec"
        print(f"\n[LOG] {text_output}")
        # if global_obj.get_sd_status() == SD_STATE_CANCEL:
        #     break
        # else:
        save_output_img(
            out_imgs[current_batch],
            seed,
            sd_kwargs,
        )
        generated_imgs.extend(out_imgs)
        yield generated_imgs, status_label(
            "Stable Diffusion", current_batch + 1, batch_count, batch_size
        )
    return generated_imgs, ""


def cancel_sd():
    print("Inject call to cancel longer API calls.")
    return


def view_json_file(file_path):
    content = ""
    with open(file_path, "r") as fopen:
        content = fopen.read()
    return content


if __name__ == "__main__":
    from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
    import apps.shark_studio.web.utils.globals as global_obj

    global_obj._init()

    sd_json = view_json_file(get_resource_path("../configs/default_sd_config.json"))
    sd_kwargs = json.loads(sd_json)
    for arg in vars(cmd_opts):
        if arg in sd_kwargs:
            sd_kwargs[arg] = getattr(cmd_opts, arg)
    for i in shark_sd_fn_dict_input(sd_kwargs):
        print(i)
