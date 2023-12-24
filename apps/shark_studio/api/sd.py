import gc
import torch
import time
import os
import json
import numpy as np
import transformers
import importlib
import inspect
import re
import fnmatch
import sys


from requests.exceptions import HTTPError
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
from apps.shark_studio.modules.meta_model import SharkMetaLoader
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
    get_class_from_dynamic_module,
    maybe_raise_or_warn,
    load_sub_model,
    _unwrap_model,
    LOADABLE_CLASSES,
)
from diffusers.utils import logging, PushToHubMixin, CONFIG_NAME
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.configuration_utils import ConfigMixin

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
    # "vae_encode": {
    #     "initializer": vae.export_vae_model,
    #     "ireec_flags": [
    #         "--iree-flow-collapse-reduction-dims",
    #         "--iree-opt-const-expr-hoisting=False",
    #         "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
    #         "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-detach-elementwise-from-named-ops,iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))",
    #         "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-preprocessing-pad-linalg-ops{pad-size=16}))",
    #     ],
    # },
    "unet": {
        "initializer": unet.export_unet_model,
        "ireec_flags": [
            "--iree-flow-collapse-reduction-dims",
            "--iree-opt-const-expr-hoisting=False",
            "--iree-codegen-linalg-max-constant-fold-elements=9223372036854775807",
            "--iree-preprocessing-pass-pipeline=builtin.module(func.func(iree-global-opt-convert-1x1-filter-conv2d-to-matmul,iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=32},iree-linalg-ext-convert-conv2d-to-winograd))",
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

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def setup_shark(
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
    # self.dtype = torch.float16 if precision == "fp16" else torch.float32
    self.height = height
    self.width = width
    self.scheduler_obj = {}
    self.base_model_id = base_model_id #pipe.config["_name_or_path"]
    self.compile_static_args = {
        "pipe": {
            "external_weights": "safetensors",
        },
        "clip": { "hf_model_name": base_model_id },
        "unet": {
            "hf_model_name": base_model_id,
            "unet_model": unet.UnetModel(base_model_id, self.unet),
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
            "vae_model": vae.VaeModel(base_model_id, self.vae),
            "batch_size": batch_size,
            "height": height,
            "width": width,
            "precision": precision,
        },
        "vae_decode": {
            "hf_model_name": base_model_id,
            "vae_model": vae.VaeModel(base_model_id, self.vae),
            "batch_size": batch_size,
            "height": height,
            "width": width,
            "precision": precision,
        },
    }
    pipe_id_list = [
        safe_name(self.base_model_id),
        str(batch_size),
        str(self.model_max_length),
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
    gc.collect()
    

def prepare_pipe(self, custom_weights, adapters, embeddings, is_img2img):
    print(f"\n[LOG] Preparing pipeline...")
    self.is_img2img = is_img2img
    self.schedulers = get_schedulers(self.base_model_id)

    self.weights_path = os.path.join(
        get_checkpoints_path(), self.shark_meta.safe_name(self.base_model_id)
    )
    if not os.path.exists(self.weights_path):
        os.mkdir(self.weights_path)

    for model in adapters:
        self.model_map[model] = adapters[model]

    for submodel in self.compile_static_args:
        if custom_weights:
            custom_weights_params, _ = process_custom_pipe_weights(custom_weights)
            if submodel not in ["clip", "clip2"]:
                self.compile_static_args[submodel][
                    "external_weight_file"
                ] = custom_weights_params
            else:
                self.compile_static_args[submodel]["external_weight_path"] = os.path.join(
                    self.weights_path, submodel + ".safetensors"
                )
        else:
            self.compile_static_args[submodel]["external_weight_path"] = os.path.join(
                self.weights_path, submodel + ".safetensors"
            )

    self.shark_meta.get_compiled_map(pipe_id=self.pipe_id, static_kwargs=self.compile_static_args)
    print("\n[LOG] Pipeline successfully prepared for runtime.")
    return


def _get_pipeline_class(
    class_obj,
    config,
    load_connected_pipeline=False,
    custom_pipeline=None,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
    shark_device="cpu",
):
    if custom_pipeline is not None:
        if custom_pipeline.endswith(".py"):
            path = Path(custom_pipeline)
            # decompose into folder & file
            file_name = path.name
            custom_pipeline = path.parent.absolute()
        elif repo_id is not None:
            file_name = f"{custom_pipeline}.py"
            custom_pipeline = repo_id
        else:
            file_name = CUSTOM_PIPELINE_FILE_NAME

        if repo_id is not None and hub_revision is not None:
            # if we load the pipeline code from the Hub
            # make sure to overwrite the `revison`
            revision = hub_revision

        return get_class_from_dynamic_module(
            custom_pipeline,
            module_file=file_name,
            class_name=class_name,
            cache_dir=cache_dir,
            revision=revision,
        )

    if class_obj != SharkDiffusionPipeline:
        return class_obj

    diffusers_module = importlib.import_module("diffusers")
    class_name = config["_class_name"]
    class_name = class_name[4:] if class_name.startswith("Flax") else class_name

    pipeline_cls = getattr(diffusers_module, class_name)

    if load_connected_pipeline:
        from diffusers.pipelines.auto_pipeline import _get_connected_pipeline

        connected_pipeline_cls = _get_connected_pipeline(pipeline_cls)
        if connected_pipeline_cls is not None:
            logger.info(
                f"Loading connected pipeline {connected_pipeline_cls.__name__} instead of {pipeline_cls.__name__} as specified via `load_connected_pipeline=True`"
            )
        else:
            logger.info(f"{pipeline_cls.__name__} has no connected pipeline class. Loading {pipeline_cls.__name__}.")

        pipeline_cls = connected_pipeline_cls or pipeline_cls

    return pipeline_cls



class SharkDiffusionPipeline(ConfigMixin, PushToHubMixin):

    r'''
    Instantiates a diffusers pipeline and replaces any model preparation/device
    methods and properties with our custom model loading and runtime 
    (provided by SharkPipelineBase).

    This class is responsible for creating and managing a set of compiled
    modules to run a diffusers 'DiffusionPipeline'. The init
    aims to be as general as possible, and the class will infer and compile
    a list of necessary modules or a combined "pipeline module" via Turbine
    for a specified job based on the inference task.
    '''    
    config_name = "model_index.json"
    model_cpu_offload_seq = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_onnx = False

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns:
            `torch.dtype`: The torch dtype on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return module.dtype

        return torch.float32
    
    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        for module in modules:
            return torch.device("cpu")

        return torch.device("cpu")
    
    def register_modules(self, **kwargs):
        print("sharkified")
        # import it here to avoid circular import
        diffusers_module = importlib.import_module("diffusers")
        pipelines = getattr(diffusers_module, "pipelines")

        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                # register the config from the original module, not the dynamo compiled one
                not_compiled_module = _unwrap_model(module)

                library = not_compiled_module.__module__.split(".")[0]

                # check if the module is a pipeline module
                module_path_items = not_compiled_module.__module__.split(".")
                pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

                path = not_compiled_module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if is_pipeline_module:
                    library = pipeline_dir
                elif library not in LOADABLE_CLASSES:
                    library = not_compiled_module.__module__

                # retrieve class_name
                class_name = not_compiled_module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__ and hasattr(self.config, name):
            # We need to overwrite the config if name exists in config
            if isinstance(getattr(self.config, name), (tuple, list)):
                if value is not None and self.config[name][0] is not None:
                    class_library_tuple = (value.__module__.split(".")[0], value.__class__.__name__)
                else:
                    class_library_tuple = (None, None)

                self.register_to_config(**{name: class_library_tuple})
            else:
                self.register_to_config(**{name: value})

        super().__setattr__(name, value)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~DiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a pipeline to. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_diffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warn(f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved.")
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory)

        if push_to_hub:
            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )


    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], device, **kwargs):
        
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
                shark_device=device,
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
            shark_device=device,
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
        
        # 10. (SHARK): Monkey-patch in our classmethods to the auto-generated SD pipeline class.
        pipeline_class.setup_shark = setup_shark
        pipeline_class.prepare_pipe = prepare_pipe
        pipeline_class.shark_meta = SharkMetaLoader(sd_model_map, device)

        # class shark_pipeline_class(pipeline_class):
        #     def __init__(self, pipeline_class, device, model_map, **init_kwargs):
        #         super().__init__(device, model_map, pipeline_class, **init_kwargs)


        #     def __getattribute__(cls, __name: str) -> Any:
        #         return super().__getattribute__(__name)

        #pipeline_class = shark_pipeline_class

        # 8. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 9. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)


        
        return model
    
    @classmethod
    @validate_hf_hub_args
    def download(cls, pretrained_model_name, shark_device, **kwargs) -> Union[str, os.PathLike]:
 
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        allow_patterns = None
        ignore_patterns = None

        model_info_call_error: Optional[Exception] = None
        if not local_files_only:
            try:
                info = model_info(pretrained_model_name, token=token, revision=revision)
            except HTTPError as e:
                logger.warn(f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache.")
                local_files_only = True
                model_info_call_error = e  # save error to reraise it if model is not cached locally

        if not local_files_only:
            config_file = hf_hub_download(
                pretrained_model_name,
                cls.config_name,
                cache_dir=cache_dir,
                revision=revision,
                proxies=proxies,
                force_download=force_download,
                resume_download=resume_download,
                token=token,
            )

            config_dict = cls._dict_from_json_file(config_file)
            ignore_filenames = config_dict.pop("_ignore_files", [])

            # retrieve all folder_names that contain relevant files
            folder_names = [k for k, v in config_dict.items() if isinstance(v, list) and k != "_class_name"]

            filenames = {sibling.rfilename for sibling in info.siblings}
            model_filenames, variant_filenames = variant_compatible_siblings(filenames, variant=variant)

            diffusers_module = importlib.import_module("diffusers")
            pipelines = getattr(diffusers_module, "pipelines")

            # optionally create a custom component <> custom file mapping
            custom_components = {}
            for component in folder_names:
                module_candidate = config_dict[component][0]

                if module_candidate is None or not isinstance(module_candidate, str):
                    continue

                # We compute candidate file path on the Hub. Do not use `os.path.join`.
                candidate_file = f"{component}/{module_candidate}.py"

                if candidate_file in filenames:
                    custom_components[component] = module_candidate
                elif module_candidate not in LOADABLE_CLASSES and not hasattr(pipelines, module_candidate):
                    raise ValueError(
                        f"{candidate_file} as defined in `model_index.json` does not exist in {pretrained_model_name} and is not a module in 'diffusers/pipelines'."
                    )

            # remove ignored filenames
            model_filenames = set(model_filenames) - set(ignore_filenames)
            variant_filenames = set(variant_filenames) - set(ignore_filenames)

            # if the whole pipeline is cached we don't have to ping the Hub

            model_folder_names = {os.path.split(f)[0] for f in model_filenames if os.path.split(f)[0] in folder_names}

            custom_class_name = None
            if custom_pipeline is None and isinstance(config_dict["_class_name"], (list, tuple)):
                custom_pipeline = config_dict["_class_name"][0]
                custom_class_name = config_dict["_class_name"][1]

            # all filenames compatible with variant will be added
            allow_patterns = list(model_filenames)

            # allow all patterns from non-model folders
            # this enables downloading schedulers, tokenizers, ...
            allow_patterns += [f"{k}/*" for k in folder_names if k not in model_folder_names]
            # add custom component files
            allow_patterns += [f"{k}/{f}.py" for k, f in custom_components.items()]
            # add custom pipeline file
            allow_patterns += [f"{custom_pipeline}.py"] if f"{custom_pipeline}.py" in filenames else []
            # also allow downloading config.json files with the model
            allow_patterns += [os.path.join(k, "config.json") for k in model_folder_names]

            allow_patterns += [
                SCHEDULER_CONFIG_NAME,
                CONFIG_NAME,
                cls.config_name,
                CUSTOM_PIPELINE_FILE_NAME,
            ]

            load_pipe_from_hub = custom_pipeline is not None and f"{custom_pipeline}.py" in filenames
            load_components_from_hub = len(custom_components) > 0

            if load_pipe_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {custom_pipeline}.py which must be executed to correctly "
                    f"load the model. You can inspect the repository content at https://hf.co/{pretrained_model_name}/blob/main/{custom_pipeline}.py.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            if load_components_from_hub and not trust_remote_code:
                raise ValueError(
                    f"The repository for {pretrained_model_name} contains custom code in {'.py, '.join([os.path.join(k, v) for k,v in custom_components.items()])} which must be executed to correctly "
                    f"load the model. You can inspect the repository content at {', '.join([f'https://hf.co/{pretrained_model_name}/{k}/{v}.py' for k,v in custom_components.items()])}.\n"
                    f"Please pass the argument `trust_remote_code=True` to allow custom code to be run."
                )

            # retrieve passed components that should not be downloaded
            pipeline_class = _get_pipeline_class(
                cls,
                config_dict,
                load_connected_pipeline=load_connected_pipeline,
                custom_pipeline=custom_pipeline,
                repo_id=pretrained_model_name if load_pipe_from_hub else None,
                hub_revision=revision,
                class_name=custom_class_name,
                cache_dir=cache_dir,
                revision=custom_revision,
                shark_device=shark_device,
            )

            expected_components, _ = cls._get_signature_keys(pipeline_class)
            passed_components = [k for k in expected_components if k in kwargs]

            if (
                use_safetensors
                and not allow_pickle
                and not is_safetensors_compatible(
                    model_filenames, variant=variant, passed_components=passed_components
                )
            ):
                raise EnvironmentError(
                    f"Could not find the necessary `safetensors` weights in {model_filenames} (variant={variant})"
                )
            if from_flax:
                ignore_patterns = ["*.bin", "*.safetensors", "*.onnx", "*.pb"]
            elif use_safetensors and is_safetensors_compatible(
                model_filenames, variant=variant, passed_components=passed_components
            ):
                ignore_patterns = ["*.bin", "*.msgpack"]

                use_onnx = use_onnx if use_onnx is not None else pipeline_class._is_onnx
                if not use_onnx:
                    ignore_patterns += ["*.onnx", "*.pb"]

                safetensors_variant_filenames = {f for f in variant_filenames if f.endswith(".safetensors")}
                safetensors_model_filenames = {f for f in model_filenames if f.endswith(".safetensors")}
                if (
                    len(safetensors_variant_filenames) > 0
                    and safetensors_model_filenames != safetensors_variant_filenames
                ):
                    logger.warn(
                        f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(safetensors_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                    )
            else:
                ignore_patterns = ["*.safetensors", "*.msgpack"]

                use_onnx = use_onnx if use_onnx is not None else pipeline_class._is_onnx
                if not use_onnx:
                    ignore_patterns += ["*.onnx", "*.pb"]

                bin_variant_filenames = {f for f in variant_filenames if f.endswith(".bin")}
                bin_model_filenames = {f for f in model_filenames if f.endswith(".bin")}
                if len(bin_variant_filenames) > 0 and bin_model_filenames != bin_variant_filenames:
                    logger.warn(
                        f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(bin_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(bin_model_filenames - bin_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                    )

            # Don't download any objects that are passed
            allow_patterns = [
                p for p in allow_patterns if not (len(p.split("/")) == 2 and p.split("/")[0] in passed_components)
            ]

            if pipeline_class._load_connected_pipes:
                allow_patterns.append("README.md")

            # Don't download index files of forbidden patterns either
            ignore_patterns = ignore_patterns + [f"{i}.index.*json" for i in ignore_patterns]

            re_ignore_pattern = [re.compile(fnmatch.translate(p)) for p in ignore_patterns]
            re_allow_pattern = [re.compile(fnmatch.translate(p)) for p in allow_patterns]

            expected_files = [f for f in filenames if not any(p.match(f) for p in re_ignore_pattern)]
            expected_files = [f for f in expected_files if any(p.match(f) for p in re_allow_pattern)]

            snapshot_folder = Path(config_file).parent
            pipeline_is_cached = all((snapshot_folder / f).is_file() for f in expected_files)

            if pipeline_is_cached and not force_download:
                # if the pipeline is cached, we can directly return it
                # else call snapshot_download
                return snapshot_folder

        user_agent = {"pipeline_class": cls.__name__}
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                pretrained_model_name,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            # retrieve pipeline class from local file
            cls_name = cls.load_config(os.path.join(cached_folder, "model_index.json")).get("_class_name", None)
            cls_name = cls_name[4:] if isinstance(cls_name, str) and cls_name.startswith("Flax") else cls_name

            diffusers_module = importlib.import_module(__name__.split(".")[0])
            pipeline_class = getattr(diffusers_module, cls_name, None) if isinstance(cls_name, str) else None

            if pipeline_class is not None and pipeline_class._load_connected_pipes:
                modelcard = ModelCard.load(os.path.join(cached_folder, "README.md"))
                connected_pipes = sum([getattr(modelcard.data, k, []) for k in CONNECTED_PIPES_KEYS], [])
                for connected_pipe_repo_id in connected_pipes:
                    download_kwargs = {
                        "cache_dir": cache_dir,
                        "resume_download": resume_download,
                        "force_download": force_download,
                        "proxies": proxies,
                        "local_files_only": local_files_only,
                        "token": token,
                        "variant": variant,
                        "use_safetensors": use_safetensors,
                    }
                    DiffusionPipeline.download(connected_pipe_repo_id, **download_kwargs)

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise EnvironmentError(
                    f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occured"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error
    
    def to(self, *args, **kwargs):

        torch_dtype = kwargs.pop("torch_dtype", None)
        torch_device = kwargs.pop("torch_device", None)
        dtype_kwarg = kwargs.pop("dtype", None)
        device_kwarg = kwargs.pop("device", None)
        silence_dtype_warnings = kwargs.pop("silence_dtype_warnings", False)

        dtype = torch_dtype or dtype_kwarg

        device = torch_device or device_kwarg

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            return False

        def module_is_offloaded(module):
            return False

        # .to("cuda") would raise an error if the pipeline is sequentially offloaded, so we raise our own to make it clearer
        pipeline_is_sequentially_offloaded = any(
            module_is_sequentially_offloaded(module) for _, module in self.components.items()
        )
        if pipeline_is_sequentially_offloaded:
            raise ValueError(
                "Sequential offload not supported."
            )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded:
            logger.warning(
                "Sequential offload not supported."
            )

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]

        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for module in modules:
            is_loaded_in_8bit = hasattr(module, "is_loaded_in_8bit") and module.is_loaded_in_8bit

            if is_loaded_in_8bit and dtype is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and conversion to {torch_dtype} is not yet supported. Module is still in 8bit precision."
                )

            if is_loaded_in_8bit and device is not None:
                logger.warning(
                    f"The module '{module.__class__.__name__}' has been loaded in 8bit and moving it to {torch_dtype} via `.to()` is not yet supported. Module is still on {module.device}."
                )
            else:
                module.to(device, dtype)

            if (
                module.dtype == torch.float16
                and str(device) in ["cpu"]
                and not silence_dtype_warnings
                and not is_offloaded
            ):
                logger.warning(
                    "Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device. It"
                    " is not recommended to move them to `cpu` as running them will fail. Please make"
                    " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                    " support for`float16` operations on this device in PyTorch. Please, remove the"
                    " `torch_dtype=torch.float16` argument, or use another device for inference."
                )
        return self
    
    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return expected_modules, optional_parameters


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

        sd_pipe = DiffusionPipeline.from_pretrained(
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
