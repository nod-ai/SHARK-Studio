import ast
import copy
import functools
import glob
import inspect
import queue
import sys
import os
import time
import traceback
import types
import typing
import warnings
from datetime import datetime
import filelock
import requests
import psutil
from requests import ConnectTimeout, JSONDecodeError
from urllib3.exceptions import (
    ConnectTimeoutError,
    MaxRetryError,
    ConnectionError,
)
from requests.exceptions import ConnectionError as ConnectionError2
from requests.exceptions import ReadTimeout as ReadTimeout2

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from evaluate_params import eval_func_param_names, no_default_param_names
from enums import (
    DocumentChoices,
    LangChainMode,
    no_lora_str,
    model_token_mapping,
    no_model_str,
    source_prefix,
    source_postfix,
    LangChainAction,
)
from loaders import get_loaders
from utils import (
    set_seed,
    clear_torch_cache,
    save_generate_output,
    NullContext,
    wrapped_partial,
    EThread,
    get_githash,
    import_matplotlib,
    get_device,
    makedirs,
    get_kwargs,
    start_faulthandler,
    get_hf_server,
    FakeTokenizer,
    remove,
)

start_faulthandler()
import_matplotlib()

SEED = 1236
set_seed(SEED)

from typing import Union

# import fire
import torch
from transformers import GenerationConfig, AutoModel, TextIteratorStreamer

from prompter import (
    Prompter,
    inv_prompt_type_to_model_lower,
    non_hf_types,
    PromptType,
    get_prompt,
    generate_prompt,
)
from stopping import get_stopping

langchain_modes = [x.value for x in list(LangChainMode)]

langchain_actions = [x.value for x in list(LangChainAction)]

scratch_base_dir = "/tmp/"


class Langchain:
    def __init__(self, device="cuda", precision="fp16"):
        super().__init__()
        self.device = device
        self.precision = precision

    def get_config(
        self,
        base_model,
        use_auth_token=False,
        trust_remote_code=True,
        offload_folder=None,
        triton_attn=False,
        long_sequence=True,
        return_model=False,
        raise_exception=False,
    ):
        from accelerate import init_empty_weights

        with init_empty_weights():
            from transformers import AutoConfig

            try:
                config = AutoConfig.from_pretrained(
                    base_model,
                    use_auth_token=use_auth_token,
                    trust_remote_code=trust_remote_code,
                    offload_folder=offload_folder,
                )
            except OSError as e:
                if raise_exception:
                    raise
                if "not a local folder and is not a valid model identifier listed on" in str(
                    e
                ) or "404 Client Error" in str(
                    e
                ):
                    # e.g. llama, gpjt, etc.
                    # e.g. HF TGI but not model on HF or private etc.
                    # HF TGI server only should really require prompt_type, not HF model state
                    return None, None
                else:
                    raise
            if triton_attn and "mpt-" in base_model.lower():
                config.attn_config["attn_impl"] = "triton"
            if long_sequence:
                if "mpt-7b-storywriter" in base_model.lower():
                    config.update({"max_seq_len": 83968})
                if "mosaicml/mpt-7b-chat" in base_model.lower():
                    config.update({"max_seq_len": 4096})
                if "mpt-30b" in base_model.lower():
                    config.update({"max_seq_len": 2 * 8192})
            if return_model and issubclass(
                config.__class__, tuple(AutoModel._model_mapping.keys())
            ):
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
            else:
                # can't infer
                model = None
        if "falcon" in base_model.lower():
            config.use_cache = False

        return config, model

    def get_non_lora_model(
        self,
        base_model,
        model_loader,
        load_half,
        load_gptq,
        use_safetensors,
        model_kwargs,
        reward_type,
        config,
        model,
        gpu_id=0,
    ):
        """
        Ensure model gets on correct device
        """

        device_map = None
        if model is not None:
            # NOTE: Can specify max_memory={0: max_mem, 1: max_mem}, to shard model
            # NOTE: Some models require avoiding sharding some layers,
            # then would pass no_split_module_classes and give list of those layers.
            from accelerate import infer_auto_device_map

            device_map = infer_auto_device_map(
                model,
                dtype=torch.float16 if load_half else torch.float32,
            )
            if hasattr(model, "model"):
                device_map_model = infer_auto_device_map(
                    model.model,
                    dtype=torch.float16 if load_half else torch.float32,
                )
                device_map.update(device_map_model)

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0

        if device_map is None:
            if self.device == "cuda":
                if n_gpus > 0:
                    if gpu_id >= 0:
                        # FIXME: If really distributes model, tend to get things like: ValueError: gpt_neox.embed_in.weight doesn't have any device set.
                        # So avoid for now, just put on first GPU, unless score_model, put on last
                        if reward_type:
                            device_map = {"": n_gpus - 1}
                        else:
                            device_map = {"": min(n_gpus - 1, gpu_id)}
                    if gpu_id == -1:
                        device_map = {"": "cuda"}
            else:
                device_map = {"": "cpu"}
                model_kwargs["load_in_8bit"] = False
                model_kwargs["load_in_4bit"] = False
        print("device_map: %s" % device_map, flush=True)

        load_in_8bit = model_kwargs.get("load_in_8bit", False)
        load_in_4bit = model_kwargs.get("load_in_4bit", False)
        model_kwargs["device_map"] = device_map
        model_kwargs["use_safetensors"] = use_safetensors
        self.pop_unused_model_kwargs(model_kwargs)

        if load_gptq:
            model_kwargs.pop("torch_dtype", None)
            model_kwargs.pop("device_map")
            model = model_loader(
                model_name_or_path=base_model,
                model_basename=load_gptq,
                **model_kwargs,
            )
        elif load_in_8bit or load_in_4bit or not load_half:
            model = model_loader(
                base_model,
                config=config,
                **model_kwargs,
            )
        else:
            model = model_loader(
                base_model,
                config=config,
                **model_kwargs,
            ).half()
        return model

    def get_client_from_inference_server(
        self,
        inference_server,
        base_model=None,
        raise_connection_exception=False,
    ):
        inference_server, headers = get_hf_server(inference_server)
        # preload client since slow for gradio case especially
        from gradio_utils.grclient import GradioClient

        gr_client = None
        hf_client = None
        if headers is None:
            try:
                print(
                    "GR Client Begin: %s %s" % (inference_server, base_model),
                    flush=True,
                )
                # first do sanity check if alive, else gradio client takes too long by default
                requests.get(
                    inference_server,
                    timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                )
                gr_client = GradioClient(inference_server)
                print("GR Client End: %s" % inference_server, flush=True)
            except (OSError, ValueError) as e:
                # Occurs when wrong endpoint and should have been HF client, so don't hard raise, just move to HF
                gr_client = None
                print(
                    "GR Client Failed %s %s: %s"
                    % (inference_server, base_model, str(e)),
                    flush=True,
                )
            except (
                ConnectTimeoutError,
                ConnectTimeout,
                MaxRetryError,
                ConnectionError,
                ConnectionError2,
                JSONDecodeError,
                ReadTimeout2,
                KeyError,
            ) as e:
                t, v, tb = sys.exc_info()
                ex = "".join(traceback.format_exception(t, v, tb))
                print(
                    "GR Client Failed %s %s: %s"
                    % (inference_server, base_model, str(ex)),
                    flush=True,
                )
                if raise_connection_exception:
                    raise

        if gr_client is None:
            res = None
            from text_generation import Client as HFClient

            print("HF Client Begin: %s %s" % (inference_server, base_model))
            try:
                hf_client = HFClient(
                    inference_server,
                    headers=headers,
                    timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                )
                # quick check valid TGI endpoint
                res = hf_client.generate("What?", max_new_tokens=1)
                hf_client = HFClient(
                    inference_server, headers=headers, timeout=300
                )
            except (
                ConnectTimeoutError,
                ConnectTimeout,
                MaxRetryError,
                ConnectionError,
                ConnectionError2,
                JSONDecodeError,
                ReadTimeout2,
                KeyError,
            ) as e:
                hf_client = None
                t, v, tb = sys.exc_info()
                ex = "".join(traceback.format_exception(t, v, tb))
                print(
                    "HF Client Failed %s %s: %s"
                    % (inference_server, base_model, str(ex))
                )
                if raise_connection_exception:
                    raise
            print(
                "HF Client End: %s %s : %s"
                % (inference_server, base_model, res)
            )
        return inference_server, gr_client, hf_client

    def get_model(
        self,
        load_8bit: bool = False,
        load_4bit: bool = False,
        load_half: bool = False,
        load_gptq: str = "",
        use_safetensors: bool = False,
        infer_devices: bool = True,
        device: str = None,
        base_model: str = "",
        inference_server: str = "",
        tokenizer_base_model: str = "",
        lora_weights: str = "",
        gpu_id: int = 0,
        reward_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: bool = True,
        offload_folder: str = None,
        compile_model: bool = True,
        verbose: bool = False,
    ):
        """

        :param load_8bit: load model in 8-bit, not supported by all models
        :param load_4bit: load model in 4-bit, not supported by all models
        :param load_half: load model in 16-bit
        :param load_gptq: GPTQ model_basename
        :param use_safetensors: use safetensors file
        :param infer_devices: Use torch infer of optimal placement of layers on devices (for non-lora case)
            For non-LORA case, False will spread shards across multiple GPUs, but this can lead to cuda:x cuda:y mismatches
            So it is not the default
        :param base_model: name/path of base model
        :param inference_server: whether base_model is hosted locally ('') or via http (url)
        :param tokenizer_base_model: name/path of tokenizer
        :param lora_weights: name/path
        :param gpu_id: which GPU (0..n_gpus-1) or allow all GPUs if relevant (-1)
        :param reward_type: reward type model for sequence classification
        :param local_files_only: use local files instead of from HF
        :param resume_download: resume downloads from HF
        :param use_auth_token: assumes user did on CLI `huggingface-cli login` to access private repo
        :param trust_remote_code: trust code needed by model
        :param offload_folder: offload folder
        :param compile_model: whether to compile torch model
        :param verbose:
        :return:
        """
        if verbose:
            print("Get %s model" % base_model, flush=True)

        triton_attn = False
        long_sequence = True
        config_kwargs = dict(
            use_auth_token=use_auth_token,
            trust_remote_code=trust_remote_code,
            offload_folder=offload_folder,
            triton_attn=triton_attn,
            long_sequence=long_sequence,
        )
        config, _ = self.get_config(
            base_model, **config_kwargs, raise_exception=False
        )

        if base_model in non_hf_types:
            assert config is None, "Expected config None for %s" % base_model

        llama_type_from_config = "llama" in str(config).lower()
        llama_type_from_name = "llama" in base_model.lower()
        llama_type = llama_type_from_config or llama_type_from_name
        if "xgen" in base_model.lower():
            llama_type = False
        if llama_type:
            if verbose:
                print(
                    "Detected as llama type from"
                    " config (%s) or name (%s)"
                    % (llama_type_from_config, llama_type_from_name),
                    flush=True,
                )

        model_loader, tokenizer_loader = get_loaders(
            model_name=base_model,
            reward_type=reward_type,
            llama_type=llama_type,
            load_gptq=load_gptq,
        )

        tokenizer_kwargs = dict(
            local_files_only=local_files_only,
            resume_download=resume_download,
            use_auth_token=use_auth_token,
            trust_remote_code=trust_remote_code,
            offload_folder=offload_folder,
            padding_side="left",
            config=config,
        )
        if not tokenizer_base_model:
            tokenizer_base_model = base_model

        if (
            config is not None
            and tokenizer_loader is not None
            and not isinstance(tokenizer_loader, str)
        ):
            tokenizer = tokenizer_loader.from_pretrained(
                tokenizer_base_model, **tokenizer_kwargs
            )
            # sets raw (no cushion) limit
            self.set_model_max_len(config, tokenizer, verbose=False)
            # if using fake tokenizer, not really accurate when lots of numbers, give a bit of buffer, else get:
            # Generation Failed: Input validation error: `inputs` must have less than 2048 tokens. Given: 2233
            tokenizer.model_max_length = tokenizer.model_max_length - 50
        else:
            tokenizer = FakeTokenizer()

        if isinstance(inference_server, str) and inference_server.startswith(
            "http"
        ):
            (
                inference_server,
                gr_client,
                hf_client,
            ) = self.get_client_from_inference_server(
                inference_server, base_model=base_model
            )
            client = gr_client or hf_client
            # Don't return None, None for model, tokenizer so triggers
            return client, tokenizer, "http"
        if isinstance(inference_server, str) and inference_server.startswith(
            "openai"
        ):
            assert os.getenv(
                "OPENAI_API_KEY"
            ), "Set environment for OPENAI_API_KEY"
            # Don't return None, None for model, tokenizer so triggers
            # include small token cushion
            tokenizer = FakeTokenizer(
                model_max_length=model_token_mapping[base_model] - 50
            )
            return inference_server, tokenizer, inference_server
        assert not inference_server, (
            "Malformed inference_server=%s" % inference_server
        )
        if base_model in non_hf_types:
            from gpt4all_llm import get_model_tokenizer_gpt4all

            model, tokenizer, _ = get_model_tokenizer_gpt4all(base_model)
            return model, tokenizer, self.device

        # get local torch-HF model
        return self.get_hf_model(
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            load_half=load_half,
            load_gptq=load_gptq,
            use_safetensors=use_safetensors,
            infer_devices=infer_devices,
            device=self.device,
            base_model=base_model,
            tokenizer_base_model=tokenizer_base_model,
            lora_weights=lora_weights,
            gpu_id=gpu_id,
            reward_type=reward_type,
            local_files_only=local_files_only,
            resume_download=resume_download,
            use_auth_token=use_auth_token,
            trust_remote_code=trust_remote_code,
            offload_folder=offload_folder,
            compile_model=compile_model,
            llama_type=llama_type,
            config_kwargs=config_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            verbose=verbose,
        )

    def get_hf_model(
        self,
        load_8bit: bool = False,
        load_4bit: bool = False,
        load_half: bool = True,
        load_gptq: str = "",
        use_safetensors: bool = False,
        infer_devices: bool = True,
        device: str = None,
        base_model: str = "",
        tokenizer_base_model: str = "",
        lora_weights: str = "",
        gpu_id: int = 0,
        reward_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: bool = True,
        offload_folder: str = None,
        compile_model: bool = True,
        llama_type: bool = False,
        config_kwargs=None,
        tokenizer_kwargs=None,
        verbose: bool = False,
    ):
        assert config_kwargs is not None
        assert tokenizer_kwargs is not None

        if lora_weights is not None and lora_weights.strip():
            if verbose:
                print("Get %s lora weights" % lora_weights, flush=True)

        if "gpt2" in base_model.lower():
            # RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Half
            load_8bit = False
            load_4bit = False

        assert (
            base_model.strip()
        ), "Please choose a base model with --base_model (CLI) or load one from Models Tab (gradio)"

        model_loader, tokenizer_loader = get_loaders(
            model_name=base_model,
            reward_type=reward_type,
            llama_type=llama_type,
            load_gptq=load_gptq,
        )

        config, _ = self.get_config(
            base_model,
            return_model=False,
            raise_exception=True,
            **config_kwargs,
        )

        if tokenizer_loader is not None and not isinstance(
            tokenizer_loader, str
        ):
            tokenizer = tokenizer_loader.from_pretrained(
                tokenizer_base_model, **tokenizer_kwargs
            )
        else:
            tokenizer = tokenizer_loader

        if isinstance(tokenizer, str):
            # already a pipeline, tokenizer_loader is string for task
            model = model_loader(
                tokenizer,
                model=base_model,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16
                if self.device == "cuda"
                else torch.float32,
            )
        else:
            assert self.device in ["cuda", "cpu", "mps"], (
                "Unsupported device %s" % self.device
            )
            model_kwargs = dict(
                local_files_only=local_files_only,
                torch_dtype=torch.float16
                if self.device == "cuda"
                else torch.float32,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                trust_remote_code=trust_remote_code,
                offload_folder=offload_folder,
            )
            if (
                "mbart-" not in base_model.lower()
                and "mpt-" not in base_model.lower()
            ):
                if (
                    infer_devices
                    and gpu_id is not None
                    and gpu_id >= 0
                    and self.device == "cuda"
                ):
                    device_map = {"": gpu_id}
                else:
                    device_map = "auto"
                model_kwargs.update(
                    dict(
                        load_in_8bit=load_8bit,
                        load_in_4bit=load_4bit,
                        device_map=device_map,
                    )
                )
            if (
                "mpt-" in base_model.lower()
                and gpu_id is not None
                and gpu_id >= 0
            ):
                # MPT doesn't support spreading over GPUs
                model_kwargs.update(
                    dict(
                        device_map={"": gpu_id}
                        if self.device == "cuda"
                        else "cpu"
                    )
                )

            if "OpenAssistant/reward-model".lower() in base_model.lower():
                # FIXME: could put on other GPUs
                model_kwargs["device_map"] = (
                    {"": 0} if self.device == "cuda" else {"": "cpu"}
                )
                model_kwargs.pop("torch_dtype", None)
            self.pop_unused_model_kwargs(model_kwargs)

            if not lora_weights:
                # torch.device context uses twice memory for AutoGPTQ
                context = NullContext if load_gptq else torch.device
                with context(self.device):
                    if infer_devices:
                        config, model = self.get_config(
                            base_model,
                            return_model=True,
                            raise_exception=True,
                            **config_kwargs,
                        )
                        model = self.get_non_lora_model(
                            base_model,
                            model_loader,
                            load_half,
                            load_gptq,
                            use_safetensors,
                            model_kwargs,
                            reward_type,
                            config,
                            model,
                            gpu_id=gpu_id,
                        )
                    else:
                        config, _ = self.get_config(
                            base_model, **config_kwargs
                        )
                        if load_half and not (
                            load_8bit or load_4bit or load_gptq
                        ):
                            model = model_loader(
                                base_model, config=config, **model_kwargs
                            ).half()
                        else:
                            model = model_loader(
                                base_model, config=config, **model_kwargs
                            )
            elif load_8bit or load_4bit:
                config, _ = self.get_config(base_model, **config_kwargs)
                model = model_loader(base_model, config=config, **model_kwargs)
                from peft import (
                    PeftModel,
                )  # loads cuda, so avoid in global scope

                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16
                    if self.device == "cuda"
                    else torch.float32,
                    local_files_only=local_files_only,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    trust_remote_code=trust_remote_code,
                    offload_folder=offload_folder,
                    device_map={"": 0}
                    if self.device == "cuda"
                    else {"": "cpu"},  # seems to be required
                )
            else:
                with torch.device(self.device):
                    config, _ = self.get_config(
                        base_model, raise_exception=True, **config_kwargs
                    )
                    model = model_loader(
                        base_model, config=config, **model_kwargs
                    )
                    from peft import (
                        PeftModel,
                    )  # loads cuda, so avoid in global scope

                    model = PeftModel.from_pretrained(
                        model,
                        lora_weights,
                        torch_dtype=torch.float16
                        if self.device == "cuda"
                        else torch.float32,
                        local_files_only=local_files_only,
                        resume_download=resume_download,
                        use_auth_token=use_auth_token,
                        trust_remote_code=trust_remote_code,
                        offload_folder=offload_folder,
                        device_map="auto",
                    )
                    if load_half and not load_gptq:
                        model.half()

        # unwind broken decapoda-research config
        if llama_type:
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
        if "gpt2" in base_model.lower():
            # add special tokens that otherwise all share the same id
            tokenizer.add_special_tokens(
                {
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                    "pad_token": "<pad>",
                }
            )

        if not isinstance(tokenizer, str):
            model.eval()
            # if torch.__version__ >= "2" and sys.platform != "win32" and compile_model:
            #     model = torch.compile(model)

        self.set_model_max_len(
            config, tokenizer, verbose=False, reward_type=reward_type
        )

        return model, tokenizer, self.device

    def set_model_max_len(
        self, config, tokenizer, verbose=False, reward_type=False
    ):
        if reward_type:
            # limit deberta, else uses too much memory and not worth response score
            tokenizer.model_max_length = 512
        if hasattr(config, "max_seq_len") and isinstance(
            config.max_seq_len, int
        ):
            tokenizer.model_max_length = config.max_seq_len
        elif hasattr(config, "max_position_embeddings") and isinstance(
            config.max_position_embeddings, int
        ):
            # help automatically limit inputs to generate
            tokenizer.model_max_length = config.max_position_embeddings
        else:
            if verbose:
                print(
                    "Could not determine model_max_length, setting to 2048",
                    flush=True,
                )
            tokenizer.model_max_length = 2048
        # for bug in HF transformers
        if tokenizer.model_max_length > 100000000:
            tokenizer.model_max_length = 2048

    def pop_unused_model_kwargs(self, model_kwargs):
        """
        in-place pop unused kwargs that are not dependency-upgrade friendly
        no point passing in False, is default, and helps avoid needing to update requirements for new deps
        :param model_kwargs:
        :return:
        """
        check_list = ["load_in_8bit", "load_in_4bit"]
        for k in check_list:
            if k in model_kwargs and not model_kwargs[k]:
                model_kwargs.pop(k)

    def get_score_model(
        self,
        score_model: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        load_half: bool = True,
        load_gptq: str = "",
        infer_devices: bool = True,
        base_model: str = "",
        inference_server: str = "",
        tokenizer_base_model: str = "",
        lora_weights: str = "",
        gpu_id: int = 0,
        reward_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: bool = True,
        offload_folder: str = None,
        compile_model: bool = True,
        verbose: bool = False,
    ):
        if score_model is not None and score_model.strip():
            load_8bit = False
            load_4bit = False
            load_half = False
            load_gptq = ""
            use_safetensors = False
            base_model = score_model.strip()
            tokenizer_base_model = ""
            lora_weights = ""
            inference_server = ""
            llama_type = False
            compile_model = False
            smodel, stokenizer, _ = self.get_model(
                reward_type=True,
                **get_kwargs(
                    self.get_model, exclude_names=["reward_type"], **locals()
                ),
            )
        else:
            smodel, stokenizer, _ = None, None, None
        return smodel, stokenizer, self.device

    def evaluate(
        self,
        model_state,
        my_db_state,
        # START NOTE: Examples must have same order of parameters
        instruction,
        iinput,
        context,
        stream_output,
        prompt_type,
        prompt_dict,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        max_time,
        repetition_penalty,
        num_return_sequences,
        do_sample,
        chat,
        instruction_nochat,
        iinput_nochat,
        langchain_mode,
        langchain_action,
        top_k_docs,
        chunk,
        chunk_size,
        document_choice,
        # END NOTE: Examples must have same order of parameters
        src_lang=None,
        tgt_lang=None,
        debug=False,
        concurrency_count=None,
        save_dir=None,
        sanitize_bot_response=False,
        model_state0=None,
        memory_restriction_level=None,
        max_max_new_tokens=None,
        is_public=None,
        max_max_time=None,
        raise_generate_gpu_exceptions=None,
        chat_context=None,
        lora_weights=None,
        load_db_if_exists=True,
        dbs=None,
        user_path=None,
        detect_user_path_changes_every_query=None,
        use_openai_embedding=None,
        use_openai_model=None,
        hf_embedding_model=None,
        db_type=None,
        n_jobs=None,
        first_para=None,
        text_limit=None,
        verbose=False,
        cli=False,
        reverse_docs=True,
        use_cache=None,
        auto_reduce_chunks=None,
        max_chunks=None,
        model_lock=None,
        force_langchain_evaluate=None,
        model_state_none=None,
    ):
        # ensure passed these
        assert concurrency_count is not None
        assert memory_restriction_level is not None
        assert raise_generate_gpu_exceptions is not None
        assert chat_context is not None
        assert use_openai_embedding is not None
        assert use_openai_model is not None
        assert hf_embedding_model is not None
        assert db_type is not None
        assert top_k_docs is not None and isinstance(top_k_docs, int)
        assert chunk is not None and isinstance(chunk, bool)
        assert chunk_size is not None and isinstance(chunk_size, int)
        assert n_jobs is not None
        assert first_para is not None

        if debug:
            locals_dict = locals().copy()
            locals_dict.pop("model_state", None)
            locals_dict.pop("model_state0", None)
            locals_dict.pop("model_states", None)
            print(locals_dict)

        no_model_msg = (
            "Please choose a base model with --base_model (CLI) or load in Models Tab (gradio).\n"
            "Then start New Conversation"
        )

        if model_state is None:
            model_state = model_state_none.copy()
        if model_state0 is None:
            # e.g. for no gradio case, set dummy value, else should be set
            model_state0 = model_state_none.copy()

        # model_state['model] is only 'model' if should use model_state0
        # model could also be None
        have_model_lock = model_lock is not None
        have_fresh_model = model_state["model"] not in [
            None,
            "model",
            no_model_str,
        ]
        # for gradio UI control, expect model_state and model_state0 to match, so if have_model_lock=True, then should have_fresh_model=True
        # but gradio API control will only use nochat api etc. and won't use fresh model, so can't assert in general
        # if have_model_lock:
        #    assert have_fresh_model, "Expected model_state and model_state0 to match if have_model_lock"
        have_cli_model = model_state0["model"] not in [
            None,
            "model",
            no_model_str,
        ]

        if have_fresh_model:
            # USE FRESH MODEL
            if not have_model_lock:
                # model_state0 is just one of model_state if model_lock, so don't nuke
                # try to free-up original model (i.e. list was passed as reference)
                if model_state0["model"] and hasattr(
                    model_state0["model"], "cpu"
                ):
                    model_state0["model"].cpu()
                    model_state0["model"] = None
                # try to free-up original tokenizer (i.e. list was passed as reference)
                if model_state0["tokenizer"]:
                    model_state0["tokenizer"] = None
                clear_torch_cache()
            chosen_model_state = model_state
        elif have_cli_model:
            # USE MODEL SETUP AT CLI
            assert isinstance(
                model_state["model"], str
            )  # expect no fresh model
            chosen_model_state = model_state0
        else:
            raise AssertionError(no_model_msg)
        # get variables
        model = chosen_model_state["model"]
        tokenizer = chosen_model_state["tokenizer"]
        base_model = chosen_model_state["base_model"]
        tokenizer_base_model = chosen_model_state["tokenizer_base_model"]
        lora_weights = chosen_model_state["lora_weights"]
        inference_server = chosen_model_state["inference_server"]
        # prefer use input from API over model state
        prompt_type = prompt_type or chosen_model_state["prompt_type"]
        prompt_dict = prompt_dict or chosen_model_state["prompt_dict"]

        if base_model is None:
            raise AssertionError(no_model_msg)

        assert base_model.strip(), no_model_msg
        assert model, "Model is missing"
        assert tokenizer, "Tokenizer is missing"

        # choose chat or non-chat mode
        print(instruction)
        if not chat:
            instruction = instruction_nochat
            iinput = iinput_nochat
        print(instruction)

        # in some cases, like lean nochat API, don't want to force sending prompt_type, allow default choice
        model_lower = base_model.lower()
        if (
            not prompt_type
            and model_lower in inv_prompt_type_to_model_lower
            and prompt_type != "custom"
        ):
            prompt_type = inv_prompt_type_to_model_lower[model_lower]
            if verbose:
                print(
                    "Auto-selecting prompt_type=%s for %s"
                    % (prompt_type, model_lower),
                    flush=True,
                )
        assert prompt_type is not None, "prompt_type was None"

        # Control generation hyperparameters
        # adjust for bad inputs, e.g. in case also come from API that doesn't get constrained by gradio sliders
        # below is for TGI server, not required for HF transformers
        # limits are chosen similar to gradio_runner.py sliders/numbers
        top_p = min(max(1e-3, top_p), 1.0 - 1e-3)
        top_k = min(max(1, int(top_k)), 100)
        temperature = min(max(0.01, temperature), 2.0)
        # FIXME: https://github.com/h2oai/h2ogpt/issues/106
        num_beams = (
            1 if stream_output else num_beams
        )  # See max_beams in gradio_runner
        max_max_new_tokens = self.get_max_max_new_tokens(
            chosen_model_state,
            memory_restriction_level=memory_restriction_level,
            max_new_tokens=max_new_tokens,
            max_max_new_tokens=max_max_new_tokens,
        )
        model_max_length = 2048  # get_model_max_length(chosen_model_state)
        max_new_tokens = min(max(1, int(max_new_tokens)), max_max_new_tokens)
        min_new_tokens = min(max(0, int(min_new_tokens)), max_new_tokens)
        max_time = min(max(0, max_time), max_max_time)
        repetition_penalty = min(max(0.01, repetition_penalty), 3.0)
        num_return_sequences = (
            1 if chat else min(max(1, int(num_return_sequences)), 10)
        )
        (
            min_top_k_docs,
            max_top_k_docs,
            label_top_k_docs,
        ) = self.get_minmax_top_k_docs(is_public)
        top_k_docs = min(max(min_top_k_docs, int(top_k_docs)), max_top_k_docs)
        chunk_size = min(max(128, int(chunk_size)), 2048)
        if not context:
            # get hidden context if have one
            context = self.get_context(chat_context, prompt_type)

        # restrict instruction, typically what has large input
        from h2oai_pipeline import H2OTextGenerationPipeline

        print(instruction)
        (
            instruction,
            num_prompt_tokens1,
        ) = H2OTextGenerationPipeline.limit_prompt(instruction, tokenizer)
        context, num_prompt_tokens2 = H2OTextGenerationPipeline.limit_prompt(
            context, tokenizer
        )
        iinput, num_prompt_tokens3 = H2OTextGenerationPipeline.limit_prompt(
            iinput, tokenizer
        )
        num_prompt_tokens = (
            (num_prompt_tokens1 or 0)
            + (num_prompt_tokens2 or 0)
            + (num_prompt_tokens3 or 0)
        )

        # get prompt
        prompter = Prompter(
            prompt_type,
            prompt_dict,
            debug=debug,
            chat=chat,
            stream_output=stream_output,
        )
        data_point = dict(
            context=context, instruction=instruction, input=iinput
        )
        prompt = prompter.generate_prompt(data_point)

        # THIRD PLACE where LangChain referenced, but imports only occur if enabled and have db to use
        assert langchain_mode in langchain_modes, (
            "Invalid langchain_mode %s" % langchain_mode
        )
        assert langchain_action in langchain_actions, (
            "Invalid langchain_action %s" % langchain_action
        )
        if (
            langchain_mode in ["MyData"]
            and my_db_state is not None
            and len(my_db_state) > 0
            and my_db_state[0] is not None
        ):
            db1 = my_db_state[0]
        elif dbs is not None and langchain_mode in dbs:
            db1 = dbs[langchain_mode]
        else:
            db1 = None
        do_langchain_path = (
            langchain_mode not in [False, "Disabled", "ChatLLM", "LLM"]
            or base_model in non_hf_types
            or force_langchain_evaluate
        )
        if do_langchain_path:
            outr = ""
            # use smaller cut_distanct for wiki_full since so many matches could be obtained, and often irrelevant unless close
            from gpt_langchain import run_qa_db

            gen_hyper_langchain = dict(
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                early_stopping=early_stopping,
                max_time=max_time,
                num_return_sequences=num_return_sequences,
            )
            outr, extra = run_qa_db(
                query=instruction,
                iinput=iinput,
                context=context,
                model_name=base_model,
                model=model,
                tokenizer=tokenizer,
                inference_server=inference_server,
                stream_output=stream_output,
                prompter=prompter,
                load_db_if_exists=load_db_if_exists,
                db=db1,
                user_path=user_path,
                detect_user_path_changes_every_query=detect_user_path_changes_every_query,
                cut_distanct=1.1
                if langchain_mode in ["wiki_full"]
                else 1.64,  # FIXME, too arbitrary
                use_openai_embedding=use_openai_embedding,
                use_openai_model=use_openai_model,
                hf_embedding_model=hf_embedding_model,
                first_para=first_para,
                text_limit=text_limit,
                chunk=chunk,
                chunk_size=chunk_size,
                langchain_mode=langchain_mode,
                langchain_action=langchain_action,
                document_choice=document_choice,
                db_type=db_type,
                top_k_docs=top_k_docs,
                **gen_hyper_langchain,
                prompt_type=prompt_type,
                prompt_dict=prompt_dict,
                n_jobs=n_jobs,
                verbose=verbose,
                cli=cli,
                sanitize_bot_response=sanitize_bot_response,
                reverse_docs=reverse_docs,
                lora_weights=lora_weights,
                auto_reduce_chunks=auto_reduce_chunks,
                max_chunks=max_chunks,
                device=self.device,
            )
            response = dict(response=outr, sources=extra)
            if outr or base_model in non_hf_types:
                # if got no response (e.g. not showing sources and got no sources,
                # so nothing to give to LLM), then slip through and ask LLM
                # Or if llama/gptj, then just return since they had no response and can't go down below code path
                # clear before return, since .then() never done if from API
                clear_torch_cache()
            return response

    inputs_list_names = list(inspect.signature(evaluate).parameters)
    global inputs_kwargs_list
    inputs_kwargs_list = [
        x
        for x in inputs_list_names
        if x not in eval_func_param_names + ["model_state", "my_db_state"]
    ]

    def get_cutoffs(
        self,
        memory_restriction_level,
        for_context=False,
        model_max_length=2048,
    ):
        # help to avoid errors like:
        # RuntimeError: The size of tensor a (2048) must match the size of tensor b (2049) at non-singleton dimension 3
        # RuntimeError: expected scalar type Half but found Float
        # with - 256
        if memory_restriction_level > 0:
            max_length_tokenize = (
                768 - 256 if memory_restriction_level <= 2 else 512 - 256
            )
        else:
            # at least give room for 1 paragraph output
            max_length_tokenize = model_max_length - 256
        cutoff_len = (
            max_length_tokenize * 4
        )  # if reaches limit, then can't generate new tokens
        output_smallest = 30 * 4
        max_prompt_length = cutoff_len - output_smallest

        if for_context:
            # then lower even more to avoid later chop, since just estimate tokens in context bot
            max_prompt_length = max(64, int(max_prompt_length * 0.8))

        return (
            cutoff_len,
            output_smallest,
            max_length_tokenize,
            max_prompt_length,
        )

    def generate_with_exceptions(
        self,
        func,
        *args,
        prompt="",
        inputs_decoded="",
        raise_generate_gpu_exceptions=True,
        **kwargs,
    ):
        try:
            func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            print(
                "GPU OOM 2: prompt: %s inputs_decoded: %s exception: %s"
                % (prompt, inputs_decoded, str(e)),
                flush=True,
            )
            if "input_ids" in kwargs:
                if kwargs["input_ids"] is not None:
                    kwargs["input_ids"].cpu()
                kwargs["input_ids"] = None
            traceback.print_exc()
            clear_torch_cache()
            return
        except (Exception, RuntimeError) as e:
            if (
                "Expected all tensors to be on the same device" in str(e)
                or "expected scalar type Half but found Float" in str(e)
                or "probability tensor contains either" in str(e)
                or "cublasLt ran into an error!" in str(e)
                or "mat1 and mat2 shapes cannot be multiplied" in str(e)
            ):
                print(
                    "GPU Error: prompt: %s inputs_decoded: %s exception: %s"
                    % (prompt, inputs_decoded, str(e)),
                    flush=True,
                )
                traceback.print_exc()
                clear_torch_cache()
                if raise_generate_gpu_exceptions:
                    raise
                return
            else:
                clear_torch_cache()
                if raise_generate_gpu_exceptions:
                    raise

    def get_generate_params(
        self,
        model_lower,
        chat,
        stream_output,
        show_examples,
        prompt_type,
        prompt_dict,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        max_time,
        repetition_penalty,
        num_return_sequences,
        do_sample,
        top_k_docs,
        chunk,
        chunk_size,
        verbose,
    ):
        use_defaults = False
        use_default_examples = True
        examples = []
        task_info = "LLM"
        if model_lower:
            print(f"Using Model {model_lower}", flush=True)
        else:
            if verbose:
                print("No model defined yet", flush=True)

        min_new_tokens = min_new_tokens if min_new_tokens is not None else 0
        early_stopping = (
            early_stopping if early_stopping is not None else False
        )
        max_time_defaults = 60 * 3
        max_time = max_time if max_time is not None else max_time_defaults

        if (
            not prompt_type
            and model_lower in inv_prompt_type_to_model_lower
            and prompt_type != "custom"
        ):
            prompt_type = inv_prompt_type_to_model_lower[model_lower]
            if verbose:
                print(
                    "Auto-selecting prompt_type=%s for %s"
                    % (prompt_type, model_lower),
                    flush=True,
                )

        # examples at first don't include chat, instruction_nochat, iinput_nochat, added at end
        if show_examples is None:
            if chat:
                show_examples = False
            else:
                show_examples = True

        summarize_example1 = """Jeff: Can I train a ? Transformers model on Amazon SageMaker? 
    Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
    Jeff: ok.
    Jeff: and how can I get started? 
    Jeff: where can I find documentation? 
    Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face"""

        use_placeholder_instruction_as_example = False
        if (
            "bart-large-cnn-samsum" in model_lower
            or "flan-t5-base-samsum" in model_lower
        ):
            placeholder_instruction = summarize_example1
            placeholder_input = ""
            use_defaults = True
            use_default_examples = False
            use_placeholder_instruction_as_example = True
            task_info = "Summarization"
        elif (
            "t5-" in model_lower
            or "t5" == model_lower
            or "flan-" in model_lower
        ):
            placeholder_instruction = "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
            placeholder_input = ""
            use_defaults = True
            use_default_examples = True
            task_info = "Multi-Task: Q/A, translation, Chain-of-Thought, Logical Reasoning, Summarization, etc.  Best to use task prefix as trained on, e.g. `translate English to German: ` (space after colon)"
        elif "mbart-" in model_lower:
            placeholder_instruction = "The girl has long hair."
            placeholder_input = ""
            use_defaults = True
            use_default_examples = False
            use_placeholder_instruction_as_example = True
        elif "gpt2" in model_lower:
            placeholder_instruction = "The sky is"
            placeholder_input = ""
            prompt_type = prompt_type or "plain"
            use_default_examples = (
                True  # some will be odd "continuations" but can be ok
            )
            use_placeholder_instruction_as_example = True
            task_info = "Auto-complete phrase, code, etc."
            use_defaults = True
        else:
            if chat:
                placeholder_instruction = ""
            else:
                placeholder_instruction = "Give detailed answer for whether Einstein or Newton is smarter."
            placeholder_input = ""
            if model_lower in inv_prompt_type_to_model_lower:
                if prompt_type != "custom":
                    prompt_type = inv_prompt_type_to_model_lower[model_lower]
            elif model_lower:
                # default is plain, because might rely upon trust_remote_code to handle prompting
                prompt_type = prompt_type or "plain"
            else:
                prompt_type = ""
            task_info = "No task"
            if prompt_type == "instruct":
                task_info = "Answer question or follow imperative as instruction with optionally input."
            elif prompt_type == "plain":
                task_info = "Auto-complete phrase, code, etc."
            elif prompt_type == "human_bot":
                if chat:
                    task_info = "Chat (Shift-Enter to give question/imperative, input concatenated with instruction)"
                else:
                    task_info = "Ask question/imperative (input concatenated with instruction)"

        # revert to plain if still nothing
        prompt_type = prompt_type or "plain"
        if use_defaults:
            temperature = 1.0 if temperature is None else temperature
            top_p = 1.0 if top_p is None else top_p
            top_k = 40 if top_k is None else top_k
            num_beams = num_beams or 1
            max_new_tokens = max_new_tokens or 128
            repetition_penalty = repetition_penalty or 1.07
            num_return_sequences = min(num_beams, num_return_sequences or 1)
            do_sample = False if do_sample is None else do_sample
        else:
            temperature = 0.1 if temperature is None else temperature
            top_p = 0.75 if top_p is None else top_p
            top_k = 40 if top_k is None else top_k
            num_beams = num_beams or 1
            max_new_tokens = max_new_tokens or 256
            repetition_penalty = repetition_penalty or 1.07
            num_return_sequences = min(num_beams, num_return_sequences or 1)
            do_sample = False if do_sample is None else do_sample
        # doesn't include chat, instruction_nochat, iinput_nochat, added later
        params_list = [
            "",
            stream_output,
            prompt_type,
            prompt_dict,
            temperature,
            top_p,
            top_k,
            num_beams,
            max_new_tokens,
            min_new_tokens,
            early_stopping,
            max_time,
            repetition_penalty,
            num_return_sequences,
            do_sample,
        ]

        if use_placeholder_instruction_as_example:
            examples += [[placeholder_instruction, ""] + params_list]

        if use_default_examples:
            examples += [
                ["Translate English to French", "Good morning"] + params_list,
                [
                    "Give detailed answer for whether Einstein or Newton is smarter.",
                    "",
                ]
                + params_list,
                [
                    "Explain in detailed list, all the best practices for coding in python.",
                    "",
                ]
                + params_list,
                [
                    "Create a markdown table with 3 rows for the primary colors, and 2 columns, with color name and hex codes.",
                    "",
                ]
                + params_list,
                ["Translate to German:  My name is Arthur", ""] + params_list,
                [
                    "Please answer to the following question. Who is going to be the next Ballon d'or?",
                    "",
                ]
                + params_list,
                [
                    "Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.",
                    "",
                ]
                + params_list,
                [
                    "Please answer the following question. What is the boiling point of Nitrogen?",
                    "",
                ]
                + params_list,
                [
                    "Answer the following yes/no question. Can you write a whole Haiku in a single tweet?",
                    "",
                ]
                + params_list,
                [
                    "Simplify the following expression: (False or False and True). Explain your answer.",
                    "",
                ]
                + params_list,
                [
                    "Premise: At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?",
                    "",
                ]
                + params_list,
                [
                    "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?",
                    "",
                ]
                + params_list,
                [
                    "Answer the following question by reasoning step by step.  The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?",
                    "",
                ]
                + params_list,
                [
                    """def area_of_rectangle(a: float, b: float):
        \"\"\"Return the area of the rectangle.\"\"\"""",
                    "",
                ]
                + params_list,
                [
                    """# a function in native python:
    def mean(a):
        return sum(a)/len(a)

    # the same function using numpy:
    import numpy as np
    def mean(a):""",
                    "",
                ]
                + params_list,
                [
                    """X = np.random.randn(100, 100)
    y = np.random.randint(0, 1, 100)

    # fit random forest classifier with 20 estimators""",
                    "",
                ]
                + params_list,
            ]
        # add summary example
        examples += [
            [
                summarize_example1,
                "Summarize"
                if prompt_type not in ["plain", "instruct_simple"]
                else "",
            ]
            + params_list
        ]

        src_lang = "English"
        tgt_lang = "Russian"

        # move to correct position
        for example in examples:
            example += [
                chat,
                "",
                "",
                "Disabled",
                LangChainAction.QUERY.value,
                top_k_docs,
                chunk,
                chunk_size,
                [DocumentChoices.All_Relevant.name],
            ]
            # adjust examples if non-chat mode
            if not chat:
                example[
                    eval_func_param_names.index("instruction_nochat")
                ] = example[eval_func_param_names.index("instruction")]
                example[eval_func_param_names.index("instruction")] = ""

                example[
                    eval_func_param_names.index("iinput_nochat")
                ] = example[eval_func_param_names.index("iinput")]
                example[eval_func_param_names.index("iinput")] = ""
            assert len(example) == len(
                eval_func_param_names
            ), "Wrong example: %s %s" % (
                len(example),
                len(eval_func_param_names),
            )

        if prompt_type == PromptType.custom.name and not prompt_dict:
            raise ValueError(
                "Unexpected to get non-empty prompt_dict=%s for prompt_type=%s"
                % (prompt_dict, prompt_type)
            )

        # get prompt_dict from prompt_type, so user can see in UI etc., or for custom do nothing except check format
        prompt_dict, error0 = get_prompt(
            prompt_type,
            prompt_dict,
            chat=False,
            context="",
            reduced=False,
            making_context=False,
            return_dict=True,
        )
        if error0:
            raise RuntimeError("Prompt wrong: %s" % error0)

        return (
            placeholder_instruction,
            placeholder_input,
            stream_output,
            show_examples,
            prompt_type,
            prompt_dict,
            temperature,
            top_p,
            top_k,
            num_beams,
            max_new_tokens,
            min_new_tokens,
            early_stopping,
            max_time,
            repetition_penalty,
            num_return_sequences,
            do_sample,
            src_lang,
            tgt_lang,
            examples,
            task_info,
        )

    def languages_covered(self):
        # https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered
        covered = """Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)"""
        covered = covered.split(", ")
        covered = {
            x.split(" ")[0]: x.split(" ")[1].replace(")", "").replace("(", "")
            for x in covered
        }
        return covered

    def get_context(self, chat_context, prompt_type):
        if chat_context and prompt_type == "human_bot":
            context0 = """<bot>: I am an intelligent, helpful, truthful, and fair assistant named h2oGPT, who will give accurate, balanced, and reliable responses.  I will not respond with I don't know or I don't understand.
    <human>: I am a human person seeking useful assistance and request all questions be answered completely, and typically expect detailed responses.  Give answers in numbered list format if several distinct but related items are being listed."""
        else:
            context0 = ""
        return context0

    def score_qa(
        self,
        smodel,
        stokenizer,
        max_length_tokenize,
        question,
        answer,
        cutoff_len,
    ):
        question = question[-cutoff_len:]
        answer = answer[-cutoff_len:]

        inputs = stokenizer(
            question,
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=max_length_tokenize,
        ).to(smodel.device)
        try:
            score = (
                torch.sigmoid(smodel(**inputs).logits[0])
                .cpu()
                .detach()
                .numpy()[0]
            )
        except torch.cuda.OutOfMemoryError as e:
            print(
                "GPU OOM 3: question: %s answer: %s exception: %s"
                % (question, answer, str(e)),
                flush=True,
            )
            del inputs
            traceback.print_exc()
            clear_torch_cache()
            return "Response Score: GPU OOM"
        except (Exception, RuntimeError) as e:
            if (
                "Expected all tensors to be on the same device" in str(e)
                or "expected scalar type Half but found Float" in str(e)
                or "probability tensor contains either" in str(e)
                or "cublasLt ran into an error!" in str(e)
                or "device-side assert triggered" in str(e)
            ):
                print(
                    "GPU Error: question: %s answer: %s exception: %s"
                    % (question, answer, str(e)),
                    flush=True,
                )
                traceback.print_exc()
                clear_torch_cache()
                return "Response Score: GPU Error"
            else:
                raise
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        return score

    def check_locals(self, **kwargs):
        # ensure everything in evaluate is here
        can_skip_because_locally_generated = no_default_param_names + [
            # get_model:
            "reward_type"
        ]
        for k in eval_func_param_names:
            if k in can_skip_because_locally_generated:
                continue
            assert k in kwargs, "Missing %s" % k
        for k in inputs_kwargs_list:
            if k in can_skip_because_locally_generated:
                continue
            assert k in kwargs, "Missing %s" % k

        for k in list(inspect.signature(self.get_model).parameters):
            if k in can_skip_because_locally_generated:
                continue
            assert k in kwargs, "Missing %s" % k

    def get_model_max_length(self, model_state):
        if not isinstance(model_state["tokenizer"], (str, types.NoneType)):
            return model_state["tokenizer"].model_max_length
        else:
            return 2048

    def get_max_max_new_tokens(self, model_state, **kwargs):
        if not isinstance(model_state["tokenizer"], (str, types.NoneType)):
            max_max_new_tokens = model_state["tokenizer"].model_max_length
        else:
            max_max_new_tokens = None

        if (
            kwargs["max_max_new_tokens"] is not None
            and max_max_new_tokens is not None
        ):
            return min(max_max_new_tokens, kwargs["max_max_new_tokens"])
        elif kwargs["max_max_new_tokens"] is not None:
            return kwargs["max_max_new_tokens"]
        elif kwargs["memory_restriction_level"] == 1:
            return 768
        elif kwargs["memory_restriction_level"] == 2:
            return 512
        elif kwargs["memory_restriction_level"] >= 3:
            return 256
        else:
            # FIXME: Need to update after new model loaded, so user can control with slider
            return 2048

    def get_minmax_top_k_docs(self, is_public):
        if is_public:
            min_top_k_docs = 1
            max_top_k_docs = 3
            label_top_k_docs = "Number of document chunks"
        else:
            min_top_k_docs = -1
            max_top_k_docs = 100
            label_top_k_docs = (
                "Number of document chunks (-1 = auto fill model context)"
            )
        return min_top_k_docs, max_top_k_docs, label_top_k_docs

    def history_to_context(
        self,
        history,
        langchain_mode1,
        prompt_type1,
        prompt_dict1,
        chat1,
        model_max_length1,
        memory_restriction_level1,
        keep_sources_in_context1,
    ):
        """
        consumes all history up to (but not including) latest history item that is presumed to be an [instruction, None] pair
        :param history:
        :param langchain_mode1:
        :param prompt_type1:
        :param prompt_dict1:
        :param chat1:
        :param model_max_length1:
        :param memory_restriction_level1:
        :param keep_sources_in_context1:
        :return:
        """
        # ensure output will be unique to models
        _, _, _, max_prompt_length = self.get_cutoffs(
            memory_restriction_level1,
            for_context=True,
            model_max_length=model_max_length1,
        )
        context1 = ""
        if max_prompt_length is not None and langchain_mode1 not in ["LLM"]:
            context1 = ""
            # - 1 below because current instruction already in history from user()
            for histi in range(0, len(history) - 1):
                data_point = dict(
                    instruction=history[histi][0],
                    input="",
                    output=history[histi][1],
                )
                (
                    prompt,
                    pre_response,
                    terminate_response,
                    chat_sep,
                    chat_turn_sep,
                ) = generate_prompt(
                    data_point,
                    prompt_type1,
                    prompt_dict1,
                    chat1,
                    reduced=True,
                    making_context=True,
                )
                # md -> back to text, maybe not super important if model trained enough
                if (
                    not keep_sources_in_context1
                    and langchain_mode1 != "Disabled"
                    and prompt.find(source_prefix) >= 0
                ):
                    # FIXME: This is relatively slow even for small amount of text, like 0.3s each history item
                    import re

                    prompt = re.sub(
                        f"{re.escape(source_prefix)}.*?{re.escape(source_postfix)}",
                        "",
                        prompt,
                        flags=re.DOTALL,
                    )
                    if prompt.endswith("\n<p>"):
                        prompt = prompt[:-4]
                prompt = prompt.replace("<br>", chat_turn_sep)
                if not prompt.endswith(chat_turn_sep):
                    prompt += chat_turn_sep
                # most recent first, add older if can
                # only include desired chat history
                if len(prompt + context1) > max_prompt_length:
                    break
                context1 += prompt

            (
                _,
                pre_response,
                terminate_response,
                chat_sep,
                chat_turn_sep,
            ) = generate_prompt(
                {},
                prompt_type1,
                prompt_dict1,
                chat1,
                reduced=True,
                making_context=True,
            )
            if context1 and not context1.endswith(chat_turn_sep):
                context1 += chat_turn_sep  # ensure if terminates abruptly, then human continues on next line
        return context1


class H2OTextIteratorStreamer(TextIteratorStreamer):
    """
    normally, timeout required for now to handle exceptions, else get()
    but with H2O version of TextIteratorStreamer, loop over block to handle
    """

    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        timeout: typing.Optional[float] = None,
        block=True,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.do_stop = False
        self.timeout = timeout
        self.block = block

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                value = (
                    self.stop_signal
                )  # value looks unused in pycharm, not true
                if self.do_stop:
                    print("hit stop", flush=True)
                    # could raise or break, maybe best to raise and make parent see if any exception in thread
                    self.clear_queue()
                    self.do_stop = False
                    raise StopIteration()
                    # break
                value = self.text_queue.get(
                    block=self.block, timeout=self.timeout
                )
                break
            except queue.Empty:
                time.sleep(0.01)
        if value == self.stop_signal:
            self.clear_queue()
            self.do_stop = False
            raise StopIteration()
        else:
            return value

    def clear_queue(self):
        # make sure streamer is reusable after stop hit
        with self.text_queue.mutex:
            self.text_queue.queue.clear()


def entrypoint_main():
    """
    Examples:

    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca_6B
    python generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights='lora-alpaca_6B'
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --lora_weights='lora-alpaca_20B'

    # generate without lora weights, no prompt
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --prompt_type='plain'
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='dai_faq'

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='dai_faq' --lora_weights='lora_20B_daifaq'
    # OpenChatKit settings:
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='human_bot --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0

    python generate.py --base_model='distilgpt2' --prompt_type='plain' --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0 --share=False
    python generate.py --base_model='t5-large' --prompt_type='simple_instruct'
    python generate.py --base_model='philschmid/bart-large-cnn-samsum'
    python generate.py --base_model='philschmid/flan-t5-base-samsum'
    python generate.py --base_model='facebook/mbart-large-50-many-to-many-mmt'

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='human_bot' --lora_weights='GPT-NeoXT-Chat-Base-20B.merged.json.8_epochs.57b2892c53df5b8cefac45f84d019cace803ef26.28'

    must have 4*48GB GPU and run without 8bit in order for sharding to work with infer_devices=False
    can also pass --prompt_type='human_bot' and model can somewhat handle instructions without being instruct tuned
    python generate.py --base_model=decapoda-research/llama-65b-hf --load_8bit=False --infer_devices=False --prompt_type='human_bot'

    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b
    """
    import fire

    langchain = Langchain()

    fire.Fire(langchain.main)


if __name__ == "__main__":
    entrypoint_main()
