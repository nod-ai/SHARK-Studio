from turbine_models.custom_models import stateless_llama
from turbine_models.model_runner import vmfbRunner
from turbine_models.gen_external_params.gen_external_params import gen_external_params
import time
from shark.iree_utils.compile_utils import compile_module_to_flatbuffer
from apps.shark_studio.web.utils.file_utils import get_resource_path
from apps.shark_studio.modules.shared_cmd_opts import cmd_opts
import iree.runtime as ireert
from itertools import chain
import gc
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

llm_model_map = {
    "llama2_7b": {
        "initializer": stateless_llama.export_transformer_model,
        "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "compile_flags": ["--iree-opt-const-expr-hoisting=False"],
        "stop_token": 2,
        "max_tokens": 4096,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
    "Trelis/Llama-2-7b-chat-hf-function-calling-v2": {
        "initializer": stateless_llama.export_transformer_model,
        "hf_model_name": "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        "compile_flags": ["--iree-opt-const-expr-hoisting=False"],
        "stop_token": 2,
        "max_tokens": 4096,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
    "TinyPixel/small-llama2": {
        "initializer": stateless_llama.export_transformer_model,
        "hf_model_name": "TinyPixel/small-llama2",
        "compile_flags": ["--iree-opt-const-expr-hoisting=True"],
        "stop_token": 2,
        "max_tokens": 1024,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s>", "</s>"

DEFAULT_CHAT_SYS_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n
"""


def append_user_prompt(history, input_prompt):
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history


class LanguageModel:
    def __init__(
        self,
        model_name,
        hf_auth_token=None,
        device=None,
        quantization="int4",
        precision="",
        external_weights=None,
        use_system_prompt=True,
        streaming_llm=False,
    ):
        self.hf_model_name = llm_model_map[model_name]["hf_model_name"]
        self.device = device.split("=>")[-1].strip()
        self.backend = self.device.split("://")[0]
        self.driver = self.backend
        if "cpu" in device:
            self.device = "cpu"
            self.backend = "llvm-cpu"
            self.driver = "local-task"

        print(f"Selected {self.backend} as IREE target backend.")
        self.precision = "f32" if "cpu" in device else "f16"
        self.quantization = quantization
        self.safe_name = self.hf_model_name.replace("/", "_").replace("-", "_")
        self.external_weight_file = None
        # TODO: find a programmatic solution for model arch spec instead of hardcoding llama2
        self.file_spec = "_".join(
            [
                self.safe_name,
                self.precision,
            ]
        )
        if self.quantization != "None":
            self.file_spec += "_" + self.quantization

        if external_weights in ["safetensors", "gguf"]:
            self.external_weight_file = get_resource_path(
                os.path.join("..", self.file_spec + "." + external_weights)
            )
        else:
            self.external_weights = None
            self.external_weight_file = None

        if streaming_llm:
            # Add streaming suffix to file spec after setting external weights filename.
            self.file_spec += "_streaming"
        self.streaming_llm = streaming_llm

        self.tempfile_name = get_resource_path(
            os.path.join("..", f"{self.file_spec}.tempfile")
        )
        # TODO: Tag vmfb with target triple of device instead of HAL backend
        self.vmfb_name = str(
            get_resource_path(
                os.path.join("..", f"{self.file_spec}_{self.backend}.vmfb.tempfile")
            )
        )

        self.max_tokens = llm_model_map[model_name]["max_tokens"]
        self.iree_module_dict = None
        self.use_system_prompt = use_system_prompt
        self.global_iter = 0
        self.prev_token_len = 0
        self.first_input = True
        self.hf_auth_token = hf_auth_token
        if self.external_weight_file is not None:
            if not os.path.exists(self.external_weight_file):
                print(
                    f"External weight file {self.external_weight_file} does not exist. Generating..."
                )
                gen_external_params(
                    hf_model_name=self.hf_model_name,
                    quantization=self.quantization,
                    weight_path=self.external_weight_file,
                    hf_auth_token=hf_auth_token,
                    precision=self.precision,
                )
            else:
                print(
                    f"External weight file {self.external_weight_file} found for {self.vmfb_name}"
                )
            self.external_weight_file = str(self.external_weight_file)

        if os.path.exists(self.vmfb_name) and (
            external_weights is None or os.path.exists(str(self.external_weight_file))
        ):
            self.runner = vmfbRunner(
                device=self.driver,
                vmfb_path=self.vmfb_name,
                external_weight_path=self.external_weight_file,
            )
            if self.streaming_llm:
                self.model = self.runner.ctx.modules.streaming_state_update
            else:
                self.model = self.runner.ctx.modules.state_update
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                use_fast=False,
                use_auth_token=hf_auth_token,
            )
        elif not os.path.exists(self.tempfile_name):
            self.torch_ir, self.tokenizer = llm_model_map[model_name]["initializer"](
                self.hf_model_name,
                hf_auth_token,
                compile_to="torch",
                external_weights=external_weights,
                precision=self.precision,
                quantization=self.quantization,
                streaming_llm=self.streaming_llm,
            )
            with open(self.tempfile_name, "w+") as f:
                f.write(self.torch_ir)
            del self.torch_ir
            gc.collect()
            self.compile()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                use_fast=False,
                use_auth_token=hf_auth_token,
            )
            self.compile()
        # Reserved for running HF torch model as reference.
        self.hf_mod = None

    def compile(self) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        # ONLY architecture/api-specific compile-time flags for each backend, if needed.
        # hf_model_id-specific global flags currently in model map.
        flags = []
        if "cpu" in self.backend:
            flags.extend(
                [
                    "--iree-global-opt-enable-quantized-matmul-reassociation",
                ]
            )
        elif self.backend == "vulkan":
            flags.extend(["--iree-stream-resource-max-allocation-size=4294967296"])
        flags.extend(llm_model_map[self.hf_model_name]["compile_flags"])
        flatbuffer_blob = compile_module_to_flatbuffer(
            self.tempfile_name,
            device=self.device,
            frontend="torch",
            model_config_path=None,
            extra_args=flags,
            write_to=self.vmfb_name,
        )
        self.runner = vmfbRunner(
            device=self.driver,
            vmfb_path=self.vmfb_name,
            external_weight_path=self.external_weight_file,
        )
        if self.streaming_llm:
            self.model = self.runner.ctx.modules.streaming_state_update
        else:
            self.model = self.runner.ctx.modules.state_update

    def sanitize_prompt(self, prompt):
        if isinstance(prompt, list):
            prompt = list(chain.from_iterable(prompt))
            prompt = " ".join([x for x in prompt if isinstance(x, str)])
        prompt = prompt.replace("\n", " ")
        prompt = prompt.replace("\t", " ")
        prompt = prompt.replace("\r", " ")
        if self.use_system_prompt and self.global_iter == 0:
            prompt = append_user_prompt(DEFAULT_CHAT_SYS_PROMPT, prompt)
            return prompt
        else:
            return f"{B_INST} {prompt} {E_INST}"

    def chat(self, prompt):
        prompt = self.sanitize_prompt(prompt)

        input_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids

        def format_out(results):
            return torch.tensor(results.to_host()[0][0])

        history = []
        for iter in range(self.max_tokens):
            if self.streaming_llm:
                token_slice = max(self.prev_token_len - 1, 0)
                input_tensor = input_tensor[:, token_slice:]
            if self.streaming_llm and self.model["get_seq_step"]() > 600:
                print("Evicting cache space!")
                self.model["evict_kvcache_space"]()
            token_len = input_tensor.shape[-1]
            device_inputs = [
                ireert.asdevicearray(self.runner.config.device, input_tensor)
            ]
            if self.first_input or not self.streaming_llm:
                st_time = time.time()
                token = self.model["run_initialize"](*device_inputs)
                total_time = time.time() - st_time
                token_len += 1
                self.first_input = False
            else:
                st_time = time.time()
                token = self.model["run_cached_initialize"](*device_inputs)
                total_time = time.time() - st_time
                token_len += 1

            history.append(format_out(token))
            while (
                format_out(token) != llm_model_map["llama2_7b"]["stop_token"]
                and len(history) < self.max_tokens
            ):
                dec_time = time.time()
                if self.streaming_llm and self.model["get_seq_step"]() > 600:
                    print("Evicting cache space!")
                    self.model["evict_kvcache_space"]()
                token = self.model["run_forward"](token)
                history.append(format_out(token))
                total_time = time.time() - dec_time
                yield self.tokenizer.decode(history), total_time

            self.prev_token_len = token_len + len(history)

            if format_out(token) == llm_model_map["llama2_7b"]["stop_token"]:
                break

        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
        self.global_iter += 1
        return result_output, total_time

    # Reference HF model function for sanity checks.
    def chat_hf(self, prompt):
        if self.hf_mod is None:
            self.hf_mod = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float,
                token=self.hf_auth_token,
            )
        prompt = self.sanitize_prompt(prompt)

        input_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids
        history = []
        for iter in range(self.max_tokens):
            token_len = input_tensor.shape[-1]
            if self.first_input:
                st_time = time.time()
                result = self.hf_mod(input_tensor)
                token = torch.argmax(result.logits[:, -1, :], dim=1)
                total_time = time.time() - st_time
                token_len += 1
                pkv = result.past_key_values
                self.first_input = False

            history.append(int(token))
            while token != llm_model_map["llama2_7b"]["stop_token"]:
                dec_time = time.time()
                result = self.hf_mod(token.reshape([1, 1]), past_key_values=pkv)
                history.append(int(token))
                total_time = time.time() - dec_time
                token = torch.argmax(result.logits[:, -1, :], dim=1)
                pkv = result.past_key_values
                yield self.tokenizer.decode(history), total_time

            self.prev_token_len = token_len + len(history)

            if token == llm_model_map["llama2_7b"]["stop_token"]:
                break
        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
        self.global_iter += 1
        return result_output, total_time


def llm_chat_api(InputData: dict):
    from datetime import datetime as dt

    import apps.shark_studio.web.utils.globals as global_obj

    print(f"Input keys : {InputData.keys()}")

    # print(f"model : {InputData['model']}")

    is_chat_completion_api = (
        "messages" in InputData.keys()
    )  # else it is the legacy `completion` api

    # For Debugging input data from API
    if is_chat_completion_api:
        print(f"message -> role : {InputData['messages'][0]['role']}")
        print(f"message -> content : {InputData['messages'][0]['content']}")
    else:
        print(f"prompt : {InputData['prompt']}")

    model_name = InputData["model"] if "model" in InputData.keys() else "llama2_7b"
    model_path = llm_model_map[model_name]
    device = InputData["device"] if "device" in InputData.keys() else "cpu"
    precision = "fp16"
    max_tokens = InputData["max_tokens"] if "max_tokens" in InputData.keys() else 4096

    device_id = None
    if not global_obj.get_llm_obj():
        print("\n[LOG] Initializing new pipeline...")
        global_obj.clear_cache()
        gc.collect()
        if "cuda" in device:
            device = "cuda"
        elif "vulkan" in device:
            device_id = int(device.split("://")[1])
            device = "vulkan"
        elif "cpu" in device:
            device = "cpu"
            precision = "fp32"
        else:
            print("unrecognized device")
        llm_model = LanguageModel(
            model_name=model_name,
            hf_auth_token=cmd_opts.hf_auth_token,
            device=device,
            quantization=cmd_opts.quantization,
            external_weights="safetensors",
            use_system_prompt=True,
            streaming_llm=False,
        )
        global_obj.set_llm_obj(llm_model)
    else:
        llm_model = global_obj.get_llm_obj()

    llm_model.max_tokens = max_tokens
    # TODO: add role dict for different models
    if is_chat_completion_api:
        # TODO: add funtionality for multiple messages
        prompt = append_user_prompt(
            InputData["messages"][0]["role"], InputData["messages"][0]["content"]
        )
    else:
        prompt = InputData["prompt"]
    print("prompt = ", prompt)

    for res_op, _ in llm_model.chat(prompt):

        if is_chat_completion_api:
            choices = [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": res_op,  # since we are yeilding the result
                    },
                    "finish_reason": "stop",  # or length
                }
            ]
        else:
            choices = [
                {
                    "text": res_op,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",  # or length
                }
            ]
    end_time = dt.now().strftime("%Y%m%d%H%M%S%f")
    return {
        "id": end_time,
        "object": "chat.completion" if is_chat_completion_api else "text_completion",
        "created": int(end_time),
        "choices": choices,
    }


if __name__ == "__main__":
    lm = LanguageModel(
        "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        hf_auth_token=None,
        device="cpu-task",
        external_weights="safetensors",
    )

    print("model loaded")
    for i in lm.chat("hi, what are you?"):
        print(i)
