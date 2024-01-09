from turbine_models.custom_models import stateless_llama
from turbine_models.gen_external_params.gen_external_params import gen_external_params
import time
from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    load_vmfb_using_mmap,
)
from apps.shark_studio.web.utils import get_resource_path
import iree.runtime as ireert
from itertools import chain
import gc
import os
import torch
from transformers import AutoTokenizer

llm_model_map = {
    "llama2_7b": {
        "initializer": stateless_llama.export_transformer_model,
        "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "stop_token": 2,
        "max_tokens": 4096,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
    "Trelis/Llama-2-7b-chat-hf-function-calling-v2": {
        "initializer": stateless_llama.export_transformer_model,
        "hf_model_name": "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        "stop_token": 2,
        "max_tokens": 4096,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s>", "</s>"

def append_user_prompt(history, input_prompt):
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history

def append_bot_prompt(history, input_prompt):
    user_prompt = f"{B_SYS} {input_prompt}{E_SYS} {E_SYS}"
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
        print(llm_model_map[model_name])
        self.hf_model_name = llm_model_map[model_name]["hf_model_name"]
        self.device = device.split("=>")[-1].strip()
        self.driver = self.device.split("://")[0]
        print(f"Selected {self.driver} as device driver")
        self.precision = "f32" if "cpu" in self.driver else "f16"
        self.quantization = quantization
        #TODO: find a programmatic solution for model arch spec instead of hardcoding llama2
        self.file_spec = "_".join([
            "llama2",
            "streaming" if streaming_llm else "chat",
            self.precision,
            self.quantization,
        ])
        self.tempfile_name = get_resource_path(f"{self.file_spec}.tempfile")
        #TODO: Tag vmfb with target triple of device instead of HAL backend
        self.vmfb_name = get_resource_path(f"{self.file_spec}_{self.driver}.vmfb.tempfile")    
        self.safe_name = self.hf_model_name.split("/")[-1].replace("-", "_")
        self.max_tokens = llm_model_map[model_name]["max_tokens"]
        self.iree_module_dict = None
        self.external_weight_file = None
        self.streaming_llm = streaming_llm
        if external_weights is not None:
            self.external_weight_file = get_resource_path(
                self.safe_name
                + "_" + self.precision
                + "_" + self.quantization
                + "." + external_weights
            )
        self.use_system_prompt = use_system_prompt
        self.global_iter = 0
        self.prev_token_len = 0
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
        if os.path.exists(self.vmfb_name) and (
            external_weights is None or os.path.exists(str(self.external_weight_file))
        ):
            self.iree_module_dict = dict()
            (
                self.iree_module_dict["vmfb"],
                self.iree_module_dict["config"],
                self.iree_module_dict["temp_file_to_unlink"],
            ) = load_vmfb_using_mmap(
                self.vmfb_name,
                self.driver,
                device_idx=0,
                rt_flags=[],
                external_weight_file=self.external_weight_file,
            )
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
                external_weight_file=self.external_weight_file,
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

    def compile(self) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        flags = [
            "--iree-input-type=torch",
            "--mlir-print-debuginfo",
            "--mlir-print-op-on-diagnostic=false",
            "--iree-llvmcpu-target-cpu-features=host",
            "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
            "--iree-opt-const-expr-hoisting=False",
        ]
        if "cpu" in self.driver:
            flags.extend(
                [
                "--iree-global-opt-enable-quantized-matmul-reassociation",
                "--iree-llvmcpu-enable-ukernels=all"
                ]
            )
        elif self.driver == "vulkan":
            flags.extend(
                [
                    "--iree-stream-resource-max-allocation-size=4294967296"
                ]
            )
        self.iree_module_dict = get_iree_compiled_module(
            self.tempfile_name,
            device=self.device,
            mmap=True,
            frontend="torch",
            external_weight_file=self.external_weight_file,
            write_to=self.vmfb_name,
            extra_args=flags,
        )
        # TODO: delete the temp file

    def sanitize_prompt(self, prompt):
        print(prompt)
        if isinstance(prompt, list):
            prompt = list(chain.from_iterable(prompt))
            prompt = " ".join([x for x in prompt if isinstance(x, str)])
        prompt = prompt.replace("\n", " ")
        prompt = prompt.replace("\t", " ")
        prompt = prompt.replace("\r", " ")
        if self.use_system_prompt and self.global_iter == 0:
            prompt = llm_model_map["llama2_7b"]["system_prompt"] + prompt
        prompt += " [/INST]"
        print(prompt)
        return prompt

    def chat(self, prompt):
        prompt = self.sanitize_prompt(prompt)

        input_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids

        if self.streaming_llm:
            token_slice = max(self.prev_token_len - 1, 0)
            input_tensor = input_tensor[:, token_slice:]

        def format_out(results):
            return torch.tensor(results.to_host()[0][0])

        history = []
        for iter in range(self.max_tokens):
            if self.streaming_llm and self.iree_module_dict["vmfb"]["get_seq_step"]() > 600:
                print("Evicting cache space!")
                self.iree_module_dict["vmfb"]["evict_kvcache_space"]()
            st_time = time.time()
            token_len = input_tensor.shape[-1]
            if iter == 0 and not self.streaming_llm:
                device_inputs = [
                    ireert.asdevicearray(
                        self.iree_module_dict["config"].device, input_tensor
                    )
                ]
                token = self.iree_module_dict["vmfb"]["run_initialize"](*device_inputs)
                token_len += 1
            elif iter == 0:
                device_inputs = [
                    ireert.asdevicearray(
                        self.iree_module_dict["config"].device, input_tensor
                    )
                ]
                token = self.iree_module_dict["vmfb"]["run_cached_initialize"](*device_inputs)
                token_len += 1
            else:
                if self.streaming_llm and self.iree_module_dict["vmfb"]["get_seq_step"]() > 600:
                    print("Evicting cache space!")
                    self.iree_module_dict["vmfb"]["evict_kvcache_space"]()
                device_inputs = [
                    ireert.asdevicearray(
                        self.iree_module_dict["config"].device,
                        token,
                    )
                ]
                token = self.iree_module_dict["vmfb"]["run_forward"](*device_inputs)

            total_time = time.time() - st_time
            history.append(format_out(token))
            self.prev_token_len = token_len + len(history)
            res = self.tokenizer.decode(history, skip_special_tokens=True)
            #prompt = append_bot_prompt(prompt, res)
            yield res, total_time

            if format_out(token) == llm_model_map["llama2_7b"]["stop_token"]:
                break

        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history, skip_special_tokens=True)
        self.global_iter += 1
        return result_output, total_time


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
