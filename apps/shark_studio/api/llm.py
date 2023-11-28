from turbine_models.custom_models import stateless_llama
import time
from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    load_vmfb_using_mmap,
)
from apps.shark_studio.api.utils import get_resource_path
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


def safe_name(name):
    return name.replace("/", "_").replace("-", "_")


class LanguageModel:
    def __init__(
        self,
        model_name,
        hf_auth_token=None,
        device=None,
        precision="fp32",
        external_weights=None,
        use_system_prompt=True,
    ):
        print(llm_model_map[model_name])
        self.hf_model_name = llm_model_map[model_name]["hf_model_name"]
        self.tempfile_name = get_resource_path(
            f"{safe_name(self.hf_model_name)}.mlir.tempfile"
        )
        self.vmfb_name = get_resource_path(
            f"{safe_name(self.hf_model_name)}.vmfb.tempfile"
        )
        self.device = device
        self.precision = precision
        self.max_tokens = llm_model_map[model_name]["max_tokens"]
        self.iree_module_dict = None
        self.external_weight_file = None
        if external_weights is not None:
            self.external_weight_file = (
                f"{safe_name(self.hf_model_name)}.{external_weights}"
            )
        self.use_system_prompt = use_system_prompt
        self.global_iter = 0
        if os.path.exists(self.vmfb_name) and (
            os.path.exists(self.external_weight_file)
            or external_weights is None
        ):
            self.iree_module_dict = dict()
            (
                self.iree_module_dict["vmfb"],
                self.iree_module_dict["config"],
                self.iree_module_dict["temp_file_to_unlink"],
            ) = load_vmfb_using_mmap(
                self.vmfb_name,
                device,
                device_idx=0,
                rt_flags=[],
                external_weight_file=self.external_weight_file,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                use_fast=False,
                token=hf_auth_token,
            )
        elif not os.path.exists(self.tempfile_name):
            self.torch_ir, self.tokenizer = llm_model_map[model_name][
                "initializer"
            ](
                self.hf_model_name,
                hf_auth_token,
                compile_to="torch",
                external_weights=external_weights,
                external_weight_file=self.external_weight_file,
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
                token=hf_auth_token,
            )
            self.compile()

    def compile(self) -> None:
        # this comes with keys: "vmfb", "config", and "temp_file_to_unlink".
        self.iree_module_dict = get_iree_compiled_module(
            self.tempfile_name,
            device=self.device,
            mmap=True,
            frontend="torch",
            external_weight_file=self.external_weight_file,
            write_to=self.vmfb_name,
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

        def format_out(results):
            return torch.tensor(results.to_host()[0][0])

        history = []
        for iter in range(self.max_tokens):
            st_time = time.time()
            if iter == 0:
                device_inputs = [
                    ireert.asdevicearray(
                        self.iree_module_dict["config"].device, input_tensor
                    )
                ]
                token = self.iree_module_dict["vmfb"]["run_initialize"](
                    *device_inputs
                )
            else:
                device_inputs = [
                    ireert.asdevicearray(
                        self.iree_module_dict["config"].device,
                        token,
                    )
                ]
                token = self.iree_module_dict["vmfb"]["run_forward"](
                    *device_inputs
                )

            total_time = time.time() - st_time
            history.append(format_out(token))
            yield self.tokenizer.decode(history), total_time

            if format_out(token) == llm_model_map["llama2_7b"]["stop_token"]:
                break

        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
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
