import re
import json
import numpy as np

import torch_mlir
from iree.compiler import compile_str
from shark.shark_importer import import_with_fx, get_f16_inputs


class GenerateConfigFile:
    def __init__(
        self,
        model,
        num_sharding_stages: int,
        sharding_stages_id: list[str],
        units_in_each_stage: list[int],
        model_input=None,
        config_file_path="model_config.json",
    ):
        self.model = model
        self.num_sharding_stages = num_sharding_stages
        self.sharding_stages_id = sharding_stages_id
        assert self.num_sharding_stages == len(
            self.sharding_stages_id
        ), "Number of sharding stages should be equal to the list of their ID"
        self.model_input = model_input
        self.config_file_path = config_file_path
        # (Nithin) this is a quick fix - revisit and rewrite
        self.units_in_each_stage = np.array(units_in_each_stage)
        self.track_loop = np.zeros(len(self.sharding_stages_id)).astype(int)

    def split_into_dispatches(
        self,
        backend,
        fx_tracing_required=False,
        f16_model=False,
        torch_mlir_tracing=True,
    ):
        graph_for_compilation = self.model
        if fx_tracing_required:
            graph_for_compilation = import_with_fx(
                self.model,
                self.model_input,
                is_f16=f16_model,
                f16_input_mask=[False, False],
                mlir_type="torchscript",
            )

        module = torch_mlir.compile(
            graph_for_compilation,
            (self.model_input),
            torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=torch_mlir_tracing,
            verbose=False,
        )
        module = module.operation.get_asm(large_elements_limit=4)
        compiled_module_str = str(
            compile_str(
                str(module),
                target_backends=[backend],
                extra_args=[
                    "--compile-to=flow",
                    "--mlir-elide-elementsattrs-if-larger=4",
                ],
            )
        )

        substring_start_idx = [
            m.start()
            for m in re.finditer("flow.dispatch @", compiled_module_str)
        ]
        dispatch_list = dict()

        # dispatch_no is the 'i'th index of a dispatch out of n total dispatches of a model
        # dispatch_id is the unique id of a dispatch, multiple instances of the same dispatch
        # can occur in a model
        for dispatch_no, substring_idx in enumerate(substring_start_idx):
            dispatch_idx = (
                compiled_module_str[substring_idx:]
                .split(":")[0]
                .split("@")[-1]
            )
            key = "dispatch_no_" + str(dispatch_no)
            dispatch_list[key] = {n: "None" for n in self.sharding_stages_id}
            dispatch_list[key]["dispatch_id"] = dispatch_idx

        self.generate_json(dispatch_list)

    def split_into_layers(self):
        model_dictionary = dict()

        for name, m in self.model.named_modules():
            if name == "":
                continue

            # Remove non-leaf nodes from the config as they aren't an operation
            substring_before_final_period = name.split(".")[:-1]
            substring_before_final_period = ".".join(
                substring_before_final_period
            )
            if substring_before_final_period in model_dictionary:
                del model_dictionary[substring_before_final_period]

            # layer_dict = {n: "None" for n in self.sharding_stages_id}

            # By default embed increasing device id's for each layer
            increasing_wraparound_idx_list = (
                self.track_loop % self.units_in_each_stage
            )
            layer_dict = {
                n: int(increasing_wraparound_idx_list[idx][0][0])
                for idx, n in enumerate(self.sharding_stages_id)
            }
            self.track_loop += 1
            model_dictionary[name] = layer_dict

        self.generate_json(model_dictionary)

    def generate_json(self, artifacts):
        with open(self.config_file_path, "w") as outfile:
            json.dump(artifacts, outfile)


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer

    hf_model_path = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, use_fast=False)
    compilation_prompt = "".join(["0" for _ in range(17)])
    compilation_input_ids = tokenizer(
        compilation_prompt,
        return_tensors="pt",
    ).input_ids
    compilation_input_ids = torch.tensor(compilation_input_ids).reshape(
        [1, 19]
    )
    firstVicunaCompileInput = (compilation_input_ids,)
    from apps.language_models.src.model_wrappers.vicuna_model import (
        FirstVicuna,
        SecondVicuna,
        CombinedModel,
    )

    model = CombinedModel()
    c = GenerateConfigFile(model, 1, ["gpu_id"], firstVicunaCompileInput)
    c.split_into_layers()
