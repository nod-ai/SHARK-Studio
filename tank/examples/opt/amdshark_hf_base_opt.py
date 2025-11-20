import os
import torch
from transformers import AutoTokenizer, OPTForCausalLM
from amdshark.amdshark_inference import AMDSharkInference
from amdshark.amdshark_importer import import_with_fx, save_mlir
from amdshark_opt_wrapper import OPTForCausalLMModel

model_name = "facebook/opt-1.3b"
base_model = OPTForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = OPTForCausalLMModel(base_model)

prompt = "What is the meaning of life?"
model_inputs = tokenizer(prompt, return_tensors="pt")
inputs = (
    model_inputs["input_ids"],
    model_inputs["attention_mask"],
)

(
    mlir_module,
    func_name,
) = import_with_fx(
    model=model,
    inputs=inputs,
    is_f16=False,
)
mlir_module = save_mlir(
    mlir_module,
    model_name=model_name.split("/")[1],
    frontend="torch",
    mlir_dialect="linalg",
)
amdshark_module = AMDSharkInference(
    mlir_module,
    device="cpu-sync",
    mlir_dialect="tm_tensor",
)
amdshark_module.compile()
# Generated logits.
logits = amdshark_module("forward", inputs=inputs)
print("AMDSHARK module returns logits:")
print(logits[0])

hf_logits = base_model.forward(inputs[0], inputs[1], return_dict=False)[0]

print("PyTorch baseline returns logits:")
print(hf_logits)
