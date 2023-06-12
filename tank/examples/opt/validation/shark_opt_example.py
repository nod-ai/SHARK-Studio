from shark.shark_inference import SharkInference
from transformers import AutoTokenizer
import os


model_name = "facebook/opt-1.3b"
vmfb_path = "../opt-1.3b_causallm_30_torch_cpu-sync.vmfb"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
shark_module = SharkInference(mlir_module=None, device="cpu-sync")
shark_module.load_module(vmfb_path)

prompt = "What is the meaning of life?"
model_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=30,
    truncation=True,
    return_tensors="pt",
)
inputs = (
    model_inputs["input_ids"],
    model_inputs["attention_mask"],
)
# Generate logits output of OPT model.
logits = shark_module("forward", inputs)
# Print output logits to validate vs. pytorch + base transformers
print(logits[0])
