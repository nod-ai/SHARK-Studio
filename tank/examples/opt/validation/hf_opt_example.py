from transformers import AutoTokenizer, OPTForCausalLM

model_name = "facebook/opt-1.3b"
model = OPTForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")
# Generated logits.
logits = model.forward(
    inputs.input_ids, inputs.attention_mask, return_dict=False
)
print(logits[0])
