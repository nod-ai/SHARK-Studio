from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
from iree.compiler import compile_str
from iree import runtime as ireert
import os
import numpy as np

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1


class AlbertModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained("albert-base-v2")
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

if __name__ == "__main__":
    # Prepping Data
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    text = "This [MASK] is very tasty."
    encoded_inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
    inputs = (encoded_inputs["input_ids"],encoded_inputs["attention_mask"])
    mlir_importer = SharkImporter(
        AlbertModule(),
        inputs,
        frontend="torch",
    )
    minilm_mlir, func_name = mlir_importer.import_mlir(
        is_dynamic=False, tracing_required=True
    )
    shark_module = SharkInference(minilm_mlir, func_name, mlir_dialect="linalg")
    shark_module.compile()
    token_logits = torch.tensor(shark_module.forward(inputs))
    mask_id = torch.where(encoded_inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_id, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    for token in top_5_tokens:
        print(f"'>>> Sample/Warmup output: {text.replace(tokenizer.mask_token, tokenizer.decode(token))}'")    
    while True:
        try:
            new_text = input("Give me a sentence with [MASK] to fill: ")
            encoded_inputs = tokenizer(new_text, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
            inputs = (encoded_inputs["input_ids"],encoded_inputs["attention_mask"])
            token_logits = torch.tensor(shark_module.forward(inputs))
            mask_id = torch.where(encoded_inputs["input_ids"] == tokenizer.mask_token_id)[1]
            mask_token_logits = token_logits[0, mask_id, :]
            top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
            for token in top_5_tokens:
                print(f"'>>> {new_text.replace(tokenizer.mask_token, tokenizer.decode(token))}'")    
        except KeyboardInterrupt:
            print("Exiting program.")
            break

