from PIL import Image
import requests

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
from iree.compiler import tf as tfc
from iree.compiler import compile_str
from iree import runtime as ireert
import os
import numpy as np

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

if __name__ == "__main__":
    # Prepping Data
    model = AutoModelForMaskedLM.from_pretrained("albert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    text = "This [MASK] is very tasty."
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt")
    token_logits = model(**inputs).logits
    print(token_logits)
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # print(mask_token_logits)
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    print(np.argsort(mask_token_logits.detach().numpy()))
    # print(top_5_tokens)

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
