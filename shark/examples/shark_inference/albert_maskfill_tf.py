from PIL import Image
import requests

from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
from iree.compiler import tf as tfc
from iree.compiler import compile_str
from iree import runtime as ireert
import os
import numpy as np
import sys

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of inputs
t5_inputs = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
]


class AlbertModule(tf.Module):
    def __init__(self):
        super(AlbertModule, self).__init__()
        self.m = TFAutoModelForMaskedLM.from_pretrained("albert-base-v2")
        self.m.predict = lambda x, y: self.m(input_ids=x, attention_mask=y)

    @tf.function(input_signature=t5_inputs, jit_compile=True)
    def forward(self, input_ids, attention_mask):
        return self.m.predict(input_ids, attention_mask)


if __name__ == "__main__":
    # Prepping Data
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    # text = "This is a great [MASK]."
    text = "This [MASK] is very tasty."
    encoded_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf",
    )
    inputs = (encoded_inputs["input_ids"], encoded_inputs["attention_mask"])
    mlir_importer = SharkImporter(
        AlbertModule(),
        inputs,
        frontend="tf",
    )
    minilm_mlir, func_name = mlir_importer.import_mlir(
        is_dynamic=False, tracing_required=False
    )
    shark_module = SharkInference(minilm_mlir, func_name, mlir_dialect="mhlo")
    shark_module.compile()
    output_idx = 0
    data_idx = 1
    token_logits = shark_module.forward(inputs)[output_idx][data_idx]
    mask_id = np.where(
        tf.squeeze(encoded_inputs["input_ids"]) == tokenizer.mask_token_id
    )
    mask_token_logits = token_logits[0, mask_id, :]
    top_5_tokens = np.flip(np.argsort(mask_token_logits)).squeeze()[0:5]
    for token in top_5_tokens:
        print(
            f"'>>> Sample/Warmup output: {text.replace(tokenizer.mask_token, tokenizer.decode(token))}'"
        )
    while True:
        try:
            new_text = input("Give me a sentence with [MASK] to fill: ")
            encoded_inputs = tokenizer(
                new_text,
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="tf",
            )
            inputs = (
                encoded_inputs["input_ids"],
                encoded_inputs["attention_mask"],
            )
            token_logits = shark_module.forward(inputs)[output_idx][data_idx]
            mask_id = np.where(
                tf.squeeze(encoded_inputs["input_ids"])
                == tokenizer.mask_token_id
            )
            mask_token_logits = token_logits[0, mask_id, :]
            top_5_tokens = np.flip(np.argsort(mask_token_logits)).squeeze()[
                0:5
            ]
            for token in top_5_tokens:
                print(
                    f"'>>> {new_text.replace(tokenizer.mask_token, tokenizer.decode(token))}'"
                )
        except KeyboardInterrupt:
            print("Exiting program.")
            sys.exit()
