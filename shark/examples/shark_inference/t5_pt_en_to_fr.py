from PIL import Image
import requests

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
from iree.compiler import tf as tfc
from iree.compiler import compile_str
from iree import runtime as ireert
import os

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1


class T5Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model.eval()

    def forward(self, input_ids):
        return self.model.generate(input_ids)


if __name__ == "__main__":
    # Prepping Data
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text = "I love the distilled version of models."
    task_prefix = "translate English to German: "
    encoded_input = tokenizer(task_prefix + text, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="pt").input_ids
    inputs = (encoded_input)
    mlir_importer = SharkImporter(
        T5Module(),
        inputs,
        frontend="torch",
    )
    import pdb; pdb.set_trace()
    minilm_mlir, func_name = mlir_importer.import_mlir(
        is_dynamic=True, tracing_required=True
    )
    shark_module = SharkInference(minilm_mlir, func_name, mlir_dialect="linalg")
    shark_module.compile()
    import pdb; pdb.set_trace()
    output = shark_module.forward(inputs)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
