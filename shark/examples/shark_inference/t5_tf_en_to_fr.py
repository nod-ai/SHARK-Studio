from PIL import Image
import requests

from transformers import T5Tokenizer, TFT5Model, TFT5ForConditionalGeneration
import tensorflow as tf
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
from iree.compiler import tf as tfc
from iree.compiler import compile_str
from iree import runtime as ireert
import os

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of inputs
t5_inputs = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32)
]

class T5Module(tf.Module):
    def __init__(self):
        super(T5Module, self).__init__()
        self.m = TFT5ForConditionalGeneration.from_pretrained("t5-small")
        self.m.predict = lambda x: self.m.generate(input_ids=x)

    @tf.function(input_signature=t5_inputs)
    def forward(self, input_ids):
        return self.m.predict(input_ids)


if __name__ == "__main__":
    # Prepping Data
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text = "I love the distilled version of models."
    task_prefix = "translate English to German: "
    encoded_input = tokenizer(task_prefix + text, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_tensors="tf").input_ids
    inputs = (encoded_input)
    mlir_importer = SharkImporter(
        T5Module(),
        inputs,
        frontend="tf",
    )
    minilm_mlir, func_name = mlir_importer.import_mlir(
        is_dynamic=False, tracing_required=False
    )
    shark_module = SharkInference(minilm_mlir, func_name, mlir_dialect="mhlo")
    shark_module.compile()
    import pdb; pdb.set_trace()
    output = shark_module.forward(inputs)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
