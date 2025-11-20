from PIL import Image
import requests

from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf
from amdshark.amdshark_inference import AMDSharkInference

# Create a set of inputs
gpt2_inputs = [
    tf.TensorSpec(shape=[1, 8], dtype=tf.int32),
    tf.TensorSpec(shape=[1, 8], dtype=tf.int32),
]


class GPT2Module(tf.Module):
    def __init__(self):
        super(GPT2Module, self).__init__()
        self.m = TFGPT2Model.from_pretrained("distilgpt2")

        self.m.predict = lambda x, y: self.m(input_ids=x, attention_mask=y)

    @tf.function(input_signature=gpt2_inputs, jit_compile=True)
    def forward(self, input_ids, attention_mask):
        return self.m.predict(input_ids, attention_mask)


if __name__ == "__main__":
    # Prepping Data
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    text = "I love the distilled version of models."

    inputs = tokenizer(text, return_tensors="tf")
    amdshark_module = AMDSharkInference(
        GPT2Module(), (inputs["input_ids"], inputs["attention_mask"])
    )
    amdshark_module.set_frontend("tensorflow")
    amdshark_module.compile()
    print(
        amdshark_module.forward((inputs["input_ids"], inputs["attention_mask"]))
    )
