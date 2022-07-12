from PIL import Image
import requests

from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf
from shark.shark_inference import SharkInference

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

    @tf.function(input_signature=gpt2_inputs)
    def forward(self, input_ids, attention_mask):
        return self.m.predict(input_ids, attention_mask)


if __name__ == "__main__":
    # Prepping Data
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    text = "I love the distilled version of models."

    inputs = tokenizer(text, return_tensors="tf")
    shark_module = SharkInference(
        GPT2Module(), (inputs["input_ids"], inputs["attention_mask"])
    )
    shark_module.set_frontend("tensorflow")
    shark_module.compile()
    print(
        shark_module.forward((inputs["input_ids"], inputs["attention_mask"]))
    )
