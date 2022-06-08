from PIL import Image
import requests

from transformers import T5Tokenizer, TFT5Model
import tensorflow as tf
from shark.shark_inference import SharkInference

# Create a set of inputs
t5_inputs = [
    tf.TensorSpec(shape=[1, 10], dtype=tf.int32),
    tf.TensorSpec(shape=[1, 10], dtype=tf.int32),
]

class T5Module(tf.Module):

    def __init__(self):
        super(T5Module, self).__init__()
        self.m = TFT5Model.from_pretrained("t5-small")
        self.m.predict = lambda x,y: self.m(input_ids=x, decoder_input_ids=y)

    @tf.function(input_signature=t5_inputs)
    def forward(self, input_ids, decoder_input_ids):
        return self.m.predict(input_ids, decoder_input_ids)


if __name__ == "__main__":
    # Prepping Data
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    text = "I love the distilled version of models."
    inputs = tokenizer(
        text, return_tensors="tf"
    ).input_ids

    shark_module = SharkInference(
        T5Module(), (inputs, inputs))
    shark_module.set_frontend("tensorflow")
    shark_module.compile()
    print(shark_module.forward((inputs,inputs)))
