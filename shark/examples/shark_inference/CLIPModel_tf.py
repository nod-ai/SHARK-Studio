from PIL import Image
import requests

from transformers import CLIPProcessor, TFCLIPModel
import tensorflow as tf
from shark.shark_inference import SharkInference

# Create a set of inputs
clip_vit_inputs = [
    tf.TensorSpec(shape=[2, 7], dtype=tf.int32),
    tf.TensorSpec(shape=[2, 7], dtype=tf.int32),
    tf.TensorSpec(shape=[1, 3, 224, 224], dtype=tf.float32),
]


class CLIPModule(tf.Module):
    def __init__(self):
        super(CLIPModule, self).__init__()
        self.m = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.m.predict = lambda x, y, z: self.m(
            input_ids=x, attention_mask=y, pixel_values=z
        )

    @tf.function(input_signature=clip_vit_inputs)
    def forward(self, input_ids, attention_mask, pixel_values):
        return self.m.predict(
            input_ids, attention_mask, pixel_values
        ).logits_per_image


if __name__ == "__main__":
    # Prepping Data
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"],
        images=image,
        return_tensors="tf",
        padding=True,
    )

    shark_module = SharkInference(
        CLIPModule(),
        (inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]),
    )
    shark_module.set_frontend("tensorflow")
    shark_module.compile()

    print(
        shark_module.forward(
            (
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["pixel_values"],
            )
        )
    )
