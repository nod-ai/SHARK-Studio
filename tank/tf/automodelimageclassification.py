from transformers import TFAutoModelForImageClassification
from transformers import ConvNextFeatureExtractor, ViTFeatureExtractor
from transformers import BeitFeatureExtractor, AutoFeatureExtractor
import tensorflow as tf
from PIL import Image
import requests
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_tf_model

# Create a set of input signature.
inputs_signature = [
    tf.TensorSpec(shape=[1, 3, 224, 224], dtype=tf.float32),
]


class AutoModelImageClassfication(tf.Module):
    def __init__(self, model_name):
        super(AutoModelImageClassfication, self).__init__()
        self.m = TFAutoModelForImageClassification.from_pretrained(
            model_name, output_attentions=False
        )
        self.m.predict = lambda x: self.m(x)

    @tf.function(input_signature=inputs_signature)
    def forward(self, inputs):
        return self.m.predict(inputs)


fail_models = [
    "facebook/data2vec-vision-base-ft1k",
    "microsoft/swin-tiny-patch4-window7-224",
]

supported_models = [
    # "facebook/convnext-tiny-224",
    "google/vit-base-patch16-224",
]

img_models_fe_dict = {
    "facebook/convnext-tiny-224": ConvNextFeatureExtractor,
    "facebook/data2vec-vision-base-ft1k": BeitFeatureExtractor,
    "microsoft/swin-tiny-patch4-window7-224": AutoFeatureExtractor,
    "google/vit-base-patch16-224": ViTFeatureExtractor,
}


def preprocess_input_image(model_name):
    # from datasets import load_dataset
    # dataset = load_dataset("huggingface/cats-image")
    # image1 = dataset["test"]["image"][0]
    # # print("image1: ", image1) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FA0B86BB6D0>
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FA0B86BB6D0>
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = img_models_fe_dict[model_name].from_pretrained(
        model_name
    )
    # inputs: {'pixel_values': <tf.Tensor: shape=(1, 3, 224, 224), dtype=float32, numpy=array([[[[]]]], dtype=float32)>}
    inputs = feature_extractor(images=image, return_tensors="tf")

    return [inputs[str(*inputs)]]


def get_causal_image_model(hf_name):
    model = AutoModelImageClassfication(hf_name)
    test_input = preprocess_input_image(hf_name)
    # TFSequenceClassifierOutput(loss=None, logits=<tf.Tensor: shape=(1, 1000), dtype=float32, numpy=
    # array([[]], dtype=float32)>, hidden_states=None, attentions=None)
    actual_out = model.forward(*test_input)
    return model, test_input, actual_out


if __name__ == "__main__":
    for model_name in supported_models:
        print(f"Running model: {model_name}")
        inputs = preprocess_input_image(model_name)
        model = AutoModelImageClassfication(model_name)

        # 1. USE SharkImporter to get the mlir
        # from shark.shark_importer import SharkImporter
        # mlir_importer = SharkImporter(
        #     model,
        #     inputs,
        #     frontend="tf",
        # )
        # imported_mlir, func_name = mlir_importer.import_mlir()

        # 2. USE SharkDownloader to get the mlir
        imported_mlir, func_name, inputs, golden_out = download_tf_model(
            model_name
        )

        shark_module = SharkInference(
            imported_mlir, func_name, device="cpu", mlir_dialect="mhlo"
        )
        shark_module.compile()
        shark_module.forward(inputs)
