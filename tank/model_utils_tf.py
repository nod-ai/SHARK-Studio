import tensorflow as tf
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    BertTokenizer,
    TFBertModel,
)

visible_default = tf.config.list_physical_devices("GPU")
try:
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 128

################################## MHLO/TF models #########################################
# TODO : Generate these lists or fetch model source from tank/tf/tf_model_list.csv
keras_models = [
    "resnet50",
]
maskedlm_models = [
    "albert-base-v2",
    "bert-base-uncased",
    "camembert-base",
    "convbert-base-turkish-cased",
    "deberta-base",
    "distilbert-base-uncased",
    "electra-small-discriminator",
    "funnel-transformer",
    "layoutlm-base-uncased",
    "longformer-base-4096",
    "mobilebert-uncased",
    "mpnet-base",
    "rembert",
    "roberta-base",
    "tapas-base",
    "tiny-random-flaubert",
    "xlm-roberta",
]
tfhf_models = [
    "microsoft/MiniLM-L12-H384-uncased",
]


def get_tf_model(name):
    if name in keras_models:
        return get_keras_model(name)
    elif name in maskedlm_models:
        return get_causal_lm_model(name)
    elif name in tfhf_models:
        return get_TFhf_model(name)
    else:
        return get_causal_image_model(name)


##################### Tensorflow Hugging Face LM Models ###################################
# Create a set of 2-dimensional inputs
tf_bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
]


class TFHuggingFaceLanguage(tf.Module):
    def __init__(self, hf_model_name):
        super(TFHuggingFaceLanguage, self).__init__()
        # Create a BERT trainer with the created network.
        self.m = TFBertModel.from_pretrained(hf_model_name, from_pt=True)

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m.predict = lambda x, y, z: self.m.call(
            input_ids=x, attention_mask=y, token_type_ids=z, training=False
        )

    @tf.function(input_signature=tf_bert_input)
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m.predict(input_ids, attention_mask, token_type_ids)


def get_TFhf_model(name):
    model = TFHuggingFaceLanguage(name)
    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/MiniLM-L12-H384-uncased"
    )
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    for key in encoded_input:
        encoded_input[key] = tf.expand_dims(
            tf.convert_to_tensor(encoded_input[key]), 0
        )
    test_input = (
        encoded_input["input_ids"],
        encoded_input["attention_mask"],
        encoded_input["token_type_ids"],
    )
    actual_out = model.forward(*test_input)
    return model, test_input, actual_out


# Utility function for comparing two tensors (tensorflow).
def compare_tensors_tf(tf_tensor, numpy_tensor):
    # setting the absolute and relative tolerance
    rtol = 1e-02
    atol = 1e-03
    tf_to_numpy = tf_tensor.numpy()
    return np.allclose(tf_to_numpy, numpy_tensor, rtol, atol)


##################### Tensorflow Hugging Face Masked LM Models ###################################
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf

# Create a set of input signature.
inputs_signature = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
]

# For supported models please see here:
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.TFAutoModelForCasualLM


def preprocess_input(
    model_name, text="This is just used to compile the model"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        text,
        padding="max_length",
        return_tensors="tf",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    return inputs


class MaskedLM(tf.Module):
    def __init__(self, model_name):
        super(MaskedLM, self).__init__()
        self.m = TFAutoModelForMaskedLM.from_pretrained(
            model_name, output_attentions=False, num_labels=2
        )
        self.m.predict = lambda x, y: self.m(input_ids=x, attention_mask=y)[0]

    @tf.function(input_signature=input_signature_maskedlm)
    def forward(self, input_ids, attention_mask):
        return self.m.predict(input_ids, attention_mask)


def get_causal_lm_model(hf_name, text="Hello, this is the default text."):
    model = MaskedLM(hf_name)
    encoded_input = preprocess_input(hf_name, text)
    test_input = (encoded_input["input_ids"], encoded_input["attention_mask"])
    actual_out = model.forward(*test_input)
    return model, test_input, actual_out


##################### TensorFlow Keras Resnet Models #########################################################
# Static shape, including batch size (1).
# Can be dynamic once dynamic shape support is ready.
INPUT_SHAPE = [1, 224, 224, 3]

tf_model = tf.keras.applications.resnet50.ResNet50(
    weights="imagenet", include_top=True, input_shape=tuple(INPUT_SHAPE[1:])
)


class ResNetModule(tf.Module):
    def __init__(self):
        super(ResNetModule, self).__init__()
        self.m = tf_model
        self.m.predict = lambda x: self.m.call(x, training=False)

    @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)])
    def forward(self, inputs):
        return self.m.predict(inputs)


def load_image(path_to_image):
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image[tf.newaxis, :]
    return image


def get_keras_model(modelname):
    model = ResNetModule()
    content_path = tf.keras.utils.get_file(
        "YellowLabradorLooking_new.jpg",
        "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
    )
    content_image = load_image(content_path)
    input_tensor = tf.keras.applications.resnet50.preprocess_input(
        content_image
    )
    input_data = tf.expand_dims(input_tensor, 0)
    actual_out = model.forward(*input_data)
    return model, input_data, actual_out


##################### Tensorflow Hugging Face  Image Classification Models ###################################
from transformers import TFAutoModelForImageClassification
from transformers import ConvNextFeatureExtractor, ViTFeatureExtractor
from transformers import BeitFeatureExtractor, AutoFeatureExtractor
from PIL import Image
import requests

# Create a set of input signature.
input_signature_img_cls = [
    tf.TensorSpec(shape=[1, 3, 224, 224], dtype=tf.float32),
]


class AutoModelImageClassfication(tf.Module):
    def __init__(self, model_name):
        super(AutoModelImageClassfication, self).__init__()
        self.m = TFAutoModelForImageClassification.from_pretrained(
            model_name, output_attentions=False
        )
        self.m.predict = lambda x: self.m(x)

    @tf.function(input_signature=input_signature_img_cls)
    def forward(self, inputs):
        return self.m.predict(inputs)


fail_models = [
    "facebook/data2vec-vision-base-ft1k",
    "microsoft/swin-tiny-patch4-window7-224",
]

supported_models = [
    "facebook/convnext-tiny-224",
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
