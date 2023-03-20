from shark.shark_inference import SharkInference
from shark.parser import shark_args

import torch
import numpy as np
import sys

torch.manual_seed(0)

BATCH_SIZE = 1

vision_models = [
    "alexnet",
    "resnet101",
    "resnet18",
    "resnet50",
    "resnet50_fp16",
    "squeezenet1_0",
    "wide_resnet50_2",
    "mobilenet_v3_small",
    "mnasnet1_0",
    "efficientnet_b0",
    "efficientnet_b7",
]
hf_img_cls_models = [
    "google/vit-base-patch16-224",
    "microsoft/resnet-50",
    "facebook/deit-small-distilled-patch16-224",
    "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "nvidia/mit-b0",
]
hf_seq2seq_models = [
    "t5-base",
    "t5-large",
]


def get_torch_model(modelname):
    if modelname in vision_models:
        return get_vision_model(modelname)
    elif modelname in hf_img_cls_models:
        return get_hf_img_cls_model(modelname)
    elif modelname in hf_seq2seq_models:
        return get_hf_seq2seq_model(modelname)
    elif "fp16" in modelname:
        return get_fp16_model(modelname)
    else:
        return get_hf_model(modelname)


##################### Hugging Face Image Classification Models ###################################
from transformers import AutoModelForImageClassification
from transformers import AutoFeatureExtractor
from PIL import Image
import requests


def preprocess_input_image(model_name):
    # from datasets import load_dataset
    # dataset = load_dataset("huggingface/cats-image")
    # image1 = dataset["test"]["image"][0]
    # # print("image1: ", image1) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FA0B86BB6D0>
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FA0B86BB6D0>
    image = Image.open(requests.get(url, stream=True).raw)
    # feature_extractor = img_models_fe_dict[model_name].from_pretrained(
    #     model_name
    # )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = feature_extractor(images=image, return_tensors="pt")
    # inputs = {'pixel_values': tensor([[[[ 0.1137..., -0.2000, -0.4275, -0.5294]]]])}
    #           torch.Size([1, 3, 224, 224]), torch.FloatTensor

    return inputs[str(*inputs)]


class HuggingFaceImageClassification(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            output_attentions=False,  # Whether the model returns attentions weights.
            return_dict=False,  # https://github.com/huggingface/transformers/issues/9095
            torchscript=True,
        )

    def forward(self, inputs):
        return self.model.forward(inputs)[0]


def get_hf_img_cls_model(name):
    model = HuggingFaceImageClassification(name)
    # you can use preprocess_input_image to get the test_input or just random value.
    test_input = preprocess_input_image(name)
    # test_input = torch.FloatTensor(1, 3, 224, 224).uniform_(-1, 1)
    # print("test_input.shape: ", test_input.shape)
    # test_input.shape:  torch.Size([1, 3, 224, 224])
    test_input = test_input.repeat(BATCH_SIZE, 1, 1, 1)
    actual_out = model(test_input)
    # print("actual_out.shape： ", actual_out.shape)
    # actual_out.shape：  torch.Size([1, 1000])
    return model, test_input, actual_out


##################### Hugging Face LM Models ###################################


class HuggingFaceLanguage(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        import transformers as trf

        transformers_path = trf.__path__[0]
        hf_model_path = f"{transformers_path}/models/{hf_model_name}"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_hf_model(name):
    from transformers import (
        BertTokenizer,
    )

    model = HuggingFaceLanguage(name)
    # TODO: Currently the test input is set to (1,128)
    test_input = torch.randint(2, (BATCH_SIZE, 128))
    actual_out = model(test_input)
    return model, test_input, actual_out


##################### Hugging Face Seq2SeqLM Models ###################################

# We use a maximum sequence length of 512 since this is the default used in the T5 config.
T5_MAX_SEQUENCE_LENGTH = 512


class HFSeq2SeqLanguageModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoTokenizer, T5Model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenization_kwargs = {
            "pad_to_multiple_of": T5_MAX_SEQUENCE_LENGTH,
            "padding": True,
            "return_tensors": "pt",
        }
        self.model = T5Model.from_pretrained(model_name, return_dict=True)

    def preprocess_input(self, text):
        return self.tokenizer(text, **self.tokenization_kwargs)

    def forward(self, input_ids, decoder_input_ids):
        return self.model.forward(
            input_ids, decoder_input_ids=decoder_input_ids
        )[0]


def get_hf_seq2seq_model(name):
    m = HFSeq2SeqLanguageModel(name)
    encoded_input_ids = m.preprocess_input(
        "Studies have been shown that owning a dog is good for you"
    ).input_ids
    decoder_input_ids = m.preprocess_input("Studies show that").input_ids
    decoder_input_ids = m.model._shift_right(decoder_input_ids)

    test_input = (encoded_input_ids, decoder_input_ids)
    actual_out = m.forward(*test_input)
    return m, test_input, actual_out


################################################################################

##################### Torch Vision Models    ###################################


class VisionModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


def get_vision_model(torch_model):
    import torchvision.models as models

    default_image_size = (224, 224)

    vision_models_dict = {
        "alexnet": (models.alexnet(weights="DEFAULT"), default_image_size),
        "resnet18": (models.resnet18(weights="DEFAULT"), default_image_size),
        "resnet50": (models.resnet50(weights="DEFAULT"), default_image_size),
        "resnet50_fp16": (
            models.resnet50(weights="DEFAULT"),
            default_image_size,
        ),
        "resnet101": (models.resnet101(weights="DEFAULT"), default_image_size),
        "squeezenet1_0": (
            models.squeezenet1_0(weights="DEFAULT"),
            default_image_size,
        ),
        "wide_resnet50_2": (
            models.wide_resnet50_2(weights="DEFAULT"),
            default_image_size,
        ),
        "mobilenet_v3_small": (
            models.mobilenet_v3_small(weights="DEFAULT"),
            default_image_size,
        ),
        "mnasnet1_0": (
            models.mnasnet1_0(weights="DEFAULT"),
            default_image_size,
        ),
        # EfficientNet input image size varies on the size of the model.
        "efficientnet_b0": (
            models.efficientnet_b0(weights="DEFAULT"),
            (224, 224),
        ),
        "efficientnet_b7": (
            models.efficientnet_b7(weights="DEFAULT"),
            (600, 600),
        ),
    }
    if isinstance(torch_model, str):
        fp16_model = None
        if "fp16" in torch_model:
            fp16_model = True
        torch_model, input_image_size = vision_models_dict[torch_model]
    model = VisionModule(torch_model)
    test_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    actual_out = model(test_input)
    if fp16_model is not None:
        test_input_fp16 = test_input.to(
            device=torch.device("cuda"), dtype=torch.half
        )
        model_fp16 = model.half()
        model_fp16.eval()
        model_fp16.to("cuda")
        actual_out_fp16 = model_fp16(test_input_fp16)
        model, test_input, actual_out = (
            model_fp16,
            test_input_fp16,
            actual_out_fp16,
        )
    return model, test_input, actual_out


################################################################################

####################### Other PyTorch HF Models ###############################


class BertHalfPrecisionModel(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        from transformers import AutoModelForMaskedLM

        self.model = AutoModelForMaskedLM.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
            torch_dtype=torch.float16,
        ).to("cuda")

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_fp16_model(torch_model):
    from transformers import AutoTokenizer

    modelname = torch_model.replace("_fp16", "")
    model = BertHalfPrecisionModel(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    text = "Replace me by any text you like."
    text = [text] * BATCH_SIZE
    test_input_fp16 = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).input_ids.to("cuda")
    # test_input = torch.randint(2, (1, 128))
    # test_input_fp16 = test_input.to(
    #    device=torch.device("cuda")
    # )
    model_fp16 = model.half()
    model_fp16.eval()
    with torch.no_grad():
        actual_out_fp16 = model_fp16(test_input_fp16)
    return model_fp16, test_input_fp16, actual_out_fp16


# Utility function for comparing two tensors (torch).
def compare_tensors(torch_tensor, numpy_tensor, rtol=1e-02, atol=1e-03):
    # torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_tensor, numpy_tensor, rtol, atol)
