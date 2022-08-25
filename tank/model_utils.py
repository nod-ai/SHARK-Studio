from shark.shark_inference import SharkInference
from shark.parser import shark_args

import torch
import numpy as np
import sys

torch.manual_seed(0)

vision_models = [
    "alexnet",
    "resnet101",
    "resnet18",
    "resnet50",
    "squeezenet1_0",
    "wide_resnet50_2",
    "mobilenet_v3_small",
]


def get_torch_model(modelname):
    if modelname in vision_models:
        return get_vision_model(modelname)
    else:
        return get_hf_model(modelname)

##################### Hugging Face Audio Classification Models ###################################
from transformers import  AutoModelForAudioClassification, AutoProcessor
from transformers import AutoFeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Model

def preprocess_input_audio(model_name):
    from datasets import load_dataset
    processor = AutoProcessor.from_pretrained(model_name)
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    input_values = processor(ds[0]["audio"]["array"], return_tensors="pt",
                             padding="longest").input_values  # Batch size 1
    return input_values

class HuggingFaceAudioClassification(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            output_attentions=False,  # Whether the model returns attentions weights.
            return_dict=False,  # https://github.com/huggingface/transformers/issues/9095
            torchscript=True,
        )

    def forward(self, inputs):
        return self.model.forward(inputs)[0]

def get_hf_speech_cls_model(name):
    model = HuggingFaceAudioClassification(name)
    # you can use preprocess_input_image to get the test_input or just random value.
    test_input = preprocess_input_audio(name)
    # test_input = torch.FloatTensor(1, 174160).uniform_(-1, 1)
    print("test_input: ", test_input)  # tensor([[-0.0014, -0.0037, -0.0019,  ..., -0.0467, -0.0585, -0.0571]])
    print("test_input.type: ", test_input.dtype) # torch.float32
    print("test_input.shape: ", test_input.shape)
    # test_input.shape:  torch.Size([1, 174160])
    actual_out = model(test_input)
    print("actual_out： ", actual_out)
    print("actual_out.shape： ", actual_out.shape)
    # actual_out.shape：  torch.Size([1, 544, 32])
    return model, test_input, actual_out

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
    # test_input = preprocess_input_image(name)
    test_input = torch.FloatTensor(1, 3, 224, 224).uniform_(-1, 1)
    print("test_input.shape: ", test_input.shape)
    # test_input.shape:  torch.Size([1, 3, 224, 224])
    actual_out = model(test_input)
    print("actual_out.shape： ", actual_out.shape)
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
        TFBertModel,
    )

    model = HuggingFaceLanguage(name)
    # TODO: Currently the test input is set to (1,128)
    test_input = torch.randint(2, (1, 128))
    actual_out = model(test_input)
    return model, test_input, actual_out


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

    vision_models_dict = {
        "alexnet": models.alexnet(pretrained=True),
        "resnet18": models.resnet18(pretrained=True),
        "resnet50": models.resnet50(pretrained=True),
        "resnet101": models.resnet101(pretrained=True),
        "squeezenet1_0": models.squeezenet1_0(pretrained=True),
        "wide_resnet50_2": models.wide_resnet50_2(pretrained=True),
        "mobilenet_v3_small": models.mobilenet_v3_small(pretrained=True),
    }
    if isinstance(torch_model, str):
        torch_model = vision_models_dict[torch_model]
    model = VisionModule(torch_model)
    test_input = torch.randn(1, 3, 224, 224)
    actual_out = model(test_input)
    return model, test_input, actual_out


################################################################################

# Utility function for comparing two tensors (torch).
def compare_tensors(torch_tensor, numpy_tensor, rtol=1e-02, atol=1e-03):
    # torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_tensor, numpy_tensor, rtol, atol)
