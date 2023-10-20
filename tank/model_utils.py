from shark.shark_inference import SharkInference

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


def get_torch_model(modelname, import_args):
    if modelname in vision_models:
        return get_vision_model(modelname, import_args)
    elif modelname in hf_img_cls_models:
        return get_hf_img_cls_model(modelname, import_args)
    elif modelname in hf_seq2seq_models:
        return get_hf_seq2seq_model(modelname, import_args)
    elif "fp16" in modelname:
        return get_fp16_model(modelname, import_args)
    else:
        return get_hf_causallm_model(modelname, import_args)


##################################################################################################

##################### Hugging Face Image Classification Models ###################################

from transformers import AutoModelForImageClassification
from transformers import AutoFeatureExtractor
from PIL import Image
import requests


def preprocess_input_image(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = feature_extractor(images=image, return_tensors="pt")

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
        self.train(False)

    def forward(self, inputs):
        return self.model.forward(inputs)[0]


def get_hf_img_cls_model(name, import_args):
    model = HuggingFaceImageClassification(name)
    test_input = preprocess_input_image(name)
    print(test_input.size())
    test_input = test_input.repeat(int(import_args["batch_size"]), 1, 1, 1)
    with torch.no_grad():
        actual_out = model(test_input)
    return model, test_input, actual_out


#######################################################################################

##################### Hugging Face SeqClsLM Models ####################################


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


def get_hf_model(name, import_args):
    from transformers import (
        BertTokenizer,
    )

    model = HuggingFaceLanguage(name)
    test_input = torch.randint(2, (1, 128))
    actual_out = model(test_input)
    return model, test_input, actual_out


#######################################################################################

##################### Hugging Face Seq2SeqLM Models ###################################

# We use a maximum sequence length of 512 since this is the default used in the T5 config.
S2S_MAX_SEQUENCE_LENGTH = 512


class HFSeq2SeqLanguageModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            output_attentions=False,  # Whether the model returns attentions weights.
            torchscript=True,
        )
        self.model.eval()

    def preprocess_input(self, text):
        encoded_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        prepared_tokens = (
            encoded_inputs["input_ids"],
            encoded_inputs["attention_mask"],
        )
        return prepared_tokens

    def forward(self, input_ids, attention_mask):
        combine_input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        output = self.model(**combine_input_dict)
        return output[0]


def get_hf_seq2seq_model(name, import_args):
    m = HFSeq2SeqLanguageModel(name)
    encoded_inputs = m.preprocess_input(
        "Studies have been shown that owning a dog is good for you"
    )

    actual_out = m.forward(*encoded_inputs)
    return m, encoded_inputs, actual_out


######################################################################################


##################### Hugging Face CausalLM Models ###################################
from transformers import AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    encoded_inputs = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    prepared_tokens = (
        encoded_inputs["input_ids"],
        encoded_inputs["attention_mask"],
    )
    return prepared_tokens


class HFCausalLM(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,  # The pretrained model name.
        )
        self.model.eval()

    def forward(self, input_tuple):
        combine_input_dict = {
            "input_ids": input_tuple[0],
            "attention_mask": input_tuple[1],
        }
        output = self.model(**combine_input_dict)
        return output.logits


class OPTForCausalLMModel(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = OPTForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        combine_input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        output = self.model(**combine_input_dict)
        return output.logits


def get_hf_causallm_model(name, import_args):
    test_input = prepare_sentence_tokens(
        name, "this project is very interesting"
    )
    if any(word in name for word in ["opt", "gpt", "deberta"]):
        m = OPTForCausalLMModel(name)
        actual_out = m.forward(test_input[0], test_input[1])
    else:
        m = HFCausalLM(name)
        actual_out = m.forward(test_input)
    return m, test_input, actual_out


################################################################################

##################### Torch Vision Models    ###################################


class VisionModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, input):
        return self.model.forward(
            (input),
        )


def get_vision_model(torch_model, import_args):
    import torchvision
    from torchvision.models import get_model

    default_image_size = (224, 224)
    modelname = torch_model
    if modelname == "efficientnet_b0":
        input_image_size = (224, 224)
    elif modelname == "efficientnet_b7":
        input_image_size = (600, 600)
    else:
        input_image_size = default_image_size
    weights = "DEFAULT"

    torch_model = get_model(modelname, weights=weights)
    model = VisionModule(torch_model)
    test_input = torch.randn(
        int(import_args["batch_size"]),
        3,
        input_image_size[0],
        input_image_size[1],
    )
    with torch.no_grad():
        actual_out = model(test_input)
    return model, test_input, actual_out


# Utility function for comparing two tensors (torch).
def compare_tensors(torch_tensor, numpy_tensor, rtol=1e-02, atol=1e-03):
    # torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_tensor, numpy_tensor, rtol, atol)
