from shark.shark_inference import SharkInference

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
def compare_tensors(torch_tensor, numpy_tensor):
    # setting the absolute and relative tolerance
    rtol = 1e-02
    atol = 1e-03
    # torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_tensor, numpy_tensor, rtol, atol)
