from shark.shark_runner import SharkInference

import torch
import numpy as np
import torchvision.models as models
from transformers import AutoModelForSequenceClassification
import pytest

torch.manual_seed(0)

##################### Hugging Face LM Models ###################################


class HuggingFaceLanguage(torch.nn.Module):

    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=
            2,  # The number of output labels--2 for binary classification.
            output_attentions=
            False,  # Whether the model returns attentions weights.
            output_hidden_states=
            False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_hf_model(name):
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
    model = VisionModule(torch_model)
    # TODO: Currently the test input is set to (1,128)
    test_input = torch.randn(1, 3, 224, 224)
    actual_out = model(test_input)
    return model, test_input, actual_out


################################################################################


# Utility function for comparing two tensors.
def compare_tensors(torch_tensor, numpy_tensor):
    # setting the absolute and relative tolerance
    rtol = 1e-02
    atol = 1e-03
    torch_to_numpy = torch_tensor.detach().numpy()
    return np.allclose(torch_to_numpy, numpy_tensor, rtol, atol)


################################################################################

#############################   Model Tests ####################################

# Please remove the skip flag to run the test with required param.
pytest_param = pytest.mark.parametrize(('dynamic', 'device'), [
    pytest.param(False, 'cpu'),
    pytest.param(
        True,
        'cpu',
        marks=pytest.mark.skip(reason="dynamic-shape is not supported.")),
    pytest.param(
        False,
        'gpu',
        marks=pytest.mark.skip(reason="device in the CI not supported.")),
    pytest.param(
        True,
        'gpu',
        marks=pytest.mark.skip(reason="device in the CI not supported.")),
    pytest.param(
        False,
        'vulkan',
        marks=pytest.mark.skip(reason="device in the CI not supported.")),
    pytest.param(
        True,
        'vulkan',
        marks=pytest.mark.skip(reason="device in the CI not supported.")),
])


@pytest_param
def test_bert(dynamic, device):
    model, input, act_out = get_hf_model("bert-base-uncased")
    shark_module = SharkInference(model, (input,),
                                  device=device,
                                  dynamic=dynamic,
                                  jit_trace=True)
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)


@pytest_param
def test_albert(dynamic, device):
    model, input, act_out = get_hf_model("albert-base-v2")
    shark_module = SharkInference(model, (input,),
                                  device=device,
                                  dynamic=dynamic,
                                  jit_trace=True)
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)


@pytest_param
def test_resnet18(dynamic, device):
    model, input, act_out = get_vision_model(models.resnet18(pretrained=True))
    shark_module = SharkInference(
        model,
        (input,),
        device=device,
        dynamic=dynamic,
    )
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)


@pytest_param
def test_resnet50(dynamic, device):
    model, input, act_out = get_vision_model(models.resnet50(pretrained=True))
    shark_module = SharkInference(
        model,
        (input,),
        device=device,
        dynamic=dynamic,
    )
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)


@pytest_param
def test_wide_resnet50(dynamic, device):
    model, input, act_out = get_vision_model(
        models.wide_resnet50_2(pretrained=True))
    shark_module = SharkInference(
        model,
        (input,),
        device=device,
        dynamic=dynamic,
    )
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)


@pytest_param
def test_minilm(dynamic, device):
    model, input, act_out = get_hf_model("microsoft/MiniLM-L12-H384-uncased")
    shark_module = SharkInference(model, (input,),
                                  device=device,
                                  dynamic=dynamic,
                                  jit_trace=True)
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)


@pytest_param
def test_squeezenet(dynamic, device):
    model, input, act_out = get_vision_model(
        models.squeezenet1_0(pretrained=True))
    shark_module = SharkInference(
        model,
        (input,),
        device=device,
        dynamic=dynamic,
    )
    results = shark_module.forward((input,))
    assert True == compare_tensors(act_out, results)
