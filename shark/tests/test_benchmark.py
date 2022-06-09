from shark.shark_inference import SharkInference
from shark.iree_utils import check_device_drivers

import torch
import tensorflow as tf
import numpy as np
import torchvision.models as models
from transformers import AutoModelForSequenceClassification, BertTokenizer, TFBertModel
import importlib
import pytest
import unittest

torch.manual_seed(0)

##################### Tensorflow Hugging Face LM Models ###################################
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of 2-dimensional inputs
tf_bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32)
]

class TFHuggingFaceLanguage(tf.Module):

    def __init__(self, hf_model_name):
        super(TFHuggingFaceLanguage, self).__init__()
        # Create a BERT trainer with the created network.
        self.m = TFBertModel.from_pretrained(
            hf_model_name, from_pt=True)

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m.predict = lambda x, y, z: self.m.call(
            input_ids=x, attention_mask=y, token_type_ids=z, training=False)

    @tf.function(input_signature=tf_bert_input)
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m.predict(input_ids, attention_mask, token_type_ids)


def get_TFhf_model(name):
    model = TFHuggingFaceLanguage(name)
    tokenizer = BertTokenizer.from_pretrained(
    "microsoft/MiniLM-L12-H384-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text,
                              padding='max_length',
                              truncation=True,
                              max_length=MAX_SEQUENCE_LENGTH)
    for key in encoded_input:
        encoded_input[key] = tf.expand_dims(
            tf.convert_to_tensor(encoded_input[key]), 0)
    test_input = (encoded_input["input_ids"], encoded_input["attention_mask"],
         encoded_input["token_type_ids"])
    actual_out = model.forward(*test_input)
    return model, test_input, actual_out

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


#############################   Benchmark Tests ####################################

pytest_benchmark_param = pytest.mark.parametrize(
    ('dynamic', 'device'),
    [
        pytest.param(False, 'cpu'),
        # TODO: Language models are failing for dynamic case..
        pytest.param(True, 'cpu', marks=pytest.mark.skip),
        pytest.param(False,
                     'gpu',
                     marks=pytest.mark.skipif(check_device_drivers("gpu"),
                                              reason="nvidia-smi not found")),
        pytest.param(True, 
                     'gpu', 
                     marks=pytest.mark.skipif(check_device_drivers("gpu"),
                                              reason="nvidia-smi not found")),
        pytest.param(
            False, 
            'vulkan', 
            marks=pytest.mark.skipif(
                check_device_drivers("vulkan"),
                reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )),
        pytest.param(
            True, 
            'vulkan', 
            marks=pytest.mark.skipif(
                check_device_drivers("vulkan"),
                reason="vulkaninfo not found, install from https://github.com/KhronosGroup/MoltenVK/releases"
            )),
    ])

class MiniLMTorchBenchmarkTest(unittest.TestCase):
    
    @pytest.fixture(autouse=True)
    def setup_benchmark(self, benchmark):
        self.benchmark = benchmark
    
    def setup_model(self):
        self.model, self.test_input, self.act_out = get_hf_model("microsoft/MiniLM-L12-H384-uncased")
    
    def setup_shark_module(self, dynamic, device):
        shark_module = SharkInference(self.model, (self.test_input,),
                                           device=device,
                                           dynamic=dynamic,
                                           jit_trace=True,
                                           benchmark_mode=True)
        return shark_module

    def test_bench_minilm_frontend(self):
        dynamic_opts = ["True", "False"]
        device_opts = ["cpu", "gpu", "vulkan"]
        self.setup_model()
        
        for dynamic, device in zip(dynamic_opts, device_opts):

            shark_module = self.setup_shark_module(dynamic, device)
            try:
                shark_module.benchmark_frontend((self.test_input,))
                assert True
            except Exception as e:
                assert False
    
    def test_bench_minilm_python(self):
        dynamic_opts = ["True", "False"]
        device_opts = ["cpu", "gpu", "vulkan"]
        self.setup_model()

        for dynamic, device in zip(dynamic_opts, device_opts):

            self.setup_shark_module(dynamic, device)
            try:
                self.shark_module.benchmark_python((self.test_input,)))
                assert True
            except Exception as e:
                assert False
            
    def test_bench_minilm_c(self):
        dynamic_opts = ["True", "False"]
        device_opts = ["cpu", "gpu", "vulkan"]
        self.setup_model()
        
        for dynamic, device in zip(dynamic_opts, device_opts):
            self.setup_shark_module(dynamic, device)
            try:
                self.shark_module.benchmark_frontend()
                assert True
            except Exception as e:
                assert False
            
class MiniLMTFBenchmarkTest(unittest.TestCase):
    
    def setup_model(self):
        self.model, self.test_input, self.act_out = get_TFhf_model("microsoft/MiniLM-L12-H384-uncased")
    
    def setup_and_compile_module(self, dynamic, device):
        self.shark_module = SharkInference(self.model, self.test_input,
                                           device=device,
                                           dynamic=dynamic,
                                           jit_trace=True,
                                           benchmark_mode=True)
        self.shark_module.set_frontend("tensorflow")
        self.shark_module.compile()
    
    @pytest.mark.skipif(importlib.util.find_spec("iree.tools") is None, reason = "Cannot find tools to import TF")
    def test_bench_minilm_frontend(self,):
        dynamic_opts = ["True", "False"]
        device_opts = ["cpu", "gpu", "vulkan"]
        self.setup_model()
        
        for dynamic, device in zip(dynamic_opts, device_opts):
            self.setup_and_compile_module(dynamic, device)
            try:
                self.shark_module.benchmark_frontend(self.test_input)
                assert True
            except Exception as e:
                assert False
    
    @pytest.mark.skipif(importlib.util.find_spec("iree.tools") is None, reason = "Cannot find tools to import TF")
    def test_bench_minilm_python(self):
        dynamic_opts = ["True", "False"]
        device_opts = ["cpu", "gpu", "vulkan"]
        self.setup_model()
        
        for dynamic, device in zip(dynamic_opts, device_opts):
            self.setup_and_compile_module(dynamic, device)
            try:
                self.shark_module.benchmark_python(self.test_input)
                assert True
            except Exception as e:
                assert False
            
    @pytest.mark.skipif(importlib.util.find_spec("iree.tools") is None, reason = "Cannot find tools to import TF")
    def test_bench_minilm_c(self):
        dynamic_opts = ["True", "False"]
        device_opts = ["cpu", "gpu", "vulkan"]
        self.setup_model()
        
        for dynamic, device in zip(dynamic_opts, device_opts):
            self.setup_and_compile_module(dynamic, device)
            try:
                self.shark_module.benchmark_frontend()
                assert True
            except Exception as e:
                assert False
           
