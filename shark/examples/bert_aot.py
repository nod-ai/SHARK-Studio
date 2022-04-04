from transformers import AutoModelForMaskedLM, BertConfig
import transformers
import torch
import torch.utils._pytree as pytree
from shark.shark_runner import SharkInference

pytree._register_pytree_node(
    transformers.modeling_outputs.MaskedLMOutput,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.MaskedLMOutput(
        loss=values[1], logits=values[1]
    ),
)

torch.manual_seed(42)

config = BertConfig()
model_type = AutoModelForMaskedLM
input_size = (1, 128)
device = "cpu"
dtype = torch.float

model = model_type.from_config(config).to(device, dtype=dtype)
input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
train_inputs = {"input_ids": input_ids, "labels": decoder_ids}

def inference_fn(model, input):
    return model(**input)

shark_module = SharkInference(
    model, train_inputs, from_aot=True, custom_inference_fn=inference_fn
)
