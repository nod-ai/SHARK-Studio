from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    BertConfig,
)
import transformers
import torch
from functorch.compile import (
    memory_efficient_fusion,
    aot_module,
    draw_graph_compile,
    nop,
    min_cut_rematerialization_partition,
)
import torch.utils._pytree as pytree
import time
from torch import optim, fx
import torch.nn as nn
from torch.nn.utils import _stateless
from typing import List
from shark_runner import SharkInference

pytree._register_pytree_node(
    transformers.modeling_outputs.MaskedLMOutput,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.MaskedLMOutput(
        loss=values[1], logits=values[1]
    ),
)

pytree._register_pytree_node(
    transformers.modeling_outputs.Seq2SeqLMOutput,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.Seq2SeqLMOutput(
        loss=values[0], logits=values[1]
    ),
)

pytree._register_pytree_node(
    transformers.modeling_outputs.CausalLMOutputWithCrossAttentions,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(
        loss=values[0], logits=values[1]
    ),
)

pytree._register_pytree_node(
    transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput,
    lambda x: ([x.loss, x.logits], None),
    lambda values, _: transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput(
        loss=values[0], logits=values[1]
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
