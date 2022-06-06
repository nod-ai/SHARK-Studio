from typing import Optional, List

import torch.jit
from torch import tensor, nn
from transformers import pipeline, AutoConfig, OPTForCausalLM, OPTConfig
from transformers.pipelines import infer_framework_load_model, check_task

generator = pipeline("text-generation", model="facebook/opt-350m")
print(generator("Hello, I'm am conscious and"))

model = "facebook/opt-350m"
task = "text-generation"
model_kwargs = {"use_auth_token": None, "return_dict": True}
config = AutoConfig.from_pretrained(
    model, revision=None, _from_pipeline=task, **model_kwargs
)
targeted_task, task_options = check_task(task)
model_classes = {"tf": targeted_task["tf"], "pt": targeted_task["pt"]}
framework, model = infer_framework_load_model(
    model,
    model_classes=model_classes,
    config=config,
    framework=None,
    revision=None,
    task=task,
    **model_kwargs,
)

print(model)

input_ids: torch.LongTensor = torch.LongTensor([[2, 31414, 6, 38, 437, 524, 13316, 8]])
attention_mask: Optional[torch.Tensor] = tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
head_mask: Optional[torch.Tensor] = None
past_key_values: Optional[List[torch.FloatTensor]] = None
inputs_embeds: Optional[torch.FloatTensor] = None
labels: Optional[torch.LongTensor] = None
use_cache: Optional[bool] = None
output_attentions: Optional[bool] = False
output_hidden_states: Optional[bool] = False
return_dict: Optional[bool] = True


inputs = dict(
    input_ids=input_ids,
    attention_mask=attention_mask,
    head_mask=head_mask,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    labels=labels,
    use_cache=use_cache,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    return_dict=False,
)

model(**inputs)


config = OPTConfig(
    **{
        "_name_or_path": "facebook/opt-350m",
        "activation_dropout": 0.0,
        "activation_function": "relu",
        "architectures": ["OPTForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 2,
        "do_layer_norm_before": False,
        "dropout": 0.1,
        "eos_token_id": 2,
        "ffn_dim": 4096,
        "hidden_size": 1024,
        "init_std": 0.02,
        "layerdrop": 0.0,
        "max_position_embeddings": 2048,
        "model_type": "opt",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 1,
        "prefix": "</s>",
        "torch_dtype": "float16",
        "transformers_version": "4.19.2",
        "use_cache": True,
        "vocab_size": 50272,
        "word_embed_proj_dim": 512,
    }
)


class OPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt = OPTForCausalLM(config)

    def forward(self, input_ids, attention_mask):
        return self.opt(input_ids, attention_mask, return_dict=False)


opt = OPT()

ts = torch.jit.trace(opt, (input_ids, attention_mask))
print(ts.graph)
