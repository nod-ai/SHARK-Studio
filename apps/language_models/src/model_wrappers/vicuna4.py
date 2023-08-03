import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
import numpy as np
import iree.runtime
import itertools
import subprocess

import torch
import torch_mlir
from torch_mlir import TensorPlaceholder
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaPreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.src.model_wrappers.vicuna_sharded_model import (
    FirstVicunaLayer,
    SecondVicunaLayer,
    CompiledVicunaLayer,
    ShardedVicunaModel,
    LMHead,
    LMHeadCompiled,
    VicunaEmbedding,
    VicunaEmbeddingCompiled,
    VicunaNorm,
    VicunaNormCompiled,
)
from apps.language_models.src.model_wrappers.vicuna_model import (
    FirstVicuna,
    SecondVicuna,
)
from apps.language_models.utils import (
    get_vmfb_from_path,
)
from shark.shark_downloader import download_public_file
from shark.shark_importer import get_f16_inputs
from shark.shark_importer import import_with_fx
from shark.shark_inference import SharkInference

from brevitas_examples.llm.llm_quant.quantize import quantize_model
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    _make_causal_mask,
    _expand_mask,
)
from torch import nn
from time import time


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        t1 = time()
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = (
                seq_length_with_past + past_key_values_length
            )

        if position_ids is None:
            device = (
                input_ids.device
                if input_ids is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.compressedlayers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[8 * idx : 8 * (idx + 1)]
                if past_key_values is not None
                else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1:],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        try:
            hidden_states = np.asarray(hidden_states, hidden_states.dtype)
        except:
            _ = 10

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        next_cache = tuple(itertools.chain.from_iterable(next_cache))
        print(f"Token generated in {time() - t1} seconds")
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class EightLayerLayerSV(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        assert len(layers) == 8
        self.layers = layers

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        pkv00,
        pkv01,
        pkv10,
        pkv11,
        pkv20,
        pkv21,
        pkv30,
        pkv31,
        pkv40,
        pkv41,
        pkv50,
        pkv51,
        pkv60,
        pkv61,
        pkv70,
        pkv71,
    ):
        pkvs = [
            (pkv00, pkv01),
            (pkv10, pkv11),
            (pkv20, pkv21),
            (pkv30, pkv31),
            (pkv40, pkv41),
            (pkv50, pkv51),
            (pkv60, pkv61),
            (pkv70, pkv71),
        ]
        new_pkvs = []
        for layer, pkv in zip(self.layers, pkvs):
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(
                    pkv[0],
                    pkv[1],
                ),
                use_cache=True,
            )

            hidden_states = outputs[0]
            new_pkvs.append(
                (
                    outputs[-1][0],
                    outputs[-1][1],
                )
            )
        (
            (new_pkv00, new_pkv01),
            (new_pkv10, new_pkv11),
            (new_pkv20, new_pkv21),
            (new_pkv30, new_pkv31),
            (new_pkv40, new_pkv41),
            (new_pkv50, new_pkv51),
            (new_pkv60, new_pkv61),
            (new_pkv70, new_pkv71),
        ) = new_pkvs
        return (
            hidden_states,
            new_pkv00,
            new_pkv01,
            new_pkv10,
            new_pkv11,
            new_pkv20,
            new_pkv21,
            new_pkv30,
            new_pkv31,
            new_pkv40,
            new_pkv41,
            new_pkv50,
            new_pkv51,
            new_pkv60,
            new_pkv61,
            new_pkv70,
            new_pkv71,
        )


class EightLayerLayerFV(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        assert len(layers) == 8
        self.layers = layers

    def forward(self, hidden_states, attention_mask, position_ids):
        new_pkvs = []
        for layer in self.layers:
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=True,
            )

            hidden_states = outputs[0]
            new_pkvs.append(
                (
                    outputs[-1][0],
                    outputs[-1][1],
                )
            )
        (
            (new_pkv00, new_pkv01),
            (new_pkv10, new_pkv11),
            (new_pkv20, new_pkv21),
            (new_pkv30, new_pkv31),
            (new_pkv40, new_pkv41),
            (new_pkv50, new_pkv51),
            (new_pkv60, new_pkv61),
            (new_pkv70, new_pkv71),
        ) = new_pkvs
        return (
            hidden_states,
            new_pkv00,
            new_pkv01,
            new_pkv10,
            new_pkv11,
            new_pkv20,
            new_pkv21,
            new_pkv30,
            new_pkv31,
            new_pkv40,
            new_pkv41,
            new_pkv50,
            new_pkv51,
            new_pkv60,
            new_pkv61,
            new_pkv70,
            new_pkv71,
        )


class CompiledEightLayerLayerSV(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions=False,
        use_cache=True,
    ):
        hidden_states = hidden_states.detach()
        attention_mask = attention_mask.detach()
        position_ids = position_ids.detach()
        (
            (pkv00, pkv01),
            (pkv10, pkv11),
            (pkv20, pkv21),
            (pkv30, pkv31),
            (pkv40, pkv41),
            (pkv50, pkv51),
            (pkv60, pkv61),
            (pkv70, pkv71),
        ) = past_key_value
        pkv00 = pkv00.detatch()
        pkv01 = pkv01.detatch()
        pkv10 = pkv10.detatch()
        pkv11 = pkv11.detatch()
        pkv20 = pkv20.detatch()
        pkv21 = pkv21.detatch()
        pkv30 = pkv30.detatch()
        pkv31 = pkv31.detatch()
        pkv40 = pkv40.detatch()
        pkv41 = pkv41.detatch()
        pkv50 = pkv50.detatch()
        pkv51 = pkv51.detatch()
        pkv60 = pkv60.detatch()
        pkv61 = pkv61.detatch()
        pkv70 = pkv70.detatch()
        pkv71 = pkv71.detatch()

        output = self.model(
            "forward",
            (
                hidden_states,
                attention_mask,
                position_ids,
                pkv00,
                pkv01,
                pkv10,
                pkv11,
                pkv20,
                pkv21,
                pkv30,
                pkv31,
                pkv40,
                pkv41,
                pkv50,
                pkv51,
                pkv60,
                pkv61,
                pkv70,
                pkv71,
            ),
            send_to_host=False,
        )
        return (
            output[0],
            (output[1][0], output[1][1]),
            (output[2][0], output[2][1]),
            (output[3][0], output[3][1]),
            (output[4][0], output[4][1]),
            (output[5][0], output[5][1]),
            (output[6][0], output[6][1]),
            (output[7][0], output[7][1]),
            (output[8][0], output[8][1]),
        )


def forward_compressed(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = (
            input_ids.device if input_ids is not None else inputs_embeds.device
        )
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length,
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.compressedlayers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = (
            past_key_values[8 * idx : 8 * (idx + 1)]
            if past_key_values is not None
            else None
        )

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (
                layer_outputs[2 if output_attentions else 1],
            )

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
            ]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


class CompiledEightLayerLayer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
    ):
        t2 = time()
        if past_key_value is None:
            try:
                hidden_states = np.asarray(hidden_states, hidden_states.dtype)
            except:
                pass
            attention_mask = attention_mask.detach()
            position_ids = position_ids.detach()
            t1 = time()

            output = self.model(
                "first_vicuna_forward",
                (hidden_states, attention_mask, position_ids),
                send_to_host=False,
            )
            output2 = (
                output[0],
                (
                    output[1],
                    output[2],
                ),
                (
                    output[3],
                    output[4],
                ),
                (
                    output[5],
                    output[6],
                ),
                (
                    output[7],
                    output[8],
                ),
                (
                    output[9],
                    output[10],
                ),
                (
                    output[11],
                    output[12],
                ),
                (
                    output[13],
                    output[14],
                ),
                (
                    output[15],
                    output[16],
                ),
            )
            return output2
        else:
            (
                (pkv00, pkv01),
                (pkv10, pkv11),
                (pkv20, pkv21),
                (pkv30, pkv31),
                (pkv40, pkv41),
                (pkv50, pkv51),
                (pkv60, pkv61),
                (pkv70, pkv71),
            ) = past_key_value

            try:
                hidden_states = hidden_states.detach()
                attention_mask = attention_mask.detach()
                position_ids = position_ids.detach()
                pkv00 = pkv00.detach()
                pkv01 = pkv01.detach()
                pkv10 = pkv10.detach()
                pkv11 = pkv11.detach()
                pkv20 = pkv20.detach()
                pkv21 = pkv21.detach()
                pkv30 = pkv30.detach()
                pkv31 = pkv31.detach()
                pkv40 = pkv40.detach()
                pkv41 = pkv41.detach()
                pkv50 = pkv50.detach()
                pkv51 = pkv51.detach()
                pkv60 = pkv60.detach()
                pkv61 = pkv61.detach()
                pkv70 = pkv70.detach()
                pkv71 = pkv71.detach()
            except:
                x = 10

            t1 = time()
            if type(hidden_states) == iree.runtime.array_interop.DeviceArray:
                hidden_states = np.array(hidden_states, hidden_states.dtype)
                hidden_states = torch.tensor(hidden_states)
                hidden_states = hidden_states.detach()

            output = self.model(
                "second_vicuna_forward",
                (
                    hidden_states,
                    attention_mask,
                    position_ids,
                    pkv00,
                    pkv01,
                    pkv10,
                    pkv11,
                    pkv20,
                    pkv21,
                    pkv30,
                    pkv31,
                    pkv40,
                    pkv41,
                    pkv50,
                    pkv51,
                    pkv60,
                    pkv61,
                    pkv70,
                    pkv71,
                ),
                send_to_host=False,
            )
            print(f"{time() - t1}")
            del pkv00
            del pkv01
            del pkv10
            del pkv11
            del pkv20
            del pkv21
            del pkv30
            del pkv31
            del pkv40
            del pkv41
            del pkv50
            del pkv51
            del pkv60
            del pkv61
            del pkv70
            del pkv71
            output2 = (
                output[0],
                (
                    output[1],
                    output[2],
                ),
                (
                    output[3],
                    output[4],
                ),
                (
                    output[5],
                    output[6],
                ),
                (
                    output[7],
                    output[8],
                ),
                (
                    output[9],
                    output[10],
                ),
                (
                    output[11],
                    output[12],
                ),
                (
                    output[13],
                    output[14],
                ),
                (
                    output[15],
                    output[16],
                ),
            )
            return output2
