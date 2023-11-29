import torch
from typing import Optional, Tuple


class WordEmbeddingsLayer(torch.nn.Module):
    def __init__(self, word_embedding_layer):
        super().__init__()
        self.model = word_embedding_layer

    def forward(self, input_ids):
        output = self.model.forward(input=input_ids)
        return output


class CompiledWordEmbeddingsLayer(torch.nn.Module):
    def __init__(self, compiled_word_embedding_layer):
        super().__init__()
        self.model = compiled_word_embedding_layer

    def forward(self, input_ids):
        input_ids = input_ids.detach().numpy()
        new_input_ids = self.model("forward", input_ids)
        new_input_ids = new_input_ids.reshape(
            [1, new_input_ids.shape[0], new_input_ids.shape[1]]
        )
        return torch.tensor(new_input_ids)


class LNFEmbeddingLayer(torch.nn.Module):
    def __init__(self, ln_f):
        super().__init__()
        self.model = ln_f

    def forward(self, hidden_states):
        output = self.model.forward(input=hidden_states)
        return output


class CompiledLNFEmbeddingLayer(torch.nn.Module):
    def __init__(self, ln_f):
        super().__init__()
        self.model = ln_f

    def forward(self, hidden_states):
        hidden_states = hidden_states.detach().numpy()
        new_hidden_states = self.model("forward", (hidden_states,))

        return torch.tensor(new_hidden_states)


class LMHeadEmbeddingLayer(torch.nn.Module):
    def __init__(self, embedding_layer):
        super().__init__()
        self.model = embedding_layer

    def forward(self, hidden_states):
        output = self.model.forward(input=hidden_states)
        return output


class CompiledLMHeadEmbeddingLayer(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.model = lm_head

    def forward(self, hidden_states):
        hidden_states = hidden_states.detach().numpy()
        new_hidden_states = self.model("forward", (hidden_states,))
        return torch.tensor(new_hidden_states)


class FourWayShardingDecoderLayer(torch.nn.Module):
    def __init__(self, decoder_layer_model, falcon_variant):
        super().__init__()
        self.model = decoder_layer_model
        self.falcon_variant = falcon_variant

    def forward(self, hidden_states, attention_mask):
        new_pkvs = []
        for layer in self.model:
            outputs = layer(
                hidden_states=hidden_states,
                alibi=None,
                attention_mask=attention_mask,
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
            (new_pkv80, new_pkv81),
            (new_pkv90, new_pkv91),
            (new_pkv100, new_pkv101),
            (new_pkv110, new_pkv111),
            (new_pkv120, new_pkv121),
            (new_pkv130, new_pkv131),
            (new_pkv140, new_pkv141),
            (new_pkv150, new_pkv151),
            (new_pkv160, new_pkv161),
            (new_pkv170, new_pkv171),
            (new_pkv180, new_pkv181),
            (new_pkv190, new_pkv191),
        ) = new_pkvs
        result = (
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
            new_pkv80,
            new_pkv81,
            new_pkv90,
            new_pkv91,
            new_pkv100,
            new_pkv101,
            new_pkv110,
            new_pkv111,
            new_pkv120,
            new_pkv121,
            new_pkv130,
            new_pkv131,
            new_pkv140,
            new_pkv141,
            new_pkv150,
            new_pkv151,
            new_pkv160,
            new_pkv161,
            new_pkv170,
            new_pkv171,
            new_pkv180,
            new_pkv181,
            new_pkv190,
            new_pkv191,
        )
        return result


class CompiledFourWayShardingDecoderLayer(torch.nn.Module):
    def __init__(
        self, layer_id, device_idx, falcon_variant, device, precision, model
    ):
        super().__init__()
        self.layer_id = layer_id
        self.device_index = device_idx
        self.falcon_variant = falcon_variant
        self.device = device
        self.precision = precision
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        import gc

        torch.cuda.empty_cache()
        gc.collect()

        if self.model is None:
            raise ValueError("Layer vmfb not found")

        hidden_states = hidden_states.to(torch.float32).detach().numpy()
        attention_mask = attention_mask.to(torch.float32).detach().numpy()

        if alibi is not None or layer_past is not None:
            raise ValueError("Past Key Values and alibi should be None")
        else:
            output = self.model(
                "forward",
                (
                    hidden_states,
                    attention_mask,
                ),
            )

        result = (
            torch.tensor(output[0]),
            (
                torch.tensor(output[1]),
                torch.tensor(output[2]),
            ),
            (
                torch.tensor(output[3]),
                torch.tensor(output[4]),
            ),
            (
                torch.tensor(output[5]),
                torch.tensor(output[6]),
            ),
            (
                torch.tensor(output[7]),
                torch.tensor(output[8]),
            ),
            (
                torch.tensor(output[9]),
                torch.tensor(output[10]),
            ),
            (
                torch.tensor(output[11]),
                torch.tensor(output[12]),
            ),
            (
                torch.tensor(output[13]),
                torch.tensor(output[14]),
            ),
            (
                torch.tensor(output[15]),
                torch.tensor(output[16]),
            ),
            (
                torch.tensor(output[17]),
                torch.tensor(output[18]),
            ),
            (
                torch.tensor(output[19]),
                torch.tensor(output[20]),
            ),
            (
                torch.tensor(output[21]),
                torch.tensor(output[22]),
            ),
            (
                torch.tensor(output[23]),
                torch.tensor(output[24]),
            ),
            (
                torch.tensor(output[25]),
                torch.tensor(output[26]),
            ),
            (
                torch.tensor(output[27]),
                torch.tensor(output[28]),
            ),
            (
                torch.tensor(output[29]),
                torch.tensor(output[30]),
            ),
            (
                torch.tensor(output[31]),
                torch.tensor(output[32]),
            ),
            (
                torch.tensor(output[33]),
                torch.tensor(output[34]),
            ),
            (
                torch.tensor(output[35]),
                torch.tensor(output[36]),
            ),
            (
                torch.tensor(output[37]),
                torch.tensor(output[38]),
            ),
            (
                torch.tensor(output[39]),
                torch.tensor(output[40]),
            ),
        )
        return result


class TwoWayShardingDecoderLayer(torch.nn.Module):
    def __init__(self, decoder_layer_model, falcon_variant):
        super().__init__()
        self.model = decoder_layer_model
        self.falcon_variant = falcon_variant

    def forward(self, hidden_states, attention_mask):
        new_pkvs = []
        for layer in self.model:
            outputs = layer(
                hidden_states=hidden_states,
                alibi=None,
                attention_mask=attention_mask,
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
            (new_pkv80, new_pkv81),
            (new_pkv90, new_pkv91),
            (new_pkv100, new_pkv101),
            (new_pkv110, new_pkv111),
            (new_pkv120, new_pkv121),
            (new_pkv130, new_pkv131),
            (new_pkv140, new_pkv141),
            (new_pkv150, new_pkv151),
            (new_pkv160, new_pkv161),
            (new_pkv170, new_pkv171),
            (new_pkv180, new_pkv181),
            (new_pkv190, new_pkv191),
            (new_pkv200, new_pkv201),
            (new_pkv210, new_pkv211),
            (new_pkv220, new_pkv221),
            (new_pkv230, new_pkv231),
            (new_pkv240, new_pkv241),
            (new_pkv250, new_pkv251),
            (new_pkv260, new_pkv261),
            (new_pkv270, new_pkv271),
            (new_pkv280, new_pkv281),
            (new_pkv290, new_pkv291),
            (new_pkv300, new_pkv301),
            (new_pkv310, new_pkv311),
            (new_pkv320, new_pkv321),
            (new_pkv330, new_pkv331),
            (new_pkv340, new_pkv341),
            (new_pkv350, new_pkv351),
            (new_pkv360, new_pkv361),
            (new_pkv370, new_pkv371),
            (new_pkv380, new_pkv381),
            (new_pkv390, new_pkv391),
        ) = new_pkvs
        result = (
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
            new_pkv80,
            new_pkv81,
            new_pkv90,
            new_pkv91,
            new_pkv100,
            new_pkv101,
            new_pkv110,
            new_pkv111,
            new_pkv120,
            new_pkv121,
            new_pkv130,
            new_pkv131,
            new_pkv140,
            new_pkv141,
            new_pkv150,
            new_pkv151,
            new_pkv160,
            new_pkv161,
            new_pkv170,
            new_pkv171,
            new_pkv180,
            new_pkv181,
            new_pkv190,
            new_pkv191,
            new_pkv200,
            new_pkv201,
            new_pkv210,
            new_pkv211,
            new_pkv220,
            new_pkv221,
            new_pkv230,
            new_pkv231,
            new_pkv240,
            new_pkv241,
            new_pkv250,
            new_pkv251,
            new_pkv260,
            new_pkv261,
            new_pkv270,
            new_pkv271,
            new_pkv280,
            new_pkv281,
            new_pkv290,
            new_pkv291,
            new_pkv300,
            new_pkv301,
            new_pkv310,
            new_pkv311,
            new_pkv320,
            new_pkv321,
            new_pkv330,
            new_pkv331,
            new_pkv340,
            new_pkv341,
            new_pkv350,
            new_pkv351,
            new_pkv360,
            new_pkv361,
            new_pkv370,
            new_pkv371,
            new_pkv380,
            new_pkv381,
            new_pkv390,
            new_pkv391,
        )
        return result


class CompiledTwoWayShardingDecoderLayer(torch.nn.Module):
    def __init__(
        self, layer_id, device_idx, falcon_variant, device, precision, model
    ):
        super().__init__()
        self.layer_id = layer_id
        self.device_index = device_idx
        self.falcon_variant = falcon_variant
        self.device = device
        self.precision = precision
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        import gc

        torch.cuda.empty_cache()
        gc.collect()

        if self.model is None:
            raise ValueError("Layer vmfb not found")

        hidden_states = hidden_states.to(torch.float32).detach().numpy()
        attention_mask = attention_mask.to(torch.float32).detach().numpy()

        if alibi is not None or layer_past is not None:
            raise ValueError("Past Key Values and alibi should be None")
        else:
            output = self.model(
                "forward",
                (
                    hidden_states,
                    attention_mask,
                ),
            )

        result = (
            torch.tensor(output[0]),
            (
                torch.tensor(output[1]),
                torch.tensor(output[2]),
            ),
            (
                torch.tensor(output[3]),
                torch.tensor(output[4]),
            ),
            (
                torch.tensor(output[5]),
                torch.tensor(output[6]),
            ),
            (
                torch.tensor(output[7]),
                torch.tensor(output[8]),
            ),
            (
                torch.tensor(output[9]),
                torch.tensor(output[10]),
            ),
            (
                torch.tensor(output[11]),
                torch.tensor(output[12]),
            ),
            (
                torch.tensor(output[13]),
                torch.tensor(output[14]),
            ),
            (
                torch.tensor(output[15]),
                torch.tensor(output[16]),
            ),
            (
                torch.tensor(output[17]),
                torch.tensor(output[18]),
            ),
            (
                torch.tensor(output[19]),
                torch.tensor(output[20]),
            ),
            (
                torch.tensor(output[21]),
                torch.tensor(output[22]),
            ),
            (
                torch.tensor(output[23]),
                torch.tensor(output[24]),
            ),
            (
                torch.tensor(output[25]),
                torch.tensor(output[26]),
            ),
            (
                torch.tensor(output[27]),
                torch.tensor(output[28]),
            ),
            (
                torch.tensor(output[29]),
                torch.tensor(output[30]),
            ),
            (
                torch.tensor(output[31]),
                torch.tensor(output[32]),
            ),
            (
                torch.tensor(output[33]),
                torch.tensor(output[34]),
            ),
            (
                torch.tensor(output[35]),
                torch.tensor(output[36]),
            ),
            (
                torch.tensor(output[37]),
                torch.tensor(output[38]),
            ),
            (
                torch.tensor(output[39]),
                torch.tensor(output[40]),
            ),
            (
                torch.tensor(output[41]),
                torch.tensor(output[42]),
            ),
            (
                torch.tensor(output[43]),
                torch.tensor(output[44]),
            ),
            (
                torch.tensor(output[45]),
                torch.tensor(output[46]),
            ),
            (
                torch.tensor(output[47]),
                torch.tensor(output[48]),
            ),
            (
                torch.tensor(output[49]),
                torch.tensor(output[50]),
            ),
            (
                torch.tensor(output[51]),
                torch.tensor(output[52]),
            ),
            (
                torch.tensor(output[53]),
                torch.tensor(output[54]),
            ),
            (
                torch.tensor(output[55]),
                torch.tensor(output[56]),
            ),
            (
                torch.tensor(output[57]),
                torch.tensor(output[58]),
            ),
            (
                torch.tensor(output[59]),
                torch.tensor(output[60]),
            ),
            (
                torch.tensor(output[61]),
                torch.tensor(output[62]),
            ),
            (
                torch.tensor(output[63]),
                torch.tensor(output[64]),
            ),
            (
                torch.tensor(output[65]),
                torch.tensor(output[66]),
            ),
            (
                torch.tensor(output[67]),
                torch.tensor(output[68]),
            ),
            (
                torch.tensor(output[69]),
                torch.tensor(output[70]),
            ),
            (
                torch.tensor(output[71]),
                torch.tensor(output[72]),
            ),
            (
                torch.tensor(output[73]),
                torch.tensor(output[74]),
            ),
            (
                torch.tensor(output[75]),
                torch.tensor(output[76]),
            ),
            (
                torch.tensor(output[77]),
                torch.tensor(output[78]),
            ),
            (
                torch.tensor(output[79]),
                torch.tensor(output[80]),
            ),
        )
        return result


class ShardedFalconModel:
    def __init__(self, model, layers, word_embeddings, ln_f, lm_head):
        super().__init__()
        self.model = model
        self.model.transformer.h = torch.nn.modules.container.ModuleList(
            layers
        )
        self.model.transformer.word_embeddings = word_embeddings
        self.model.transformer.ln_f = ln_f
        self.model.lm_head = lm_head

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits[:, -1, :]
