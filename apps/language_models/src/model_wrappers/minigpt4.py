import torch
import dataclasses
from enum import auto, Enum
from typing import List, Any
from transformers import StoppingCriteria, StoppingCriteriaList


class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VisionModel(torch.nn.Module):
    def __init__(self, ln_vision, visual_encoder):
        super().__init__()
        self.ln_vision = ln_vision
        self.visual_encoder = visual_encoder

    def forward(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        return image_embeds


class QformerBertModel(torch.nn.Module):
    def __init__(self, qformer_bert):
        super().__init__()
        self.qformer_bert = qformer_bert

    def forward(self, query_tokens, image_embeds, image_atts):
        query_output = self.qformer_bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state


class FirstLlamaModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        print("SHARK: Loading LLAMA Done")

    def forward(self, inputs_embeds, position_ids, attention_mask):
        print("************************************")
        print(
            "inputs_embeds: ",
            inputs_embeds.shape,
            " dtype: ",
            inputs_embeds.dtype,
        )
        print(
            "position_ids: ",
            position_ids.shape,
            " dtype: ",
            position_ids.dtype,
        )
        print(
            "attention_mask: ",
            attention_mask.shape,
            " dtype: ",
            attention_mask.dtype,
        )
        print("************************************")
        config = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "past_key_values": None,
            "use_cache": True,
            "attention_mask": attention_mask,
        }
        output = self.model(
            **config,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return_vals = []
        return_vals.append(output.logits)
        temp_past_key_values = output.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SecondLlamaModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        print("SHARK: Loading LLAMA Done")

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        i12,
        i13,
        i14,
        i15,
        i16,
        i17,
        i18,
        i19,
        i20,
        i21,
        i22,
        i23,
        i24,
        i25,
        i26,
        i27,
        i28,
        i29,
        i30,
        i31,
        i32,
        i33,
        i34,
        i35,
        i36,
        i37,
        i38,
        i39,
        i40,
        i41,
        i42,
        i43,
        i44,
        i45,
        i46,
        i47,
        i48,
        i49,
        i50,
        i51,
        i52,
        i53,
        i54,
        i55,
        i56,
        i57,
        i58,
        i59,
        i60,
        i61,
        i62,
        i63,
        i64,
    ):
        print("************************************")
        print("input_ids: ", input_ids.shape, " dtype: ", input_ids.dtype)
        print(
            "position_ids: ",
            position_ids.shape,
            " dtype: ",
            position_ids.dtype,
        )
        print(
            "attention_mask: ",
            attention_mask.shape,
            " dtype: ",
            attention_mask.dtype,
        )
        print("past_key_values: ", i1.shape, i2.shape, i63.shape, i64.shape)
        print("past_key_values dtype: ", i1.dtype)
        print("************************************")
        config = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": (
                (i1, i2),
                (
                    i3,
                    i4,
                ),
                (
                    i5,
                    i6,
                ),
                (
                    i7,
                    i8,
                ),
                (
                    i9,
                    i10,
                ),
                (
                    i11,
                    i12,
                ),
                (
                    i13,
                    i14,
                ),
                (
                    i15,
                    i16,
                ),
                (
                    i17,
                    i18,
                ),
                (
                    i19,
                    i20,
                ),
                (
                    i21,
                    i22,
                ),
                (
                    i23,
                    i24,
                ),
                (
                    i25,
                    i26,
                ),
                (
                    i27,
                    i28,
                ),
                (
                    i29,
                    i30,
                ),
                (
                    i31,
                    i32,
                ),
                (
                    i33,
                    i34,
                ),
                (
                    i35,
                    i36,
                ),
                (
                    i37,
                    i38,
                ),
                (
                    i39,
                    i40,
                ),
                (
                    i41,
                    i42,
                ),
                (
                    i43,
                    i44,
                ),
                (
                    i45,
                    i46,
                ),
                (
                    i47,
                    i48,
                ),
                (
                    i49,
                    i50,
                ),
                (
                    i51,
                    i52,
                ),
                (
                    i53,
                    i54,
                ),
                (
                    i55,
                    i56,
                ),
                (
                    i57,
                    i58,
                ),
                (
                    i59,
                    i60,
                ),
                (
                    i61,
                    i62,
                ),
                (
                    i63,
                    i64,
                ),
            ),
            "use_cache": True,
            "attention_mask": attention_mask,
        }
        output = self.model(
            **config,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return_vals = []
        return_vals.append(output.logits)
        temp_past_key_values = output.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
    "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
