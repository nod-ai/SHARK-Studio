import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from apps.stable_diffusion.web.ui.MiniGPT4.minigpt4.common.registry import registry

# SHARK dependencies
from shark.shark_inference import SharkInference
from apps.stable_diffusion.src.utils import (
    compile_through_fx,
    args,
)
import os

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
    # system_img: List[Image.Image] = []
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
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
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
            if torch.all((stop == input_ids[0][-len(stop):])).item():
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

args.device = "cuda"
def get_vision_model(ln_vision, visual_encoder):
    class VisionModel(torch.nn.Module):
        def __init__(self, ln_vision, visual_encoder):
            super().__init__()
            self.ln_vision = ln_vision
            self.visual_encoder = visual_encoder
        def forward(self, image):
            image_embeds = self.ln_vision(self.visual_encoder(image))
            return image_embeds
    visionModel = VisionModel(ln_vision, visual_encoder)
    vmfb_path = "visionModel_fp32_cuda.vmfb"
    if os.path.isfile(vmfb_path):
        shark_module = SharkInference(
            None,
            device=args.device,
            mlir_dialect="tm_tensor",
        )
        print(f"loading existing vmfb from: {vmfb_path}")
        shark_module.load_module(vmfb_path, extra_args=[])
        return shark_module, None
    print("Compiling visionModel_fp32_cuda")
    shark_visionModel, visionModel_mlir = compile_through_fx(
        visionModel,
        [torch.randint(3, (1, 3, 224, 224), dtype=torch.float32)],
        extended_model_name="visionModel_fp32_cuda",
        debug=False,
        generate_vmfb=True,
        save_dir=os.getcwd(),
        extra_args=[],
        base_model_id=None,
        model_name="visionModel_fp32_cuda",
        precision=None,
        return_mlir=False,
    )
    print("Generated visionModel_fp32_cuda.vmfb")
    return shark_visionModel, visionModel_mlir

def get_qformer_bert_model(qformer_bert):
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
    qformerBertModel = QformerBertModel(qformer_bert)
    vmfb_path = "qformerBertModel_fp32_cuda.vmfb"
    if os.path.isfile(vmfb_path):
        shark_module = SharkInference(
            None,
            device=args.device,
            mlir_dialect="tm_tensor",
        )
        print(f"loading existing vmfb from: {vmfb_path}")
        shark_module.load_module(vmfb_path, extra_args=[])
        return shark_module, None
    print("Compiling qformerBertModel_fp32_cuda")
    shark_QformerBertModel, qformerBertModel_mlir = compile_through_fx(
        qformerBertModel,
        [torch.randint(3, (1, 32, 768), dtype=torch.float32),
         torch.randint(3, (1, 257, 1408), dtype=torch.float32),
         torch.randint(3, (1, 257), dtype=torch.int64)],
        extended_model_name="qformerBertModel_fp32_cuda",
        debug=False,
        generate_vmfb=True,
        save_dir=os.getcwd(),
        extra_args=[],
        base_model_id=None,
        model_name="qformerBertModel_fp32_cuda",
        precision=None,
        return_mlir=False,
    )
    print("Generated qformerBertModel_fp32_cuda.vmfb")
    return shark_QformerBertModel, qformerBertModel_mlir

class Chat:
    def __init__(self, model, first_llama_model, second_llama_model, vis_processor, device='cpu'):
        self.device = device
        self.model = model
        self.first_llama_model_shark = first_llama_model
        self.second_llama_model_shark = second_llama_model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]),
                          torch.tensor([2277, 29937])]
        #stop_words_ids = [torch.tensor([835]).to(self.device),
        #                  torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        #embs = self.get_context_emb(conv, img_list, max_length - max_new_tokens)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        #####################################################
        # Before entering `forward` function.
        #####################################################
        # BEGIN
        #slice_tensor = torch.empty([1], dtype=torch.int64)
        #slice_tensor[0] = embs.shape[1]
        #amount_to_pad = max_length - current_max_len
        #if amount_to_pad > 0:
        #    embs = torch.nn.functional.pad(embs, (0,0,0,amount_to_pad), "constant", 32000)
        # END

        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        import pdb
        pdb.set_trace()
        #############################################
        from transformers.generation import GenerationConfig, LogitsProcessorList
        import copy
        generation_config = GenerationConfig.from_model_config(self.model.llama_model.config)
        kwargs = {
            "inputs_embeds":embs,
            "max_new_tokens":max_new_tokens,
            "num_beams":num_beams,
            "do_sample":True,
            "min_length":min_length,
            "top_p":top_p,
            "repetition_penalty":repetition_penalty,
            "length_penalty":length_penalty,
            "temperature":temperature,
        }
        generation_config = copy.deepcopy(generation_config)
        # Need to include kwargs -> which is what self.model.llama_model.generate's argument is except
        #                           `stopping_criteria`
        model_kwargs = generation_config.update(**kwargs)
        logits_processor = LogitsProcessorList()
        stopping_criteria = self.stopping_criteria
        inputs = None
        inputs_tensor, model_input_name, model_kwargs = self.model.llama_model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["attention_mask"] = self.model.llama_model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        input_ids_seq_length = input_ids.shape[-1]
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        logits_warper = self.model.llama_model._get_logits_warper(generation_config)
        input_ids, model_kwargs = self.model.llama_model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=False,
            **model_kwargs,
        )
        # DOUBT: stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = generation_config.output_scores
        output_attentions = (
            generation_config.output_attentions
        )
        output_hidden_states = (
            generation_config.output_hidden_states
        )
        scores = None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        i = 0
        while True:
            # prepare model inputs
            pdb.set_trace()
            model_inputs = self.model.llama_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            if i == 0:
                outputs = self.first_llama_model_shark(
                    model_inputs['inputs_embeds'],
                    model_inputs['position_ids'],
                    model_inputs['attention_mask'],
                )
            else:
                outputs = self.second_llama_model_shark(
                    model_inputs['input_ids'],
                    model_inputs['position_ids'],
                    model_inputs['attention_mask'],
                    model_inputs['past_key_values'],
                )

            next_token_logits = outputs[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            import pdb
            pdb.set_trace()
            model_kwargs = self.model.llama_model._update_model_kwargs_for_generation(
                {}, model_kwargs, is_encoder_decoder=False
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            
            i = i + 1
        output_token = input_ids[0]
        #############################################
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False, skip_special_tokens=True)
        import pdb
        pdb.set_trace()
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        device = image.device
        if self.model.low_resource:
            self.model.vit_to_cpu()
            image = image.to("cpu")

        with self.model.maybe_autocast():
            shark_visionModel, _ = get_vision_model(self.model.ln_vision, self.model.visual_encoder)
            image_embeds = shark_visionModel("forward", (image,))
            cpu_device = torch.device("cpu")
            image_embeds = torch.from_numpy(image_embeds)
            image_embeds = image_embeds.to(device)
            #image_embeds = image_embeds.to(cpu_device)
            #image_embeds = self.model.ln_vision(self.model.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            #image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(cpu_device)

            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
            #query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1).to(cpu_device)
            #del shark_visionModel
            #import pdb
            #pdb.set_trace()
            shark_QformerBertModel, _ = get_qformer_bert_model(self.model.Qformer.bert)
            #query_output = shark_QformerBertModel("forward", (query_tokens.to("cpu"), image_embeds.to("cpu"), image_atts.to("cpu"),))
            #import pdb
            #pdb.set_trace()
            query_output = shark_QformerBertModel("forward", (query_tokens, image_embeds, image_atts,))
            query_output = torch.from_numpy(query_output)

            inputs_llama = self.model.llama_proj(query_output)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        image_emb = inputs_llama
        #image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    """
    def get_context_emb(self, conv, img_list, max_allowed_tokens=1700):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        prompt_segs_pre = prompt_segs[:-1]
        seg_tokens_pre = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs_pre)
        ]
        import pdb
        pdb.set_trace()
        self.model.llama_model.config.pad_token_id = self.model.llama_tokenizer.pad_token_id
        #self.model.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        seg_embs_pre = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens_pre]
        mixed_embs_pre = [emb.to('cpu') for pair in zip(seg_embs_pre, img_list) for emb in pair]
        # + [seg_embs[-1].to('cpu')]
        mixed_embs_pre = torch.cat(mixed_embs_pre, dim=1)
        #pdb.set_trace()
        
        #seg_tokens_post = [
        #    self.model.llama_tokenizer(
        #        seg, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device).input_ids
            # only add bos to the first seg
        #    for i, seg in enumerate([prompt_segs[-1]])
        #]
        max_allowed_tokens = max_allowed_tokens - mixed_embs_pre.shape[1]
        #max_allowed_tokens = max_allowed_tokens - mixed_embs_pre.shape[1] - seg_tokens_post[0].shape[1]
        #dummy_pad = ". "*max_allowed_tokens
        #final_prompt = prompt_segs[-1][:-13] + dummy_pad + prompt_segs[-1][-13:]
        final_prompt = prompt_segs[-1]
        #pdb.set_trace()
        seg_tokens_post = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", padding="max_length", max_length=max_allowed_tokens, add_special_tokens=False).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate([final_prompt])
        ]

        #pdb.set_trace()
        seg_tokens_post = seg_tokens_post[0]
        seg_embs_post = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens_post]
        mixed_embs_post = [seg_embs_post[0].to('cpu')]
        # + [seg_embs[-1].to('cpu')]
        #pdb.set_trace()

        mixed_embs = [mixed_embs_pre] + [torch.unsqueeze(mixed_embs_post[0], 0)]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        #pdb.set_trace()
        
        return mixed_embs
    """
    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        #self.model.llama_model.model.to('cuda')
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        #self.model.llama_model.model.to('cpu')
        mixed_embs = [emb.to('cpu') for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1].to('cpu')]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
    #"""
