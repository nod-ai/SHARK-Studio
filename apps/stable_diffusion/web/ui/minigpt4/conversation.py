from PIL import Image

import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Any

# SHARK dependencies
# from shark.shark_inference import SharkInference
# from apps.stable_diffusion.src.utils import (
#     compile_through_fx,
#     args,
# )
import random
import contextlib
import re
from transformers import BertTokenizer
from transformers.generation import GenerationConfig, LogitsProcessorList
import copy
# QFormer, eva_vit, blip_processor, dist_utils
from apps.stable_diffusion.web.ui.minigpt4.models.Qformer import BertConfig, BertLMHeadModel
from apps.stable_diffusion.web.ui.minigpt4.dist_utils import download_cached_file
from apps.stable_diffusion.web.ui.minigpt4.models.eva_vit import create_eva_vit_g

class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url

class MiniGPT4SHARK(torch.nn.Module):
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/minigpt4.yaml",
    }
    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        # enable_autocast = self.device != torch.device("cpu")
        enable_autocast = True

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision
    
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = torch.nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        self.load_state_dict(state_dict, strict=False)
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for _, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            # logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for _, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            # logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print("Llama = ", llama_model)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float32,
            )

        for _, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = torch.nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

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
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
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
            conv_id=self.conv_id)

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

# args.device = "cuda"
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
    return visionModel
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
    return qformerBertModel
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
    def __init__(self, model, vis_processor, device='cpu'):
        self.device = device
        self.model = model
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
        # embs = self.get_context_emb(conv, img_list, max_length - max_new_tokens)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens

        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        #############################################
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
        generation_config.pad_token_id = self.model.llama_tokenizer.pad_token_id
        pad_token_id = generation_config.pad_token_id
        embs_for_pad_token_id = self.model.llama_model.model.embed_tokens(torch.tensor([pad_token_id]))
        model_kwargs["attention_mask"] = torch.logical_not(
                torch.tensor([torch.all(torch.eq(inputs_tensor[:,d,:], embs_for_pad_token_id)).int() for d in range(inputs_tensor.shape[1])]
                            ).unsqueeze(0)
        ).int()
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
        print("Generating tokens now")
        while True:
            # prepare model inputs
            model_inputs = self.model.llama_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model.llama_model.forward(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]

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
            model_kwargs = self.model.llama_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
        output_token = input_ids[0]
        #############################################
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False, skip_special_tokens=True)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        print("Output: ", output_text)
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
        # if self.model.low_resource:
        #     self.model.vit_to_cpu()
        #     image = image.to("cpu")

        with self.model.maybe_autocast():
            shark_visionModel = get_vision_model(self.model.ln_vision, self.model.visual_encoder)
            # image_embeds = shark_visionModel("forward", (image,))
            image_embeds = shark_visionModel.forward(image)
            # cpu_device = torch.device("cpu")
            # image_embeds = torch.from_numpy(image_embeds)
            image_embeds = image_embeds.to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
            shark_QformerBertModel = get_qformer_bert_model(self.model.Qformer.bert)
            # query_output = shark_QformerBertModel("forward", (query_tokens, image_embeds, image_atts,))
            query_output = shark_QformerBertModel.forward(query_tokens, image_embeds, image_atts)
            # query_output = torch.from_numpy(query_output)

            inputs_llama = self.model.llama_proj(query_output)
            # atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        image_emb = inputs_llama
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    """
    def get_context_emb(self, conv, img_list, max_allowed_tokens=200):
        self.model.llama_tokenizer.padding_side = "left"
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
        self.model.llama_model.config.pad_token_id = self.model.llama_tokenizer.pad_token_id
        
        seg_embs_pre = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens_pre]
        mixed_embs_pre = [emb.to('cpu') for pair in zip(seg_embs_pre, img_list) for emb in pair]
        mixed_embs_pre = torch.cat(mixed_embs_pre, dim=1)
        max_allowed_tokens = max_allowed_tokens - mixed_embs_pre.shape[1]
        final_prompt = prompt_segs[-1]
        seg_tokens_post = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", padding="max_length", max_length=max_allowed_tokens, add_special_tokens=False).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate([final_prompt])
        ]
        seg_tokens_post = seg_tokens_post[0]
        seg_embs_post = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens_post]
        mixed_embs_post = [seg_embs_post[0].to('cpu')]
        mixed_embs_post = torch.unsqueeze(mixed_embs_post[0], 0)
        mixed_embs = [mixed_embs_pre] + [mixed_embs_post]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        
        return mixed_embs
    #"""
    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        self.model.llama_tokenizer.padding_side = "right"
        #import pdb
        #pdb.set_trace()
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
