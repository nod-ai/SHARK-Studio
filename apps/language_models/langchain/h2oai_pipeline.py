import os
from apps.stable_diffusion.src.utils.utils import _compile_module

from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType

from stopping import get_stopping
from prompter import Prompter, PromptType


from transformers.generation import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)
import copy
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import gc
from pathlib import Path
from shark.shark_inference import SharkInference
from shark.shark_downloader import download_public_file
from apps.stable_diffusion.src import args

global_device = "cuda"
global_precision = "fp16"

if not args.run_docuchat_web:
    args.device = global_device
    args.precision = global_precision
tensor_device = "cpu" if args.device == "cpu" else "cuda"


class H2OGPTSHARKModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "h2ogpt_falcon_7b"
        extended_model_name = (
            model_name + "_" + args.precision + "_" + args.device
        )
        vmfb_path = Path(extended_model_name + ".vmfb")
        mlir_path = Path(model_name + "_" + args.precision + ".mlir")
        shark_module = None

        if not vmfb_path.exists():
            if args.device in ["cuda", "cpu"] and args.precision in [
                "fp16",
                "fp32",
            ]:
                # Downloading VMFB from shark_tank
                print("Downloading vmfb from shark tank.")
                download_public_file(
                    "gs://shark_tank/langchain/" + str(vmfb_path),
                    vmfb_path.absolute(),
                    single_file=True,
                )
            else:
                if mlir_path.exists():
                    with open(mlir_path, "rb") as f:
                        bytecode = f.read()
                else:
                    # Downloading MLIR from shark_tank
                    download_public_file(
                        "gs://shark_tank/langchain/" + str(mlir_path),
                        mlir_path.absolute(),
                        single_file=True,
                    )
                    if mlir_path.exists():
                        with open(mlir_path, "rb") as f:
                            bytecode = f.read()
                    else:
                        raise ValueError(
                            f"MLIR not found at {mlir_path.absolute()}"
                            " after downloading! Please check path and try again"
                        )
                shark_module = SharkInference(
                    mlir_module=bytecode,
                    device=args.device,
                    mlir_dialect="linalg",
                )
                print(f"[DEBUG] generating vmfb.")
                shark_module = _compile_module(
                    shark_module, extended_model_name, []
                )
                print("Saved newly generated vmfb.")

        if shark_module is None:
            if vmfb_path.exists():
                print("Compiled vmfb found. Loading it from: ", vmfb_path)
                shark_module = SharkInference(
                    None, device=args.device, mlir_dialect="linalg"
                )
                shark_module.load_module(str(vmfb_path))
                print("Compiled vmfb loaded successfully.")
            else:
                raise ValueError("Unable to download/generate a vmfb.")

        self.model = shark_module

    def forward(self, input_ids, attention_mask):
        result = torch.from_numpy(
            self.model(
                "forward",
                (input_ids.to(device="cpu"), attention_mask.to(device="cpu")),
            )
        ).to(device=tensor_device)
        return result


h2ogpt_model = H2OGPTSHARKModel()


def pad_or_truncate_inputs(
    input_ids, attention_mask, max_padding_length=400, do_truncation=False
):
    inp_shape = input_ids.shape
    if inp_shape[1] < max_padding_length:
        # do padding
        num_add_token = max_padding_length - inp_shape[1]
        padded_input_ids = torch.cat(
            [
                torch.tensor([[11] * num_add_token]).to(device=tensor_device),
                input_ids,
            ],
            dim=1,
        )
        padded_attention_mask = torch.cat(
            [
                torch.tensor([[0] * num_add_token]).to(device=tensor_device),
                attention_mask,
            ],
            dim=1,
        )
        return padded_input_ids, padded_attention_mask
    elif inp_shape[1] > max_padding_length or do_truncation:
        # do truncation
        num_remove_token = inp_shape[1] - max_padding_length
        truncated_input_ids = input_ids[:, num_remove_token:]
        truncated_attention_mask = attention_mask[:, num_remove_token:]
        return truncated_input_ids, truncated_attention_mask
    else:
        return input_ids, attention_mask


class H2OTextGenerationPipeline(TextGenerationPipeline):
    def __init__(
        self,
        *args,
        debug=False,
        chat=False,
        stream_output=False,
        sanitize_bot_response=False,
        use_prompter=True,
        prompter=None,
        prompt_type=None,
        prompt_dict=None,
        max_input_tokens=2048 - 256,
        **kwargs,
    ):
        """
        HF-like pipeline, but handle instruction prompting and stopping (for some models)
        :param args:
        :param debug:
        :param chat:
        :param stream_output:
        :param sanitize_bot_response:
        :param use_prompter: Whether to use prompter.  If pass prompt_type, will make prompter
        :param prompter: prompter, can pass if have already
        :param prompt_type: prompt_type, e.g. human_bot.  See prompt_type to model mapping in from prompter.py.
                            If use_prompter, then will make prompter and use it.
        :param prompt_dict: dict of get_prompt(, return_dict=True) for prompt_type=custom
        :param max_input_tokens:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.prompt_text = None
        self.use_prompter = use_prompter
        self.prompt_type = prompt_type
        self.prompt_dict = prompt_dict
        self.prompter = prompter
        if self.use_prompter:
            if self.prompter is not None:
                assert self.prompter.prompt_type is not None
            else:
                self.prompter = Prompter(
                    self.prompt_type,
                    self.prompt_dict,
                    debug=debug,
                    chat=chat,
                    stream_output=stream_output,
                )
            self.human = self.prompter.humanstr
            self.bot = self.prompter.botstr
            self.can_stop = True
        else:
            self.prompter = None
            self.human = None
            self.bot = None
            self.can_stop = False
        self.sanitize_bot_response = sanitize_bot_response
        self.max_input_tokens = (
            max_input_tokens  # not for generate, so ok that not kwargs
        )

    @staticmethod
    def limit_prompt(prompt_text, tokenizer, max_prompt_length=None):
        verbose = bool(int(os.getenv("VERBOSE_PIPELINE", "0")))

        if hasattr(tokenizer, "model_max_length"):
            # model_max_length only defined for generate.py, not raw use of h2oai_pipeline.py
            model_max_length = tokenizer.model_max_length
            if max_prompt_length is not None:
                model_max_length = min(model_max_length, max_prompt_length)
            # cut at some upper likely limit to avoid excessive tokenization etc
            # upper bound of 10 chars/token, e.g. special chars sometimes are long
            if len(prompt_text) > model_max_length * 10:
                len0 = len(prompt_text)
                prompt_text = prompt_text[-model_max_length * 10 :]
                if verbose:
                    print(
                        "Cut of input: %s -> %s" % (len0, len(prompt_text)),
                        flush=True,
                    )
        else:
            # unknown
            model_max_length = None

        num_prompt_tokens = None
        if model_max_length is not None:
            # can't wait for "hole" if not plain prompt_type, since would lose prefix like <human>:
            # For https://github.com/h2oai/h2ogpt/issues/192
            for trial in range(0, 3):
                prompt_tokens = tokenizer(prompt_text)["input_ids"]
                num_prompt_tokens = len(prompt_tokens)
                if num_prompt_tokens > model_max_length:
                    # conservative by using int()
                    chars_per_token = int(len(prompt_text) / num_prompt_tokens)
                    # keep tail, where question is if using langchain
                    prompt_text = prompt_text[
                        -model_max_length * chars_per_token :
                    ]
                    if verbose:
                        print(
                            "reducing %s tokens, assuming average of %s chars/token for %s characters"
                            % (
                                num_prompt_tokens,
                                chars_per_token,
                                len(prompt_text),
                            ),
                            flush=True,
                        )
                else:
                    if verbose:
                        print(
                            "using %s tokens with %s chars"
                            % (num_prompt_tokens, len(prompt_text)),
                            flush=True,
                        )
                    break

        return prompt_text, num_prompt_tokens

    def preprocess(
        self,
        prompt_text,
        prefix="",
        handle_long_generation=None,
        **generate_kwargs,
    ):
        (
            prompt_text,
            num_prompt_tokens,
        ) = H2OTextGenerationPipeline.limit_prompt(prompt_text, self.tokenizer)

        data_point = dict(context="", instruction=prompt_text, input="")
        if self.prompter is not None:
            prompt_text = self.prompter.generate_prompt(data_point)
        self.prompt_text = prompt_text
        if handle_long_generation is None:
            # forces truncation of inputs to avoid critical failure
            handle_long_generation = None  # disable with new approaches
        return super().preprocess(
            prompt_text,
            prefix=prefix,
            handle_long_generation=handle_long_generation,
            **generate_kwargs,
        )

    def postprocess(
        self,
        model_outputs,
        return_type=ReturnType.FULL_TEXT,
        clean_up_tokenization_spaces=True,
    ):
        records = super().postprocess(
            model_outputs,
            return_type=return_type,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for rec in records:
            if self.use_prompter:
                outputs = rec["generated_text"]
                outputs = self.prompter.get_response(
                    outputs,
                    prompt=self.prompt_text,
                    sanitize_bot_response=self.sanitize_bot_response,
                )
            elif self.bot and self.human:
                outputs = (
                    rec["generated_text"]
                    .split(self.bot)[1]
                    .split(self.human)[0]
                )
            else:
                outputs = rec["generated_text"]
            rec["generated_text"] = outputs
            print(
                "prompt: %s\noutputs: %s\n\n" % (self.prompt_text, outputs),
                flush=True,
            )
        return records

    def generate_new_token(self):
        model_inputs = self.model.prepare_inputs_for_generation(
            self.input_ids, **self.model_kwargs
        )

        outputs = h2ogpt_model.forward(
            model_inputs["input_ids"], model_inputs["attention_mask"]
        )

        if args.precision == "fp16":
            outputs = outputs.to(dtype=torch.float32)
        next_token_logits = outputs

        # pre-process distribution
        next_token_scores = self.logits_processor(
            self.input_ids, next_token_logits
        )
        next_token_scores = self.logits_warper(
            self.input_ids, next_token_scores
        )

        # sample
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if self.eos_token_id is not None:
            if self.pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_token = (
                next_token * self.unfinished_sequences
                + self.pad_token_id * (1 - self.unfinished_sequences)
            )

        self.input_ids = torch.cat(
            [self.input_ids, next_token[:, None]], dim=-1
        )

        self.model_kwargs["past_key_values"] = None
        if "attention_mask" in self.model_kwargs:
            attention_mask = self.model_kwargs["attention_mask"]
            self.model_kwargs["attention_mask"] = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ],
                dim=-1,
            )

        self.truncated_input_ids.append(self.input_ids[:, 0])
        self.input_ids = self.input_ids[:, 1:]
        self.model_kwargs["attention_mask"] = self.model_kwargs[
            "attention_mask"
        ][:, 1:]

        return next_token

    def generate_token(self, **generate_kwargs):
        self.truncated_input_ids = []

        generation_config_ = GenerationConfig.from_model_config(
            self.model.config
        )
        generation_config = copy.deepcopy(generation_config_)
        self.model_kwargs = generation_config.update(**generate_kwargs)

        logits_processor = LogitsProcessorList()
        self.stopping_criteria = (
            self.stopping_criteria
            if self.stopping_criteria is not None
            else StoppingCriteriaList()
        )

        eos_token_id = generation_config.eos_token_id
        generation_config.pad_token_id = eos_token_id

        (
            inputs_tensor,
            model_input_name,
            self.model_kwargs,
        ) = self.model._prepare_model_inputs(
            None, generation_config.bos_token_id, self.model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        self.model_kwargs[
            "output_attentions"
        ] = generation_config.output_attentions
        self.model_kwargs[
            "output_hidden_states"
        ] = generation_config.output_hidden_states
        self.model_kwargs["use_cache"] = generation_config.use_cache

        self.input_ids = (
            inputs_tensor
            if model_input_name == "input_ids"
            else self.model_kwargs.pop("input_ids")
        )

        input_ids_seq_length = self.input_ids.shape[-1]

        generation_config.max_length = (
            generation_config.max_new_tokens + input_ids_seq_length
        )

        self.logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        self.stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=self.stopping_criteria,
        )

        self.logits_warper = self.model._get_logits_warper(generation_config)

        (
            self.input_ids,
            self.model_kwargs,
        ) = self.model._expand_inputs_for_generation(
            input_ids=self.input_ids,
            expand_size=generation_config.num_return_sequences,  # 1
            is_encoder_decoder=self.model.config.is_encoder_decoder,  # False
            **self.model_kwargs,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(device=tensor_device)
            if eos_token_id is not None
            else None
        )

        self.pad_token_id = generation_config.pad_token_id
        self.eos_token_id = eos_token_id

        output_scores = generation_config.output_scores  # False
        output_attentions = generation_config.output_attentions  # False
        output_hidden_states = generation_config.output_hidden_states  # False
        return_dict_in_generate = (
            generation_config.return_dict_in_generate  # False
        )

        # init attention / hidden states / scores tuples
        self.scores = (
            () if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # keep track of which sequences are already finished
        self.unfinished_sequences = torch.ones(
            self.input_ids.shape[0],
            dtype=torch.long,
            device=self.input_ids.device,
        )

        timesRan = 0
        import time

        start = time.time()
        print("\n")

        while True:
            next_token = self.generate_new_token()
            new_word = self.tokenizer.decode(
                next_token.cpu().numpy(),
                add_special_tokens=False,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            print(f"{new_word}", end="", flush=True)

            # if eos_token was found in one sentence, set sentence to finished
            if self.eos_token_id_tensor is not None:
                self.unfinished_sequences = self.unfinished_sequences.mul(
                    next_token.tile(self.eos_token_id_tensor.shape[0], 1)
                    .ne(self.eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
                # stop when each sentence is finished
                if (
                    self.unfinished_sequences.max() == 0
                    or self.stopping_criteria(self.input_ids, self.scores)
                ):
                    break
            timesRan = timesRan + 1

        end = time.time()
        print(
            "\n\nTime taken is {:.2f} seconds/token\n".format(
                (end - start) / timesRan
            )
        )

        self.input_ids = torch.cat(
            [
                torch.tensor(self.truncated_input_ids)
                .to(device=tensor_device)
                .unsqueeze(dim=0),
                self.input_ids,
            ],
            dim=-1,
        )

        torch.cuda.empty_cache()
        gc.collect()

        return self.input_ids

    def _forward(self, model_inputs, **generate_kwargs):
        if self.can_stop:
            stopping_criteria = get_stopping(
                self.prompt_type,
                self.prompt_dict,
                self.tokenizer,
                self.device,
                human=self.human,
                bot=self.bot,
                model_max_length=self.tokenizer.model_max_length,
            )
            generate_kwargs["stopping_criteria"] = stopping_criteria
        # return super()._forward(model_inputs, **generate_kwargs)
        return self.__forward(model_inputs, **generate_kwargs)

    # FIXME: Copy-paste of original _forward, but removed copy.deepcopy()
    # FIXME: https://github.com/h2oai/h2ogpt/issues/172
    def __forward(self, model_inputs, **generate_kwargs):
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)
        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]
        prompt_text = model_inputs.pop("prompt_text")

        ## If there is a prefix, we may need to adjust the generation length. Do so without permanently modifying
        ## generate_kwargs, as some of the parameterization may come from the initialization of the pipeline.
        # generate_kwargs = copy.deepcopy(generate_kwargs)
        prefix_length = generate_kwargs.pop("prefix_length", 0)
        if prefix_length > 0:
            has_max_new_tokens = "max_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].max_new_tokens
                is not None
            )
            if not has_max_new_tokens:
                generate_kwargs["max_length"] = (
                    generate_kwargs.get("max_length")
                    or self.model.config.max_length
                )
                generate_kwargs["max_length"] += prefix_length
            has_min_new_tokens = "min_new_tokens" in generate_kwargs or (
                "generation_config" in generate_kwargs
                and generate_kwargs["generation_config"].min_new_tokens
                is not None
            )
            if not has_min_new_tokens and "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length

        # BS x SL
        # pad or truncate the input_ids and attention_mask
        max_padding_length = 400
        input_ids, attention_mask = pad_or_truncate_inputs(
            input_ids, attention_mask, max_padding_length=max_padding_length
        )
        self.stopping_criteria = generate_kwargs["stopping_criteria"]

        generated_sequence = self.generate_token(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        out_b = generated_sequence.shape[0]
        generated_sequence = generated_sequence.reshape(
            in_b, out_b // in_b, *generated_sequence.shape[1:]
        )
        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "prompt_text": prompt_text,
        }
