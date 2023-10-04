import os
from apps.stable_diffusion.src.utils.utils import _compile_module
from io import BytesIO
import torch_mlir

from stopping import get_stopping
from prompter import Prompter, PromptType

from transformers import TextGenerationPipeline
from transformers.pipelines.text_generation import ReturnType
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
from shark.shark_importer import import_with_fx, save_mlir
from apps.stable_diffusion.src import args

# Brevitas
from typing import List, Tuple
from brevitas_examples.llm.llm_quant.quantize import quantize_model
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl


# fmt: off
def quant〇matmul_rhs_group_quant〡shape(lhs: List[int], rhs: List[int], rhs_scale: List[int], rhs_zero_point: List[int], rhs_bit_width: int, rhs_group_size: int) -> List[int]:
    if len(lhs) == 3 and len(rhs) == 2:
        return [lhs[0], lhs[1], rhs[0]]
    elif len(lhs) == 2 and len(rhs) == 2:
        return [lhs[0], rhs[0]]
    else:
        raise ValueError("Input shapes not supported.")


def quant〇matmul_rhs_group_quant〡dtype(lhs_rank_dtype: Tuple[int, int], rhs_rank_dtype: Tuple[int, int], rhs_scale_rank_dtype: Tuple[int, int], rhs_zero_point_rank_dtype: Tuple[int, int], rhs_bit_width: int, rhs_group_size: int) -> int:
    # output dtype is the dtype of the lhs float input
    lhs_rank, lhs_dtype = lhs_rank_dtype
    return lhs_dtype


def quant〇matmul_rhs_group_quant〡has_value_semantics(lhs, rhs, rhs_scale, rhs_zero_point, rhs_bit_width, rhs_group_size) -> None:
    return


brevitas_matmul_rhs_group_quant_library = [
    quant〇matmul_rhs_group_quant〡shape,
    quant〇matmul_rhs_group_quant〡dtype,
    quant〇matmul_rhs_group_quant〡has_value_semantics]
# fmt: on

global_device = "cuda"
global_precision = "fp16"

if not args.run_docuchat_web:
    args.device = global_device
    args.precision = global_precision
tensor_device = "cpu" if args.device == "cpu" else "cuda"


class H2OGPTModel(torch.nn.Module):
    def __init__(self, device, precision):
        super().__init__()
        torch_dtype = (
            torch.float32
            if precision == "fp32" or device == "cpu"
            else torch.float16
        )
        device_map = {"": "cpu"} if device == "cpu" else {"": 0}
        model_kwargs = {
            "local_files_only": False,
            "torch_dtype": torch_dtype,
            "resume_download": True,
            "use_auth_token": False,
            "trust_remote_code": True,
            "offload_folder": "offline_folder",
            "device_map": device_map,
        }
        config = AutoConfig.from_pretrained(
            "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
            use_auth_token=False,
            trust_remote_code=True,
            offload_folder="offline_folder",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
            config=config,
            **model_kwargs,
        )
        if precision in ["int4", "int8"]:
            print("Applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                self.model.transformer.h,
                dtype=torch.float32,
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float",
                weight_quant_type="asym",
                weight_quant_granularity="per_group",
                weight_group_size=128,
                quantize_weight_zero_point=False,
            )
            print("Weight quantization applied.")

    def forward(self, input_ids, attention_mask):
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "use_cache": True,
        }
        output = self.model(
            **input_dict,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return output.logits[:, -1, :]


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

        need_to_compile = False
        if not vmfb_path.exists():
            need_to_compile = True
            # Downloading VMFB from shark_tank
            print("Trying to download pre-compiled vmfb from shark tank.")
            download_public_file(
                "gs://shark_tank/langchain/" + str(vmfb_path),
                vmfb_path.absolute(),
                single_file=True,
            )
            if vmfb_path.exists():
                print(
                    "Pre-compiled vmfb downloaded from shark tank successfully."
                )
                need_to_compile = False

        if need_to_compile:
            if not mlir_path.exists():
                print("Trying to download pre-generated mlir from shark tank.")
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
                # Generating the mlir
                bytecode = self.get_bytecode(tensor_device, args.precision)

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

    def get_bytecode(self, device, precision):
        h2ogpt_model = H2OGPTModel(device, precision)

        compilation_input_ids = torch.randint(
            low=1, high=10000, size=(1, 400)
        ).to(device=device)
        compilation_attention_mask = torch.ones(1, 400, dtype=torch.int64).to(
            device=device
        )

        h2ogptCompileInput = (
            compilation_input_ids,
            compilation_attention_mask,
        )

        print(f"[DEBUG] generating torchscript graph")
        ts_graph = import_with_fx(
            h2ogpt_model,
            h2ogptCompileInput,
            is_f16=False,
            precision=precision,
            f16_input_mask=[False, False],
            mlir_type="torchscript",
        )
        del h2ogpt_model
        del self.src_model

        print(f"[DEBUG] generating torch mlir")
        if precision in ["int4", "int8"]:
            from torch_mlir.compiler_utils import (
                run_pipeline_with_repro_report,
            )

            module = torch_mlir.compile(
                ts_graph,
                [*h2ogptCompileInput],
                output_type=torch_mlir.OutputType.TORCH,
                backend_legal_ops=["quant.matmul_rhs_group_quant"],
                extra_library=brevitas_matmul_rhs_group_quant_library,
                use_tracing=False,
                verbose=False,
            )
            print(f"[DEBUG] converting torch to linalg")
            run_pipeline_with_repro_report(
                module,
                "builtin.module(func.func(torch-unpack-quant-tensor),func.func(torch-convert-custom-quant-op),torch-backend-to-linalg-on-tensors-backend-pipeline)",
                description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
            )
        else:
            module = torch_mlir.compile(
                ts_graph,
                [*h2ogptCompileInput],
                torch_mlir.OutputType.LINALG_ON_TENSORS,
                use_tracing=False,
                verbose=False,
            )
        del ts_graph

        print(f"[DEBUG] converting to bytecode")
        bytecode_stream = BytesIO()
        module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()
        del module

        bytecode = save_mlir(
            bytecode,
            model_name=f"h2ogpt_{precision}",
            frontend="torch",
        )
        return bytecode

    def forward(self, input_ids, attention_mask):
        result = torch.from_numpy(
            self.model(
                "forward",
                (input_ids.to(device="cpu"), attention_mask.to(device="cpu")),
            )
        ).to(device=tensor_device)
        return result


def decode_tokens(tokenizer, res_tokens):
    for i in range(len(res_tokens)):
        if type(res_tokens[i]) != int:
            res_tokens[i] = int(res_tokens[i][0])

    res_str = tokenizer.decode(res_tokens, skip_special_tokens=True)
    return res_str


def generate_token(h2ogpt_shark_model, model, tokenizer, **generate_kwargs):
    del generate_kwargs["max_time"]
    generate_kwargs["input_ids"] = generate_kwargs["input_ids"].to(
        device=tensor_device
    )
    generate_kwargs["attention_mask"] = generate_kwargs["attention_mask"].to(
        device=tensor_device
    )
    truncated_input_ids = []
    stopping_criteria = generate_kwargs["stopping_criteria"]

    generation_config_ = GenerationConfig.from_model_config(model.config)
    generation_config = copy.deepcopy(generation_config_)
    model_kwargs = generation_config.update(**generate_kwargs)

    logits_processor = LogitsProcessorList()
    stopping_criteria = (
        stopping_criteria
        if stopping_criteria is not None
        else StoppingCriteriaList()
    )

    eos_token_id = generation_config.eos_token_id
    generation_config.pad_token_id = eos_token_id

    (
        inputs_tensor,
        model_input_name,
        model_kwargs,
    ) = model._prepare_model_inputs(
        None, generation_config.bos_token_id, model_kwargs
    )

    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs[
        "output_hidden_states"
    ] = generation_config.output_hidden_states
    model_kwargs["use_cache"] = generation_config.use_cache

    input_ids = (
        inputs_tensor
        if model_input_name == "input_ids"
        else model_kwargs.pop("input_ids")
    )

    input_ids_seq_length = input_ids.shape[-1]

    generation_config.max_length = (
        generation_config.max_new_tokens + input_ids_seq_length
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    )

    logits_warper = model._get_logits_warper(generation_config)

    (
        input_ids,
        model_kwargs,
    ) = model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,  # 1
        is_encoder_decoder=model.config.is_encoder_decoder,  # False
        **model_kwargs,
    )

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = (
        torch.tensor(eos_token_id).to(device=tensor_device)
        if eos_token_id is not None
        else None
    )

    pad_token_id = generation_config.pad_token_id
    eos_token_id = eos_token_id

    output_scores = generation_config.output_scores  # False
    return_dict_in_generate = (
        generation_config.return_dict_in_generate  # False
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(
        input_ids.shape[0],
        dtype=torch.long,
        device=input_ids.device,
    )

    timesRan = 0
    import time

    start = time.time()
    print("\n")

    res_tokens = []
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )

        outputs = h2ogpt_shark_model.forward(
            model_inputs["input_ids"], model_inputs["attention_mask"]
        )

        if args.precision == "fp16":
            outputs = outputs.to(dtype=torch.float32)
        next_token_logits = outputs

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError(
                    "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                )
            next_token = next_token * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)

        model_kwargs["past_key_values"] = None
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((attention_mask.shape[0], 1)),
                ],
                dim=-1,
            )

        truncated_input_ids.append(input_ids[:, 0])
        input_ids = input_ids[:, 1:]
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, 1:]

        new_word = tokenizer.decode(
            next_token.cpu().numpy(),
            add_special_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        res_tokens.append(next_token)
        if new_word == "<0x0A>":
            print("\n", end="", flush=True)
        else:
            print(f"{new_word}", end=" ", flush=True)

        part_str = decode_tokens(tokenizer, res_tokens)
        yield part_str

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_token.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
            # stop when each sentence is finished
            if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores
            ):
                break
        timesRan = timesRan + 1

    end = time.time()
    print(
        "\n\nTime taken is {:.2f} seconds/token\n".format(
            (end - start) / timesRan
        )
    )

    torch.cuda.empty_cache()
    gc.collect()

    res_str = decode_tokens(tokenizer, res_tokens)
    yield res_str


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

        return_dict = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "attention_mask": attention_mask,
        }
        return_dict = {**return_dict, **generate_kwargs}
        return return_dict
