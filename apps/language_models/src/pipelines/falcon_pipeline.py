from apps.language_models.src.model_wrappers.falcon_model import FalconModel
from apps.language_models.src.model_wrappers.falcon_sharded_model import (
    WordEmbeddingsLayer,
    CompiledWordEmbeddingsLayer,
    LNFEmbeddingLayer,
    CompiledLNFEmbeddingLayer,
    LMHeadEmbeddingLayer,
    CompiledLMHeadEmbeddingLayer,
    DecoderLayer,
    CompiledDecoderLayer,
    ShardedFalconModel,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.utils import (
    get_vmfb_from_path,
)
from io import BytesIO
from pathlib import Path
from contextlib import redirect_stdout
from shark.shark_downloader import download_public_file
from shark.shark_importer import import_with_fx, save_mlir
from shark.shark_inference import SharkInference
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from transformers.generation import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)
import copy

import re
import torch
import torch_mlir
import os
import argparse

parser = argparse.ArgumentParser(
    prog="falcon runner",
    description="runs a falcon model",
)

parser.add_argument(
    "--falcon_variant_to_use", default="7b", help="7b, 40b, 180b"
)
parser.add_argument(
    "--precision", "-p", default="fp16", choices=["fp32", "fp16", "int4"]
)
parser.add_argument("--device", "-d", default="cuda", help="vulkan, cpu, cuda")
parser.add_argument(
    "--falcon_vmfb_path", default=None, help="path to falcon's vmfb"
)
parser.add_argument(
    "--falcon_mlir_path",
    default=None,
    help="path to falcon's mlir file",
)
parser.add_argument(
    "--use_precompiled_model",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="use the precompiled vmfb",
)
parser.add_argument(
    "--load_mlir_from_shark_tank",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="download precompile mlir from shark tank",
)
parser.add_argument(
    "--cli",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Run model in cli mode",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    default=None,
    help="Specify your own huggingface authentication token for falcon-180B model.",
)
parser.add_argument(
    "-s",
    "--sharded",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Run model as sharded",
)


class ShardedFalcon(SharkLLMBase):
    def __init__(
        self,
        model_name,
        hf_model_path="tiiuae/falcon-7b-instruct",
        hf_auth_token: str = None,
        max_num_tokens=150,
        device="cuda",
        precision="fp32",
        falcon_mlir_path=None,
        falcon_vmfb_path=None,
        debug=False,
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        print("hf_model_path: ", self.hf_model_path)

        if "40b" in self.model_name:
            raise NotImplementedError(
                "Sharded Falcon not supported for 40b variant"
            )

        if (
            "180b" in self.model_name
            and precision != "int4"
            and hf_auth_token == None
        ):
            raise ValueError(
                """ HF auth token required for falcon-180b. Pass it using
                --hf_auth_token flag. You can ask for the access to the model
                here: https://huggingface.co/tiiuae/falcon-180B-chat."""
            )
        self.hf_auth_token = hf_auth_token
        self.max_padding_length = 100
        self.device = device
        self.precision = precision
        self.falcon_vmfb_path = falcon_vmfb_path
        self.falcon_mlir_path = falcon_mlir_path
        self.debug = debug
        self.tokenizer = self.get_tokenizer()
        self.src_model = self.get_src_model()
        self.shark_model = self.compile()

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_path,
            trust_remote_code=True,
            token=self.hf_auth_token,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 11
        return tokenizer

    def get_src_model(self):
        print("Loading src model: ", self.model_name)
        kwargs = {
            "torch_dtype": torch.float,
            "trust_remote_code": True,
            "token": self.hf_auth_token,
        }
        if self.precision == "int4":
            quantization_config = GPTQConfig(bits=4, disable_exllama=True)
            kwargs["quantization_config"] = quantization_config
            kwargs["load_gptq_on_cpu"] = True
            kwargs["device_map"] = "cpu" if self.device == "cpu" else "cuda:0"
        falcon_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path, **kwargs
        )
        if self.precision == "int4":
            falcon_model = falcon_model.to(torch.float32)
        return falcon_model

    def compile_layer(self, layer, falconCompileInput, layer_id):
        self.falcon_mlir_path = Path(
            f"falcon_{args.falcon_variant_to_use}_layer_{layer_id}_{self.precision}.mlir"
        )
        self.falcon_vmfb_path = Path(
            f"falcon_{args.falcon_variant_to_use}_layer_{layer_id}_{self.precision}_{self.device}.vmfb"
        )

        if args.use_precompiled_model:
            if not self.falcon_vmfb_path.exists():
                # Downloading VMFB from shark_tank
                print(f"[DEBUG] Trying to download vmfb from shark_tank")
                download_public_file(
                    f"gs://shark_tank/falcon/sharded/falcon_{args.falcon_variant_to_use}/vmfb/"
                    + str(self.falcon_vmfb_path),
                    self.falcon_vmfb_path.absolute(),
                    single_file=True,
                )
            vmfb = get_vmfb_from_path(
                self.falcon_vmfb_path, self.device, "linalg"
            )
            if vmfb is not None:
                return vmfb

        print(f"[DEBUG] vmfb not found at {self.falcon_vmfb_path.absolute()}")
        if self.falcon_mlir_path.exists():
            print(f"[DEBUG] mlir found at {self.falcon_mlir_path.absolute()}")
            with open(self.falcon_mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            mlir_generated = False
            print(
                f"[DEBUG] mlir not found at {self.falcon_mlir_path.absolute()}"
            )
            if args.load_mlir_from_shark_tank:
                # Downloading MLIR from shark_tank
                print(f"[DEBUG] Trying to download mlir from shark_tank")
                download_public_file(
                    f"gs://shark_tank/falcon/sharded/falcon_{args.falcon_variant_to_use}/mlir/"
                    + str(self.falcon_mlir_path),
                    self.falcon_mlir_path.absolute(),
                    single_file=True,
                )
                if self.falcon_mlir_path.exists():
                    print(
                        f"[DEBUG] mlir found at {self.falcon_mlir_path.absolute()}"
                    )
                    with open(self.falcon_mlir_path, "rb") as f:
                        bytecode = f.read()
                    mlir_generated = True

            if not mlir_generated:
                print(f"[DEBUG] generating MLIR locally")
                if layer_id == "word_embeddings":
                    f16_input_mask = [False]
                elif layer_id in ["ln_f", "lm_head"]:
                    f16_input_mask = [True]
                elif type(layer_id) == int:
                    f16_input_mask = [True, False]
                else:
                    raise ValueError("Unsupported layer: ", layer_id)

                print(f"[DEBUG] generating torchscript graph")
                ts_graph = import_with_fx(
                    layer,
                    falconCompileInput,
                    is_f16=True,
                    f16_input_mask=f16_input_mask,
                    mlir_type="torchscript",
                    is_gptq=True,
                )
                del layer

                print(f"[DEBUG] generating torch mlir")
                module = torch_mlir.compile(
                    ts_graph,
                    falconCompileInput,
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

                f_ = open(self.falcon_mlir_path, "wb")
                f_.write(bytecode)
                print("Saved falcon mlir at ", str(self.falcon_mlir_path))
                f_.close()
                del bytecode

        shark_module = SharkInference(
            mlir_module=self.falcon_mlir_path,
            device=self.device,
            mlir_dialect="linalg",
        )
        path = shark_module.save_module(
            self.falcon_vmfb_path.parent.absolute(),
            self.falcon_vmfb_path.stem,
            extra_args=[
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ]
            + [
                "--iree-llvmcpu-use-fast-min-max-ops",
            ]
            if self.precision == "int4"
            else [],
            debug=self.debug,
        )
        print("Saved falcon vmfb at ", str(path))
        shark_module.load_module(path)

        return shark_module

    def compile(self):
        sample_input_ids = torch.zeros([100], dtype=torch.int64)
        sample_attention_mask = torch.zeros(
            [1, 1, 100, 100], dtype=torch.float32
        )
        if "7b" in self.model_name:
            num_in_features = 4544
        else:
            num_in_features = 14848
            sample_attention_mask = sample_attention_mask.to(dtype=torch.bool)

        sample_hidden_states = torch.zeros(
            [1, 100, num_in_features], dtype=torch.float32
        )

        lm_head = LMHeadEmbeddingLayer(self.src_model.lm_head)
        print("Compiling Layer lm_head")
        shark_lm_head = self.compile_layer(
            lm_head, [sample_hidden_states], "lm_head"
        )
        shark_lm_head = CompiledLMHeadEmbeddingLayer(shark_lm_head)

        word_embedding = WordEmbeddingsLayer(
            self.src_model.transformer.word_embeddings
        )
        print("Compiling Layer word_embeddings")
        shark_word_embedding = self.compile_layer(
            word_embedding, [sample_input_ids], "word_embeddings"
        )
        shark_word_embedding = CompiledWordEmbeddingsLayer(
            shark_word_embedding
        )

        ln_f = LNFEmbeddingLayer(self.src_model.transformer.ln_f)
        print("Compiling Layer ln_f")
        shark_ln_f = self.compile_layer(ln_f, [sample_hidden_states], "ln_f")
        shark_ln_f = CompiledLNFEmbeddingLayer(shark_ln_f)

        shark_layers = []
        for i in range(len(self.src_model.transformer.h)):
            print("Compiling Layer {}".format(i))
            layer_i = self.src_model.transformer.h[i]
            pytorch_layer_i = DecoderLayer(layer_i)
            shark_module = self.compile_layer(
                pytorch_layer_i,
                [sample_hidden_states, sample_attention_mask],
                i,
            )
            shark_layer_i = CompiledDecoderLayer(shark_module)
            shark_layers.append(shark_layer_i)

        sharded_model = ShardedFalconModel(
            self.src_model,
            shark_layers,
            shark_word_embedding,
            shark_ln_f,
            shark_lm_head,
        )
        return sharded_model

    def generate(self, prompt):
        model_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_padding_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        model_inputs["prompt_text"] = prompt

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generate_kwargs = {
            "max_length": self.max_num_tokens,
            "do_sample": True,
            "top_k": 10,
            "num_return_sequences": 1,
            "eos_token_id": 11,
        }
        generate_kwargs["input_ids"] = input_ids
        generate_kwargs["attention_mask"] = attention_mask
        generation_config_ = GenerationConfig.from_model_config(
            self.src_model.config
        )
        generation_config = copy.deepcopy(generation_config_)
        model_kwargs = generation_config.update(**generate_kwargs)

        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()

        eos_token_id = generation_config.eos_token_id
        generation_config.pad_token_id = eos_token_id

        (
            inputs_tensor,
            model_input_name,
            model_kwargs,
        ) = self.src_model._prepare_model_inputs(
            None, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

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

        self.logits_processor = self.src_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        self.stopping_criteria = self.src_model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )

        self.logits_warper = self.src_model._get_logits_warper(
            generation_config
        )

        (
            self.input_ids,
            self.model_kwargs,
        ) = self.src_model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,  # 1
            is_encoder_decoder=self.src_model.config.is_encoder_decoder,  # False
            **model_kwargs,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id_tensor = (
            torch.tensor(eos_token_id) if eos_token_id is not None else None
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
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        all_text = prompt

        for i in range(self.max_num_tokens - 1):
            next_token = self.generate_new_token()
            new_word = self.tokenizer.decode(
                next_token.cpu().numpy(),
                add_special_tokens=False,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            all_text = all_text + new_word

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
                    or self.stopping_criteria(input_ids, self.scores)
                ):
                    break

        torch.cuda.empty_cache()
        gc.collect()

        return all_text

    def generate_new_token(self):
        model_inputs = self.src_model.prepare_inputs_for_generation(
            self.input_ids, **self.model_kwargs
        )
        outputs = self.shark_model.forward(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
        )
        if self.precision in ["fp16", "int4"]:
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

        self.input_ids = self.input_ids[:, 1:]
        self.model_kwargs["attention_mask"] = self.model_kwargs[
            "attention_mask"
        ][:, 1:]

        return next_token


class UnshardedFalcon(SharkLLMBase):
    def __init__(
        self,
        model_name,
        hf_model_path="tiiuae/falcon-7b-instruct",
        hf_auth_token: str = "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",
        max_num_tokens=150,
        device="cuda",
        precision="fp32",
        falcon_mlir_path=None,
        falcon_vmfb_path=None,
        debug=False,
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        print("hf_model_path: ", self.hf_model_path)

        if "180b" in self.model_name and hf_auth_token == None:
            raise ValueError(
                """ HF auth token required for falcon-180b. Pass it using
                --hf_auth_token flag. You can ask for the access to the model
                here: https://huggingface.co/tiiuae/falcon-180B-chat."""
            )
        self.hf_auth_token = hf_auth_token
        self.max_padding_length = 100
        self.device = device
        self.precision = precision
        self.falcon_vmfb_path = falcon_vmfb_path
        self.falcon_mlir_path = falcon_mlir_path
        self.debug = debug
        self.tokenizer = self.get_tokenizer()
        self.src_model = self.get_src_model()
        self.shark_model = self.compile()

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_path,
            trust_remote_code=True,
            token=self.hf_auth_token,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 11
        return tokenizer

    def get_src_model(self):
        print("Loading src model: ", self.model_name)
        kwargs = {
            "torch_dtype": torch.float,
            "trust_remote_code": True,
            "token": self.hf_auth_token,
        }
        if self.precision == "int4":
            quantization_config = GPTQConfig(bits=4, disable_exllama=True)
            kwargs["quantization_config"] = quantization_config
            kwargs["load_gptq_on_cpu"] = True
            kwargs["device_map"] = "cpu" if self.device == "cpu" else "cuda:0"
        falcon_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path, **kwargs
        )
        if self.precision == "int4":
            falcon_model = falcon_model.to(torch.float32)
        return falcon_model

    def compile(self):
        if args.use_precompiled_model:
            if not self.falcon_vmfb_path.exists():
                # Downloading VMFB from shark_tank
                download_public_file(
                    "gs://shark_tank/falcon/"
                    + "falcon_"
                    + args.falcon_variant_to_use
                    + "_"
                    + self.precision
                    + "_"
                    + self.device
                    + ".vmfb",
                    self.falcon_vmfb_path.absolute(),
                    single_file=True,
                )
            vmfb = get_vmfb_from_path(
                self.falcon_vmfb_path, self.device, "linalg"
            )
            if vmfb is not None:
                return vmfb

        print(f"[DEBUG] vmfb not found at {self.falcon_vmfb_path.absolute()}")
        if self.falcon_mlir_path.exists():
            print(f"[DEBUG] mlir found at {self.falcon_mlir_path.absolute()}")
            with open(self.falcon_mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            mlir_generated = False
            print(
                f"[DEBUG] mlir not found at {self.falcon_mlir_path.absolute()}"
            )
            if args.load_mlir_from_shark_tank:
                # Downloading MLIR from shark_tank
                print(f"[DEBUG] Trying to download mlir from shark_tank")
                download_public_file(
                    "gs://shark_tank/falcon/"
                    + "falcon_"
                    + args.falcon_variant_to_use
                    + "_"
                    + self.precision
                    + ".mlir",
                    self.falcon_mlir_path.absolute(),
                    single_file=True,
                )
                if self.falcon_mlir_path.exists():
                    print(
                        f"[DEBUG] mlir found at {self.falcon_mlir_path.absolute()}"
                    )
                    mlir_generated = True

            if not mlir_generated:
                print(f"[DEBUG] generating MLIR locally")
                compilation_input_ids = torch.randint(
                    low=1, high=10000, size=(1, 100)
                )
                compilation_attention_mask = torch.ones(
                    1, 100, dtype=torch.int64
                )
                falconCompileInput = (
                    compilation_input_ids,
                    compilation_attention_mask,
                )
                model = FalconModel(self.src_model)

                print(f"[DEBUG] generating torchscript graph")
                ts_graph = import_with_fx(
                    model,
                    falconCompileInput,
                    is_f16=self.precision in ["fp16", "int4"],
                    f16_input_mask=[False, False],
                    mlir_type="torchscript",
                    is_gptq=self.precision == "int4",
                )
                del model
                print(f"[DEBUG] generating torch mlir")

                module = torch_mlir.compile(
                    ts_graph,
                    [*falconCompileInput],
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

                f_ = open(self.falcon_mlir_path, "wb")
                f_.write(bytecode)
                print("Saved falcon mlir at ", str(self.falcon_mlir_path))
                f_.close()
                del bytecode

        shark_module = SharkInference(
            mlir_module=self.falcon_mlir_path,
            device=self.device,
            mlir_dialect="linalg",
        )
        path = shark_module.save_module(
            self.falcon_vmfb_path.parent.absolute(),
            self.falcon_vmfb_path.stem,
            extra_args=[
                "--iree-vm-target-truncate-unsupported-floats",
                "--iree-codegen-check-ir-before-llvm-conversion=false",
                "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            ]
            + [
                "--iree-llvmcpu-use-fast-min-max-ops",
            ]
            if self.precision == "int4"
            else [],
            debug=self.debug,
        )
        print("Saved falcon vmfb at ", str(path))
        shark_module.load_module(path)

        return shark_module

    def generate(self, prompt):
        model_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_padding_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        model_inputs["prompt_text"] = prompt

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask", None)

        # Allow empty prompts
        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        generate_kwargs = {
            "max_length": self.max_num_tokens,
            "do_sample": True,
            "top_k": 10,
            "num_return_sequences": 1,
            "eos_token_id": 11,
        }
        generate_kwargs["input_ids"] = input_ids
        generate_kwargs["attention_mask"] = attention_mask
        generation_config_ = GenerationConfig.from_model_config(
            self.src_model.config
        )
        generation_config = copy.deepcopy(generation_config_)
        model_kwargs = generation_config.update(**generate_kwargs)

        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()

        eos_token_id = generation_config.eos_token_id
        generation_config.pad_token_id = eos_token_id

        (
            inputs_tensor,
            model_input_name,
            model_kwargs,
        ) = self.src_model._prepare_model_inputs(
            None, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

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

        self.logits_processor = self.src_model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        self.stopping_criteria = self.src_model._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
        )

        self.logits_warper = self.src_model._get_logits_warper(
            generation_config
        )

        (
            self.input_ids,
            self.model_kwargs,
        ) = self.src_model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,  # 1
            is_encoder_decoder=self.src_model.config.is_encoder_decoder,  # False
            **model_kwargs,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id_tensor = (
            torch.tensor(eos_token_id) if eos_token_id is not None else None
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
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )

        all_text = prompt

        for i in range(self.max_num_tokens - 1):
            next_token = self.generate_new_token()
            new_word = self.tokenizer.decode(
                next_token.cpu().numpy(),
                add_special_tokens=False,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            all_text = all_text + new_word

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
                    or self.stopping_criteria(input_ids, self.scores)
                ):
                    break

        torch.cuda.empty_cache()
        gc.collect()

        return all_text

    def generate_new_token(self):
        model_inputs = self.src_model.prepare_inputs_for_generation(
            self.input_ids, **self.model_kwargs
        )
        outputs = torch.from_numpy(
            self.shark_model(
                "forward",
                (model_inputs["input_ids"], model_inputs["attention_mask"]),
            )
        )
        if self.precision in ["fp16", "int4"]:
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

        self.input_ids = self.input_ids[:, 1:]
        self.model_kwargs["attention_mask"] = self.model_kwargs[
            "attention_mask"
        ][:, 1:]

        return next_token


if __name__ == "__main__":
    args = parser.parse_args()

    falcon_mlir_path = (
        Path(
            "falcon_"
            + args.falcon_variant_to_use
            + "_"
            + args.precision
            + ".mlir"
        )
        if args.falcon_mlir_path is None
        else Path(args.falcon_mlir_path)
    )
    falcon_vmfb_path = (
        Path(
            "falcon_"
            + args.falcon_variant_to_use
            + "_"
            + args.precision
            + "_"
            + args.device
            + ".vmfb"
        )
        if args.falcon_vmfb_path is None
        else Path(args.falcon_vmfb_path)
    )

    if args.precision == "int4":
        if args.falcon_variant_to_use == "180b":
            hf_model_path_value = "TheBloke/Falcon-180B-Chat-GPTQ"
        else:
            hf_model_path_value = (
                "TheBloke/falcon-"
                + args.falcon_variant_to_use
                + "-instruct-GPTQ"
            )
    else:
        if args.falcon_variant_to_use == "180b":
            hf_model_path_value = "tiiuae/falcon-180B-chat"
        else:
            hf_model_path_value = (
                "tiiuae/falcon-" + args.falcon_variant_to_use + "-instruct"
            )

    if not args.sharded:
        falcon = UnshardedFalcon(
            model_name="falcon_" + args.falcon_variant_to_use,
            hf_model_path=hf_model_path_value,
            device=args.device,
            precision=args.precision,
            falcon_mlir_path=falcon_mlir_path,
            falcon_vmfb_path=falcon_vmfb_path,
        )
    else:
        falcon = ShardedFalcon(
            model_name="falcon_" + args.falcon_variant_to_use,
            hf_model_path=hf_model_path_value,
            device=args.device,
            precision=args.precision,
        )

    import gc

    default_prompt_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
    continue_execution = True

    print("\n-----\nScript executing for the following config: \n")
    print("Falcon Model: ", falcon.model_name)
    print("Precision:    ", args.precision)
    print("Device:       ", args.device)

    while continue_execution:
        use_default_prompt = input(
            "\nDo you wish to use the default prompt text? Y/N ?: "
        )
        if use_default_prompt in ["Y", "y"]:
            prompt = default_prompt_text
        else:
            prompt = input("Please enter the prompt text: ")
        print("\nPrompt Text: ", prompt)

        prompt_template = f"""A helpful assistant who helps the user with any questions asked.
        User: {prompt}
        Assistant:"""

        res_str = falcon.generate(prompt_template)
        torch.cuda.empty_cache()
        gc.collect()
        print(
            "\n\n-----\nHere's the complete formatted result: \n\n",
            res_str,
        )
        continue_execution = input(
            "\nDo you wish to run script one more time? Y/N ?: "
        )
        continue_execution = (
            True if continue_execution in ["Y", "y"] else False
        )
