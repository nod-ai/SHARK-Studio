import torch
import torch_mlir
from transformers import AutoTokenizer, StoppingCriteria, AutoModelForCausalLM
from io import BytesIO
from pathlib import Path
from apps.language_models.utils import (
    get_vmfb_from_path,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.src.model_wrappers.stablelm_model import (
    StableLMModel,
)
import argparse

parser = argparse.ArgumentParser(
    prog="stablelm runner",
    description="runs a StableLM model",
)

parser.add_argument(
    "--precision", "-p", default="fp16", choices=["fp32", "fp16", "int4"]
)
parser.add_argument("--device", "-d", default="cuda", help="vulkan, cpu, cuda")
parser.add_argument(
    "--stablelm_vmfb_path", default=None, help="path to StableLM's vmfb"
)
parser.add_argument(
    "--stablelm_mlir_path",
    default=None,
    help="path to StableLM's mlir file",
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
    "--hf_auth_token",
    type=str,
    default=None,
    help="Specify your own huggingface authentication token for stablelm-3B model.",
)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class SharkStableLM(SharkLLMBase):
    def __init__(
        self,
        model_name,
        hf_model_path="stabilityai/stablelm-tuned-alpha-3b",
        max_num_tokens=256,
        device="cuda",
        precision="fp32",
        debug="False",
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_len = 256
        self.device = device
        self.precision = precision
        self.debug = debug
        self.tokenizer = self.get_tokenizer()
        self.shark_model = self.compile()

    def shouldStop(self, tokens):
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if tokens[0][-1] == stop_id:
                return True
        return False

    def get_src_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            use_auth_token="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",
        )
        return model

    def get_model_inputs(self):
        input_ids = torch.randint(3, (1, self.max_sequence_len))
        attention_mask = torch.randint(3, (1, self.max_sequence_len))
        return input_ids, attention_mask

    def compile(self):
        tmp_model_name = (
            f"stableLM_linalg_{self.precision}_seqLen{self.max_sequence_len}"
        )

        # device = "cuda"  # "cpu"
        # TODO: vmfb and mlir name should include precision and device
        model_vmfb_name = None
        vmfb_path = (
            Path(tmp_model_name + f"_{self.device}.vmfb")
            if model_vmfb_name is None
            else Path(model_vmfb_name)
        )
        shark_module = get_vmfb_from_path(
            vmfb_path, self.device, mlir_dialect="tm_tensor"
        )
        if shark_module is not None:
            return shark_module

        mlir_path = Path(tmp_model_name + ".mlir")
        print(
            f"[DEBUG] mlir path {mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}"
        )
        if not mlir_path.exists():
            model = StableLMModel(self.get_src_model())
            model_inputs = self.get_model_inputs()
            from shark.shark_importer import import_with_fx

            ts_graph = import_with_fx(
                model,
                model_inputs,
                is_f16=True if self.precision in ["fp16", "int4"] else False,
                precision=self.precision,
                f16_input_mask=[False, False],
                mlir_type="torchscript",
            )
            module = torch_mlir.compile(
                ts_graph,
                [*model_inputs],
                torch_mlir.OutputType.LINALG_ON_TENSORS,
                use_tracing=False,
                verbose=False,
            )
            bytecode_stream = BytesIO()
            module.operation.write_bytecode(bytecode_stream)
            bytecode = bytecode_stream.getvalue()
            f_ = open(mlir_path, "wb")
            f_.write(bytecode)
            print("Saved mlir at: ", mlir_path)
            f_.close()
            del bytecode

        from shark.shark_inference import SharkInference

        shark_module = SharkInference(
            mlir_module=mlir_path, device=self.device, mlir_dialect="tm_tensor"
        )
        shark_module.compile()

        path = shark_module.save_module(
            vmfb_path.parent.absolute(), vmfb_path.stem, debug=self.debug
        )
        print("Saved vmfb at ", str(path))

        return shark_module

    def get_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(
            self.hf_model_path,
            use_auth_token="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",
        )
        tok.add_special_tokens({"pad_token": "<PAD>"})
        # print("[DEBUG] Sucessfully loaded the tokenizer to the memory")
        return tok

    def generate(self, prompt):
        words_list = []
        import time

        start = time.time()
        count = 0
        for i in range(self.max_num_tokens):
            count = count + 1
            params = {
                "new_text": prompt,
            }

            generated_token_op = self.generate_new_token(params)

            detok = generated_token_op["detok"]
            stop_generation = generated_token_op["stop_generation"]

            if stop_generation:
                break

            print(detok, end="", flush=True)  # this is for CLI and DEBUG
            words_list.append(detok)
            if detok == "":
                break
            prompt = prompt + detok
        end = time.time()
        print(
            "\n\nTime  taken is {:.2f} tokens/second\n".format(
                count / (end - start)
            )
        )
        return words_list

    def generate_new_token(self, params):
        new_text = params["new_text"]
        model_inputs = self.tokenizer(
            [new_text],
            padding="max_length",
            max_length=self.max_sequence_len,
            truncation=True,
            return_tensors="pt",
        )
        sum_attentionmask = torch.sum(model_inputs.attention_mask)
        output = self.shark_model(
            "forward", [model_inputs.input_ids, model_inputs.attention_mask]
        )
        output = torch.from_numpy(output)
        next_toks = torch.topk(output, 1)
        stop_generation = False
        if self.shouldStop(next_toks.indices):
            stop_generation = True
        new_token = next_toks.indices[0][int(sum_attentionmask) - 1]
        detok = self.tokenizer.decode(
            new_token,
            skip_special_tokens=True,
        )
        ret_dict = {
            "new_token": new_token,
            "detok": detok,
            "stop_generation": stop_generation,
        }
        return ret_dict


if __name__ == "__main__":
    args = parser.parse_args()

    stable_lm = SharkStableLM(
        model_name="StableLM",
        hf_model_path="stabilityai/stablelm-3b-4e1t",
        device=args.device,
        precision=args.precision,
    )

    default_prompt_text = "The weather is always wonderful"
    continue_execution = True

    print("\n-----\nScript executing for the following config: \n")
    print("StableLM Model: ", stable_lm.hf_model_path)
    print("Precision:      ", args.precision)
    print("Device:         ", args.device)

    while continue_execution:
        use_default_prompt = input(
            "\nDo you wish to use the default prompt text? Y/N ?: "
        )
        if use_default_prompt in ["Y", "y"]:
            prompt = default_prompt_text
        else:
            prompt = input("Please enter the prompt text: ")
        print("\nPrompt Text: ", prompt)

        res_str = stable_lm.generate(prompt)
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        print(
            "\n\n-----\nHere's the complete formatted result: \n\n",
            prompt + "".join(res_str),
        )
        continue_execution = input(
            "\nDo you wish to run script one more time? Y/N ?: "
        )
        continue_execution = (
            True if continue_execution in ["Y", "y"] else False
        )
