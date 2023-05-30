import torch
import torch_mlir
from transformers import AutoTokenizer, StoppingCriteria, AutoModelForCausalLM
from io import BytesIO
from pathlib import Path
from apps.language_models.utils import (
    get_torch_mlir_module_bytecode,
    get_vmfb_from_path,
)
from apps.language_models.src.pipelines.SharkLLMBase import SharkLLMBase
from apps.language_models.src.model_wrappers.stablelm_model import (
    StableLMModel,
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
        max_num_tokens=512,
        device="cuda",
        precision="fp32",
    ) -> None:
        super().__init__(model_name, hf_model_path, max_num_tokens)
        self.max_sequence_len = 256
        self.device = device
        self.precision = precision
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
            self.hf_model_path, torch_dtype=torch.float32
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
        if mlir_path.exists():
            with open(mlir_path, "rb") as f:
                bytecode = f.read()
        else:
            model = StableLMModel(self.get_src_model())
            model_inputs = self.get_model_inputs()
            ts_graph = get_torch_mlir_module_bytecode(model, model_inputs)
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
        f_ = open(tmp_model_name + ".mlir", "wb")
        f_.write(bytecode)
        print("Saved mlir")
        f_.close()

        from shark.shark_inference import SharkInference

        shark_module = SharkInference(
            mlir_module=bytecode, device=self.device, mlir_dialect="tm_tensor"
        )
        shark_module.compile()

        path = shark_module.save_module(
            vmfb_path.parent.absolute(), vmfb_path.stem
        )
        print("Saved vmfb at ", str(path))

        return shark_module

    def get_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(self.hf_model_path)
        tok.add_special_tokens({"pad_token": "<PAD>"})
        # print("[DEBUG] Sucessfully loaded the tokenizer to the memory")
        return tok

    def generate(self, prompt):
        words_list = []
        for i in range(self.max_num_tokens):
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


# Initialize a StopOnTokens object
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""
