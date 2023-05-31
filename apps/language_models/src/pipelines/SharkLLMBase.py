from abc import ABC, abstractmethod


class SharkLLMBase(ABC):
    def __init__(
        self, model_name, hf_model_path=None, max_num_tokens=512
    ) -> None:
        self.model_name = model_name
        self.hf_model_path = hf_model_path
        self.max_num_tokens = max_num_tokens
        self.shark_model = None
        self.device = "cpu"
        self.precision = "fp32"

    @classmethod
    @abstractmethod
    def compile(self):
        pass

    @classmethod
    @abstractmethod
    def generate(self, prompt):
        pass

    @classmethod
    @abstractmethod
    def generate_new_token(self, params):
        pass

    @classmethod
    @abstractmethod
    def get_tokenizer(self):
        pass

    @classmethod
    @abstractmethod
    def get_src_model(self):
        pass

    def load_init_from_config(self):
        pass
