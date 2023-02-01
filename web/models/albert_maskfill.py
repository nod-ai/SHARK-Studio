from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from shark.shark_inference import SharkInference
from shark.shark_importer import SharkImporter
import numpy as np

################################## Albert Module #########################


class AlbertModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained("albert-base-v2")
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits


################################## Preprocessing inputs ####################

DEBUG = False
compiled_module = {}
compiled_module["tokenizer"] = AutoTokenizer.from_pretrained("albert-base-v2")


def preprocess_data(text):
    global compiled_module

    # Preparing Data
    tokenizer = compiled_module["tokenizer"]
    encoded_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = (encoded_inputs["input_ids"], encoded_inputs["attention_mask"])
    return inputs


def top5_possibilities(text, inputs, token_logits, log_write):
    global DEBUG
    global compiled_module

    if DEBUG:
        log_write.write("Retrieving top 5 possible outcomes.\n")
    tokenizer = compiled_module["tokenizer"]
    mask_id = torch.where(inputs[0] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_id, :]
    percentage = torch.nn.functional.softmax(mask_token_logits, dim=1)[0]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    top5 = {}
    for token in top_5_tokens:
        label = text.replace(tokenizer.mask_token, tokenizer.decode(token))
        top5[label] = percentage[token].item()
    if DEBUG:
        log_write.write("Done.\n")
    return top5


##############################################################################


def albert_maskfill_inf(masked_text, device):
    global DEBUG
    global compiled_module

    DEBUG = False
    log_write = open(r"logs/albert_maskfill_log.txt", "w")
    if log_write:
        DEBUG = True

    inputs = preprocess_data(masked_text)
    if device not in compiled_module.keys():
        if DEBUG:
            log_write.write("Compiling the Albert Maskfill module.\n")
        mlir_importer = SharkImporter(
            AlbertModule(),
            inputs,
            frontend="torch",
        )
        minilm_mlir, func_name = mlir_importer.import_mlir(
            is_dynamic=False, tracing_required=True
        )
        shark_module = SharkInference(
            minilm_mlir, func_name, mlir_dialect="linalg", device=device
        )
        shark_module.compile()
        compiled_module[device] = shark_module
        if DEBUG:
            log_write.write("Compilation successful.\n")

    token_logits = torch.tensor(compiled_module[device].forward(inputs))
    output = top5_possibilities(masked_text, inputs, token_logits, log_write)
    log_write.close()

    std_output = ""
    with open(r"logs/albert_maskfill_log.txt", "r") as log_read:
        std_output = log_read.read()

    return output, std_output
