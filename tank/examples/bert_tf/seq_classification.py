#!/usr/bin/env python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from amdshark.amdshark_inference import AMDSharkInference
from amdshark.parser import amdshark_args
import argparse


seq_parser = argparse.ArgumentParser(
    description="AMDShark Sequence Classification."
)
seq_parser.add_argument(
    "--hf_model_name",
    type=str,
    default="bert-base-uncased",
    help="Hugging face model to run sequence classification.",
)

seq_args, unknown = seq_parser.parse_known_args()


BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 16

# Create a set of input signature.
inputs_signature = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
]

# For supported models please see here:
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.TFAutoModelForSequenceClassification


def preprocess_input(text="This is just used to compile the model"):
    tokenizer = AutoTokenizer.from_pretrained(seq_args.hf_model_name)
    inputs = tokenizer(
        text,
        padding="max_length",
        return_tensors="tf",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    return inputs


class SeqClassification(tf.Module):
    def __init__(self, model_name):
        super(SeqClassification, self).__init__()
        self.m = TFAutoModelForSequenceClassification.from_pretrained(
            model_name, output_attentions=False, num_labels=2
        )
        self.m.predict = lambda x, y: self.m(input_ids=x, attention_mask=y)[0]

    @tf.function(input_signature=inputs_signature, jit_compile=True)
    def forward(self, input_ids, attention_mask):
        return tf.math.softmax(
            self.m.predict(input_ids, attention_mask), axis=-1
        )


if __name__ == "__main__":
    inputs = preprocess_input()
    amdshark_module = AMDSharkInference(
        SeqClassification(seq_args.hf_model_name),
        (inputs["input_ids"], inputs["attention_mask"]),
    )
    amdshark_module.set_frontend("tensorflow")
    amdshark_module.compile()
    print(f"Model has been successfully compiled on {amdshark_args.device}")

    while True:
        input_text = input(
            "Enter the text to classify (press q or nothing to exit): "
        )
        if not input_text or input_text == "q":
            break
        inputs = preprocess_input(input_text)
        print(
            amdshark_module.forward(
                (inputs["input_ids"], inputs["attention_mask"])
            )
        )
