from iree import runtime as ireert
from iree.compiler import tf as tfc
from absl import app

import numpy as np
import os
import tensorflow as tf

from transformers import BertModel, BertTokenizer, TFBertModel

SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of 2-dimensional inputs
bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32)
]


class BertModule(tf.Module):

    def __init__(self):
        super(BertModule, self).__init__()
        # Create a BERT trainer with the created network.
        self.m = TFBertModel.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased", from_pt=True)

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m.predict = lambda x, y, z: self.m.call(
            input_ids=x, attention_mask=y, token_type_ids=z, training=False)

    @tf.function(input_signature=bert_input)
    def predict(self, input_word_ids, input_mask, segment_ids):
        return self.m.predict(input_word_ids, input_mask, segment_ids)


if __name__ == "__main__":
    # BertModule()
    # Compile the model using IREE
    compiler_module = tfc.compile_module(BertModule(),
                                         exported_names=["predict"],
                                         import_only=True)
    # Save module as MLIR file in a directory
    ARITFACTS_DIR = os.getcwd()
    mlir_path = os.path.join(ARITFACTS_DIR, "model.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")
