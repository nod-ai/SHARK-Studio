
from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import sys
from absl import app

import numpy as np
import os
import tempfile
import tensorflow as tf

import time
from transformers import BertModel, BertTokenizer, TFBertModel

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of 2-dimensional inputs
bert_input = [tf.TensorSpec(shape=[BATCH_SIZE,MAX_SEQUENCE_LENGTH],dtype=tf.int32),
            tf.TensorSpec(shape=[BATCH_SIZE,MAX_SEQUENCE_LENGTH], dtype=tf.int32),
            tf.TensorSpec(shape=[BATCH_SIZE,MAX_SEQUENCE_LENGTH], dtype=tf.int32)]

class BertModule(tf.Module):
    def __init__(self):
        super(BertModule, self).__init__()
        # Create a BERT trainer with the created network.
        self.m = TFBertModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased", from_pt=True)

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m.predict = lambda x,y,z: self.m.call(input_ids=x, attention_mask=y, token_type_ids=z, training=False)

    @tf.function(input_signature=bert_input)
    def predict(self, input_ids, attention_mask, token_type_ids):
        return self.m.predict(input_ids, attention_mask, token_type_ids)

if __name__ == "__main__":
    # Prepping Data
    tokenizer = BertTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_SEQUENCE_LENGTH)
    for key in encoded_input:
        encoded_input[key] = tf.expand_dims(tf.convert_to_tensor(encoded_input[key]),0)

    # Compile the model using IREE
    compiler_module = tfc.compile_module(BertModule(), exported_names = ["predict"], import_only=True)

    # Compile the model using IREE
    backend = "dylib-llvm-aot"
    args = ["--iree-llvm-target-cpu-features=host", "--iree-mhlo-demote-i64-to-i32=false", "--iree-flow-demote-i64-to-i32"]
    backend_config = "dylib"
    #backend = "cuda"
    #backend_config = "cuda"
    #args = ["--iree-cuda-llvm-target-arch=sm_80", "--iree-hal-cuda-disable-loop-nounroll-wa", "--iree-enable-fusion-with-reduction-ops"]
    flatbuffer_blob = compile_str(compiler_module, target_backends=[backend], extra_args=args, input_type="mhlo")
    #flatbuffer_blob = compile_str(compiler_module, target_backends=["dylib-llvm-aot"])

    # Save module as MLIR file in a directory
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config("dylib",tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    BertCompiled = ctx.modules.module
    result = BertCompiled.predict(encoded_input["input_ids"], encoded_input["attention_mask"], encoded_input["token_type_ids"])
    print(result)
