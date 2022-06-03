from iree import runtime as ireert
from iree.tf.support import module_utils
from iree.compiler import tf as tfc
from iree.compiler import compile_str
import sys
from absl import app
import time

import numpy as np
import os
import tempfile
import tensorflow as tf

from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_classifier

vocab_size = 100
NUM_CLASSES = 5
SEQUENCE_LENGTH = 512
BATCH_SIZE = 1
# Create a set of 2-dimensional inputs
bert_input = [tf.TensorSpec(shape=[BATCH_SIZE,SEQUENCE_LENGTH],dtype=tf.int32),
            tf.TensorSpec(shape=[BATCH_SIZE,SEQUENCE_LENGTH], dtype=tf.int32),
            tf.TensorSpec(shape=[BATCH_SIZE,SEQUENCE_LENGTH], dtype=tf.int32)]

class BertModule(tf.Module):
    def __init__(self):
        super(BertModule, self).__init__()
        dict_outputs = False
        test_network = networks.BertEncoder(
            vocab_size=vocab_size, num_layers=24, hidden_size=1024, num_attention_heads=16, dict_outputs=dict_outputs)

        # Create a BERT trainer with the created network.
        bert_trainer_model = bert_classifier.BertClassifier(
            test_network, num_classes=NUM_CLASSES)
        bert_trainer_model.summary()

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m = bert_trainer_model
        self.m.predict = lambda x: self.m.call(x, training=False)
        self.predict = tf.function(
            input_signature=[bert_input])(self.m.predict)
        self.m.learn = lambda x,y: self.m.call(x, training=False)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

    @tf.function(input_signature=[
        bert_input,  # inputs
        tf.TensorSpec(shape=[BATCH_SIZE], dtype=tf.int32)  # labels
    ])
    def learn(self,inputs,labels):
        with tf.GradientTape() as tape:
            # Capture the gradients from forward prop...
            probs = self.m(inputs, training=True)
            loss = self.loss(labels, probs)

        # ...and use them to update the model's weights.
        variables = self.m.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

if __name__ == "__main__":
    # BertModule()
    # Compile the model using IREE
    compiler_module = tfc.compile_module(BertModule(), exported_names = ["learn"], import_only=True)

    # Compile the model using IREE
    backend = "dylib-llvm-aot"
    args = ["--iree-llvm-target-cpu-features=host", "--iree-mhlo-demote-i64-to-i32=false", "--iree-stream-resource-index-bits=64", "--iree-vm-target-index-bits=64"]
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
    predict_sample_input = [np.random.randint(5, size=(BATCH_SIZE,SEQUENCE_LENGTH)), np.random.randint(5, size=(BATCH_SIZE,SEQUENCE_LENGTH)), np.random.randint(5, size=(BATCH_SIZE,SEQUENCE_LENGTH))]
    learn_sample_input = [predict_sample_input, np.random.randint(5, size=(BATCH_SIZE))]
    warmup = 5
    total_iter = 10
    num_iter = total_iter - warmup
    for i in range(10):
        if(i == warmup-1):
            start = time.time()
        print(BertCompiled.learn(predict_sample_input,np.random.randint(5, size=(BATCH_SIZE))))
    end = time.time()
    total_time = end - start
    print("time: "+str(total_time))
    print("time/iter: "+str(total_time/num_iter))
