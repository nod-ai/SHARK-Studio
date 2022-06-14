from iree import runtime as ireert
from iree.tf.support import module_utils
from iree.compiler import tf as tfc
import sys
from absl import app

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
bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32)
]


class BertModule(tf.Module):

    def __init__(self):
        super(BertModule, self).__init__()
        dict_outputs = False
        test_network = networks.BertEncoder(vocab_size=vocab_size,
                                            num_layers=24,
                                            hidden_size=1024,
                                            num_attention_heads=16,
                                            dict_outputs=dict_outputs)

        # Create a BERT trainer with the created network.
        bert_trainer_model = bert_classifier.BertClassifier(
            test_network, num_classes=NUM_CLASSES)
        bert_trainer_model.summary()

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m = bert_trainer_model
        self.m.predict = lambda x: self.m.call(x, training=False)
        self.m.learn = lambda x, y: self.m.call(x, training=False)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH],
                      dtype=tf.int32),  #input0: input_word_ids
        tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH],
                      dtype=tf.int32),  #input1: input_mask
        tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH],
                      dtype=tf.int32),  #input2: segment_ids
        tf.TensorSpec([BATCH_SIZE], tf.int32)  # input3: labels
    ])
    def learn(self, input_word_ids, input_mask, segment_ids, labels):
        with tf.GradientTape() as tape:
            # Capture the gradients from forward prop...
            inputs = [input_word_ids, input_mask, segment_ids]
            probs = self.m(inputs, training=True)
            loss = self.loss(labels, probs)

        # ...and use them to update the model's weights.
        variables = self.m.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function(input_signature=bert_input)
    def predict(self, input_word_ids, input_mask, segment_ids):
        inputs = [input_word_ids, input_mask, segment_ids]
        return self.m.predict(inputs)


if __name__ == "__main__":
    # BertModule()
    # Compile the model using IREE
    compiler_module = tfc.compile_module(BertModule(),
                                         exported_names=["learn"],
                                         import_only=True)
    # Save module as MLIR file in a directory
    ARITFACTS_DIR = os.getcwd()
    mlir_path = os.path.join(ARITFACTS_DIR, "model.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")
