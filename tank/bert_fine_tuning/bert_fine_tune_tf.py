import numpy as np

from iree import runtime as ireert
from iree.compiler import tf as tfc
from iree.compiler import compile_str

import tensorflow as tf

try:
    import tensorflow_datasets as tfds
    import tensorflow_models as tfm
    from official.nlp.modeling import layers
    from official.nlp.modeling import networks
    from official.nlp.modeling.models import bert_classifier
except ModuleNotFoundError:
    print(
        "tensorflow models or datasets not found please run the following command with your virtual env active:\npip install tf-models-nightly tf-datasets"
    )
import json
import time
import os

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)
vocab_size = 100
NUM_CLASSES = 2
SEQUENCE_LENGTH = 128
BATCH_SIZE = 1
# Create a set of 2-dimensional inputs
bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, SEQUENCE_LENGTH], dtype=tf.int32),
]


class BertModule(tf.Module):
    def __init__(self):
        super(BertModule, self).__init__()
        dict_outputs = False

        bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")

        config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
        encoder_config = tfm.nlp.encoders.EncoderConfig(
            {"type": "bert", "bert": config_dict}
        )
        bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)

        # Create a BERT trainer with the created network.
        bert_trainer_model = bert_classifier.BertClassifier(
            bert_encoder, num_classes=NUM_CLASSES
        )
        bert_trainer_model.summary()
        checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
        checkpoint.read(
            os.path.join(gs_folder_bert, "bert_model.ckpt")
        ).assert_consumed()

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m = bert_trainer_model
        self.m.predict = lambda x: self.m.call(x, training=False)
        self.predict = tf.function(input_signature=[bert_input])(
            self.m.predict
        )
        self.m.learn = lambda x, y: self.m.call(x, training=False)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

    @tf.function(
        input_signature=[
            bert_input,  # inputs
            tf.TensorSpec(shape=[BATCH_SIZE], dtype=tf.int32),  # labels
        ]
    )
    def learn(self, inputs, labels):
        with tf.GradientTape() as tape:
            # Capture the gradients from forward prop...
            probs = self.m.call(inputs, training=True)
            loss = self.loss(labels, probs)

        # ...and use them to update the model's weights.
        variables = self.m.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


if __name__ == "__main__":
    glue, info = tfds.load("glue/mrpc", with_info=True, batch_size=BATCH_SIZE)

    tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
        vocab_file=os.path.join(gs_folder_bert, "vocab.txt"), lower_case=True
    )

    max_seq_length = SEQUENCE_LENGTH

    packer = tfm.nlp.layers.BertPackInputs(
        seq_length=max_seq_length,
        special_tokens_dict=tokenizer.get_special_tokens_dict(),
    )

    class BertInputProcessor(tf.keras.layers.Layer):
        def __init__(self, tokenizer, packer):
            super().__init__()
            self.tokenizer = tokenizer
            self.packer = packer

        def call(self, inputs):
            tok1 = self.tokenizer(inputs["sentence1"])
            tok2 = self.tokenizer(inputs["sentence2"])

            packed = self.packer([tok1, tok2])

            if "label" in inputs:
                return packed, inputs["label"]
            else:
                return packed

    bert_inputs_processor = BertInputProcessor(tokenizer, packer)
    glue_train = glue["train"].map(bert_inputs_processor).prefetch(1)
    glue_validation = glue["validation"].map(bert_inputs_processor).prefetch(1)
    glue_test = glue["test"].map(bert_inputs_processor).prefetch(1)

    # base tensorflow model
    bert_model = BertModule()

    # Compile the model using IREE
    compiler_module = tfc.compile_module(
        bert_model, exported_names=["learn"], import_only=True
    )

    # choose from dylib-llvm-aot or cuda
    backend = "dylib-llvm-aot"
    if backend == "dylib-llvm-aot":
        args = [
            "--iree-llvm-target-cpu-features=host",
            "--iree-mhlo-demote-i64-to-i32=false",
            "--iree-flow-demote-i64-to-i32",
        ]
        backend_config = "dylib"

    else:
        backend_config = "cuda"
        args = [
            "--iree-cuda-llvm-target-arch=sm_80",
            "--iree-hal-cuda-disable-loop-nounroll-wa",
            "--iree-enable-fusion-with-reduction-ops",
        ]

    flatbuffer_blob = compile_str(
        compiler_module,
        target_backends=[backend],
        extra_args=args,
        input_type="mhlo",
    )

    # Save module as MLIR file in a directory
    vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
    tracer = ireert.Tracer(os.getcwd())
    config = ireert.Config("local-sync", tracer)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    BertCompiled = ctx.modules.module

    # compare output losses:
    start = time.time()
    iterations = 100
    for i in range(iterations):
        example_inputs, example_labels = next(iter(glue_train))
        example_labels = tf.cast(example_labels, tf.int32)
        example_inputs = [value for key, value in example_inputs.items()]

        # iree version
        #        iree_loss = BertCompiled.learn(
        #            example_inputs, example_labels
        #        ).to_host()

        # base tensorflow
        tf_loss = np.array(bert_model.learn(example_inputs, example_labels))
    #        print(np.allclose(iree_loss, tf_loss))
    end = time.time()
    total = (end - start) * 1000
    print("total time/iter (ms): " + str(total / iterations))
