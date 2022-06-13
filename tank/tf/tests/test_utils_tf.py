from shark.shark_inference import SharkInference
from shark.iree_utils import check_device_drivers

import tensorflow as tf
import numpy as np
from transformers import AutoModelForSequenceClassification, BertTokenizer, TFBertModel
import importlib

##################### Tensorflow Hugging Face LM Models ###################################
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of 2-dimensional inputs
tf_bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32)
]

class TFHuggingFaceLanguage(tf.Module):

    def __init__(self, hf_model_name):
        super(TFHuggingFaceLanguage, self).__init__()
        # Create a BERT trainer with the created network.
        self.m = TFBertModel.from_pretrained(
            hf_model_name, from_pt=True)

        # Invoke the trainer model on the inputs. This causes the layer to be built.
        self.m.predict = lambda x, y, z: self.m.call(
            input_ids=x, attention_mask=y, token_type_ids=z, training=False)

    @tf.function(input_signature=tf_bert_input)
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m.predict(input_ids, attention_mask, token_type_ids)


def get_TFhf_model(name):
    model = TFHuggingFaceLanguage(name)
    tokenizer = BertTokenizer.from_pretrained(
    "microsoft/MiniLM-L12-H384-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text,
                              padding='max_length',
                              truncation=True,
                              max_length=MAX_SEQUENCE_LENGTH)
    for key in encoded_input:
        encoded_input[key] = tf.expand_dims(
            tf.convert_to_tensor(encoded_input[key]), 0)
    test_input = (encoded_input["input_ids"], encoded_input["attention_mask"],
         encoded_input["token_type_ids"])
    actual_out = model.forward(*test_input)
    return model, test_input, actual_out


# Utility function for comparing two tensors (tensorflow).
def compare_tensors_tf(tf_tensor, numpy_tensor):
    # setting the absolute and relative tolerance
    rtol = 1e-02
    atol = 1e-03
    tf_to_numpy = tf_tensor.pooler_output.numpy()
    return np.allclose(tf_to_numpy, numpy_tensor, rtol, atol)


