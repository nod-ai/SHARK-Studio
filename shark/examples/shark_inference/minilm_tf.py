import tensorflow as tf
from transformers import BertModel, BertTokenizer, TFBertModel
from shark.shark_inference import SharkInference

MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 1

# Create a set of 2-dimensional inputs
bert_input = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32)
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
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m.predict(input_ids, attention_mask, token_type_ids)


if __name__ == "__main__":
    # Prepping Data
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

    shark_module = SharkInference(
        BertModule(),
        (encoded_input["input_ids"], encoded_input["attention_mask"],
         encoded_input["token_type_ids"]))
    shark_module.set_frontend("tensorflow")
    shark_module.compile()

    print(
        shark_module.forward(
            (encoded_input["input_ids"], encoded_input["attention_mask"],
             encoded_input["token_type_ids"])))
