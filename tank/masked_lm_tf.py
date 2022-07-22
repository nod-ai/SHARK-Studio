from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf

visible_default = tf.config.list_physical_devices("GPU")
try:
    tf.config.set_visible_devices([], "GPU")
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != "GPU"
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# The max_sequence_length is set small for testing purpose.
BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 16

# Create a set of input signature.
inputs_signature = [
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
    tf.TensorSpec(shape=[BATCH_SIZE, MAX_SEQUENCE_LENGTH], dtype=tf.int32),
]

# For supported models please see here:
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.TFAutoModelForCasualLM


def preprocess_input(
    model_name, text="This is just used to compile the model"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(
        text,
        padding="max_length",
        return_tensors="tf",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    return inputs


class MaskedLM(tf.Module):
    def __init__(self, model_name):
        super(MaskedLM, self).__init__()
        self.m = TFAutoModelForMaskedLM.from_pretrained(
            model_name, output_attentions=False, num_labels=2
        )
        self.m.predict = lambda x, y: self.m(input_ids=x, attention_mask=y)[0]

    @tf.function(input_signature=inputs_signature)
    def forward(self, input_ids, attention_mask):
        return self.m.predict(input_ids, attention_mask)


def get_causal_lm_model(hf_name, text="Hello, this is the default text."):
    #    gpus = tf.config.experimental.list_physical_devices("GPU")
    #    for gpu in gpus:
    #        tf.config.experimental.set_memory_growth(gpu, True)
    model = MaskedLM(hf_name)
    encoded_input = preprocess_input(hf_name, text)
    test_input = (encoded_input["input_ids"], encoded_input["attention_mask"])
    actual_out = model.forward(*test_input)
    return model, test_input, actual_out
