from transformers import TFAutoModelForMaskedLM
import tensorflow as tf
from shark.shark_inference import SharkInference

# Create a set of input signature.
inputs_signature = [
    tf.TensorSpec(shape=[1, 512], dtype=tf.int32),
]


class AutoModelMaskedLM(tf.Module):
    def __init__(self, model_name):
        super(AutoModelMaskedLM, self).__init__()
        self.m = TFAutoModelForMaskedLM.from_pretrained(model_name, output_attentions=False)
        self.m.predict = lambda x: self.m(input_ids=x)

    @tf.function(input_signature=inputs_signature)
    def forward(self, input_ids):
        return self.m.predict(input_ids)


fail_models = ["microsoft/deberta-base", "google/rembert", "google/tapas-base"]

supported_models = [
    "albert-base-v2",
    "bert-base-uncased",
    "camembert-base",
    "dbmdz/convbert-base-turkish-cased",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
    "hf-internal-testing/tiny-random-flaubert",
    "funnel-transformer/small",
    "microsoft/layoutlm-base-uncased",
    "allenai/longformer-base-4096",
    "google/mobilebert-uncased",
    "microsoft/mpnet-base",
    "roberta-base",
    "xlm-roberta-base",
]

if __name__ == "__main__":
    inputs = tf.random.uniform(shape=[1, 512], maxval=3, dtype=tf.int32, seed=10)

    for model_name in supported_models:
        print(f"Running model: {model_name}")
        shark_module = SharkInference(AutoModelMaskedLM(model_name), (inputs,))
        shark_module.set_frontend("tensorflow")
        shark_module.compile()
        print(shark_module.forward((inputs,)))
